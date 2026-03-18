# file: src/pipelines/loki_waveform_stacking_pipelines.py
from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterator
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from obspy import Stream

from common.config import (
	LokiWaveformStackingInputs,
	LokiWaveformStackingPipelineConfig,
)
from common.core import load_event_json
from common.json_io import read_json
from common.time_util import get_event_origin_utc, to_utc
from io_util.stream import build_stream_from_downloaded_win32
from loki_tools.build_loki import build_loki_with_header
from loki_tools.prob_stream import build_loki_ps_prob_stream
from pick.eqt_probs import build_probs_by_station
from pipelines.event_filtering import (
	EventContext,
	extract_event_magnitude,
	filter_event_dirs,
)
from waveform.preprocess import (
	preprocess_stream_detrend_bandpass,
	spec_from_inputs,
)

logger = logging.getLogger(__name__)

_PRE_KEYS = {
	'pre_enable',
	'pre_detrend',
	'pre_fstop_lo',
	'pre_fpass_lo',
	'pre_fpass_hi',
	'pre_fstop_hi',
	'pre_gpass',
	'pre_gstop',
	'pre_mad_scale',
	'pre_mad_eps',
	'pre_mad_c',
}

def prepare_win32_stream(
	event_dir: Path,
	*,
	base_fs_hz: int,
	pre_enable: bool,
	pre_spec: dict[str, float | str | bool | None],
) -> Stream:
	st = build_stream_from_downloaded_win32(
		event_dir,
		base_sampling_rate_hz=int(base_fs_hz),
		components_order=('U', 'N', 'E'),
	)

	if pre_enable:
		preprocess_stream_detrend_bandpass(
			st,
			spec=pre_spec,
			fs_expected=float(base_fs_hz),
		)
	return st


def iter_preprocessed_event_streams(
	event_dirs: list[Path],
	inputs: LokiWaveformStackingInputs,
	prepare_stream_fn: Callable[[Stream], Stream],
) -> Iterator[tuple[str, Stream]]:
	pre_enable = bool(inputs.pre_enable)
	pre_spec = spec_from_inputs(inputs)
	base_fs_hz = int(inputs.base_sampling_rate_hz)

	for event_dir in event_dirs:
		event_name = event_dir.name
		st = prepare_win32_stream(
			event_dir,
			base_fs_hz=base_fs_hz,
			pre_enable=pre_enable,
			pre_spec=pre_spec,
		)

		prepared_stream = prepare_stream_fn(st)
		logger.info(
			'prepared stream: event=%s n_traces=%d dir=%s pre=%s',
			event_name,
			len(prepared_stream),
			event_dir,
			'on' if pre_enable else 'off',
		)
		yield event_name, prepared_stream


def _log_db_station_summary(header: object, *, enabled: bool = True) -> set[str]:
	if not enabled:
		return set()
	stations_df = getattr(header, 'stations_df', None)
	if stations_df is None:
		return set()
	db_stas = set(stations_df['station'].astype(str).tolist())
	logger.debug('db stations: %d', len(db_stas))
	return db_stas


def _log_stream_station_overlap(
	st: Stream,
	*,
	evid: str,
	db_stas: set[str],
	cfg: LokiWaveformStackingPipelineConfig,
	enabled: bool = True,
) -> None:
	if not enabled:
		return
	st_stas = sorted(
		{
			str(tr.stats.station)
			for tr in st
			if getattr(tr.stats, 'station', None) is not None
		}
	)
	ch_suf = sorted(
		{
			str(tr.stats.channel)[-1]
			for tr in st
			if getattr(tr.stats, 'channel', None) is not None
		}
	)
	overlap = sorted(set(st_stas) & db_stas)

	logger.debug(
		'event=%s traces=%d stations=%d overlap_db=%d chan_suffixes=%s cfg.comp=%s',
		evid,
		len(st),
		len(st_stas),
		len(overlap),
		ch_suf,
		list(cfg.comp),
	)

	if len(overlap) == 0:
		# ここが真犯人候補
		logger.debug('  example station names (stream): %s', st_stas[:10])
		logger.debug('  example station names (db): %s', sorted(list(db_stas))[:10])


def list_event_dirs_filtered(cfg: LokiWaveformStackingPipelineConfig) -> list[Path]:
	def build_candidates(base: Path) -> list[Path]:
		return sorted(base.glob(cfg.event_glob))

	def read_event_dir(ev_dir: Path) -> EventContext | None:
		event_json = ev_dir / 'event.json'
		if not (ev_dir.is_dir() and event_json.is_file()):
			return None
		ev = load_event_json(ev_dir)
		origin_utc = get_event_origin_utc(ev, event_json_path=event_json)
		return EventContext(
			origin_utc=origin_utc,
			meta=ev,
			meta_path=event_json,
			event_dir=ev_dir,
		)

	def get_event_time(ctx: EventContext) -> pd.Timestamp:
		return ctx.origin_utc

	def extra_filter(ctx: EventContext) -> bool:
		mag = extract_event_magnitude(ctx.meta)
		if mag is None and (cfg.mag_min is not None or cfg.mag_max is not None):
			if cfg.drop_if_mag_missing:
				return False
			return True

		if mag is not None:
			if cfg.mag_min is not None and mag < float(cfg.mag_min):
				return False
			if cfg.mag_max is not None and mag > float(cfg.mag_max):
				return False
		return True

	return filter_event_dirs(
		cfg,
		build_candidates=build_candidates,
		read_event=read_event_dir,
		get_event_time=get_event_time,
		extra_filter=extra_filter,
		empty_error='No event dirs found under: {base} (glob={glob})',
		log_prefix='event filter',
	)


def pipeline_loki_waveform_stacking(
	cfg: LokiWaveformStackingPipelineConfig, inputs: LokiWaveformStackingInputs
) -> None:
	cfg.loki_data_path.mkdir(parents=True, exist_ok=True)
	cfg.loki_output_path.mkdir(parents=True, exist_ok=True)

	event_dirs = list_event_dirs_filtered(cfg)
	streams_by_event: dict[str, Stream] = {}

	for event_dir in event_dirs:
		(cfg.loki_data_path / event_dir.name).mkdir(parents=True, exist_ok=True)

	for event_name, stream in iter_preprocessed_event_streams(
		event_dirs,
		inputs,
		prepare_stream_fn=lambda st: st,
	):
		streams_by_event[event_name] = stream

	l1, header, _header_path = build_loki_with_header(cfg)

	inputs_dict = asdict(inputs)
	# ★ 前処理キーは LOKI に渡さない
	loki_kwargs = {k: v for k, v in inputs_dict.items() if k not in _PRE_KEYS}

	db_stas = _log_db_station_summary(header)

	for evid, st in streams_by_event.items():
		_log_stream_station_overlap(
			st,
			evid=evid,
			db_stas=db_stas,
			cfg=cfg,
		)
	l1.location(
		extension=cfg.extension,
		comp=list(cfg.comp),
		precision=cfg.precision,
		search=cfg.search,
		streams_by_event=streams_by_event,
		**loki_kwargs,
	)


def pipeline_loki_waveform_stacking_eqt(
	cfg: LokiWaveformStackingPipelineConfig,
	inputs: LokiWaveformStackingInputs,
	*,
	eqt_weights: str = 'original',
	eqt_in_samples: int = 6000,
	eqt_overlap: int = 3000,
	eqt_batch_size: int = 64,
	channel_prefix: str = 'HH',
) -> None:
	"""EqTransformer の P/S 確率系列（direct_input）を LOKI に入力して震源推定する。

	方針:
	- P/S 確率は pick.eqt_probs.build_probs_by_station で生成
	- LOKI へ渡すパラメータは direct_input 前提で最小化（proc 側 runner を正とする）
	- vfunc/hfunc/derivative 等は渡さない（結果がブレやすい）
	"""
	cfg.loki_data_path.mkdir(parents=True, exist_ok=True)
	cfg.loki_output_path.mkdir(parents=True, exist_ok=True)

	event_dirs = list_event_dirs_filtered(cfg)
	streams_by_event: dict[str, Stream] = {}

	for event_dir in event_dirs:
		(cfg.loki_data_path / event_dir.name).mkdir(parents=True, exist_ok=True)

	fs_expected = float(inputs.base_sampling_rate_hz)

	def build_eqt_prob_stream(st: Stream) -> Stream:
		probs_by_sta = build_probs_by_station(
			st,
			fs=fs_expected,
			eqt_weights=str(eqt_weights),
			eqt_in_samples=int(eqt_in_samples),
			eqt_overlap=int(eqt_overlap),
			eqt_batch_size=int(eqt_batch_size),
		)
		return build_loki_ps_prob_stream(
			ref_stream=st,
			probs_by_station=probs_by_sta,
			channel_prefix=str(channel_prefix),
			require_both_ps=True,
		)

	for event_name, stream in iter_preprocessed_event_streams(
		event_dirs,
		inputs,
		prepare_stream_fn=build_eqt_prob_stream,
	):
		streams_by_event[event_name] = stream

	comp = list(getattr(cfg, 'comp', ['P', 'S']))
	if comp != ['P', 'S']:
		logger.warning(
			"cfg.comp is %s, but EqT direct_input assumes ['P','S']",
			comp,
		)

	loki_kwargs: dict[str, object] = {
		'npr': int(getattr(inputs, 'npr', 2)),
		'model': str(getattr(inputs, 'model', 'jma2001')),
	}

	l1, _header, _header_path = build_loki_with_header(cfg)

	l1.location(
		extension=cfg.extension,
		comp=['P', 'S'],
		precision=cfg.precision,
		search=cfg.search,
		streams_by_event=streams_by_event,
		**loki_kwargs,
	)


_DAS_EVENTNUM_RE = re.compile(r'(\d+)$')


def _das_event_time_utc_from_meta(meta: dict, *, meta_path: Path) -> pd.Timestamp:
	raw = meta.get('file_time_utc')
	if raw is None:
		raw = meta.get('window_start_utc')
	if raw is None:
		raise ValueError(
			f'meta.json missing file_time_utc/window_start_utc: {meta_path}'
		)

	ts = pd.to_datetime(raw)
	if pd.isna(ts):
		raise ValueError(f'failed to parse event time in {meta_path}: {raw!r}')

	# metaはUTCのはず。もしnaiveでもUTCとして扱う。
	return to_utc(pd.Timestamp(ts), naive_tz='UTC')


def _das_event_sort_key(ev_dir: Path) -> tuple[int, str]:
	m = _DAS_EVENTNUM_RE.search(ev_dir.name)
	if m is None:
		return (10**18, ev_dir.name)
	return (int(m.group(1)), ev_dir.name)


def list_event_dirs_filtered_forge_das(
	cfg: LokiWaveformStackingPipelineConfig,
) -> list[Path]:
	"""cut_events_fromzarr_for_loki.py の生成物(event_XXXXXX/meta.json等)を列挙してフィルタする。"""
	# DASにはmagが無いので、magフィルタ指定は事故源。明示的に止める。
	if cfg.mag_min is not None or cfg.mag_max is not None:
		raise ValueError(
			'ForgeDAS events (meta.json) do not contain magnitude; mag_min/mag_max is unsupported.'
		)

	def build_candidates(base: Path) -> list[Path]:
		return sorted(
			[p for p in base.glob(cfg.event_glob) if p.is_dir()],
			key=_das_event_sort_key,
		)

	def read_event_dir(ev_dir: Path) -> EventContext | None:
		waveform_path = ev_dir / 'waveform.npy'
		meta_path = ev_dir / 'meta.json'
		stations_path = ev_dir / 'stations.csv'
		if not (
			waveform_path.is_file() and meta_path.is_file() and stations_path.is_file()
		):
			return None
		meta = read_json(meta_path, encoding='utf-8', errors='strict')
		t0 = _das_event_time_utc_from_meta(meta, meta_path=meta_path)
		return EventContext(
			origin_utc=t0,
			meta=meta,
			meta_path=meta_path,
			event_dir=ev_dir,
		)

	def get_event_time(ctx: EventContext) -> pd.Timestamp:
		return ctx.origin_utc

	return filter_event_dirs(
		cfg,
		build_candidates=build_candidates,
		read_event=read_event_dir,
		get_event_time=get_event_time,
		extra_filter=None,
		empty_error='No DAS event dirs found under: {base} (glob={glob})',
		log_prefix='das event filter',
	)
