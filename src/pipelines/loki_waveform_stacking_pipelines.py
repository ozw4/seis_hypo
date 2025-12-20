# file: src/pipelines/loki_waveform_stacking_pipelines.py
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loki.loki import Loki
from obspy import Stream
from seisbench.models import EQTransformer

from common.config import (
	LokiWaveformStackingInputs,
	LokiWaveformStackingPipelineConfig,
)
from common.core import load_event_json
from common.time_util import to_utc
from io_util.stream import build_stream_from_downloaded_win32
from waveform.filters import zscore_channelwise
from waveform.preprocess import DetrendBandpassSpec, preprocess_stream_detrend_bandpass

_DEFAULT_PRE = {
	'pre_detrend': 'linear',
	'pre_fstop_lo': 0.5,
	'pre_fpass_lo': 1.0,
	'pre_fpass_hi': 20.0,
	'pre_fstop_hi': 30.0,
	'pre_gpass': 1.0,
	'pre_gstop': 40.0,
	'pre_mad_scale': False,
	'pre_mad_eps': 1.0,
	'pre_mad_c': 6.0,
}

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


def _parse_cfg_time_utc(raw: str | None) -> pd.Timestamp | None:
	if raw is None:
		return None
	ts = pd.to_datetime(raw)
	if pd.isna(ts):
		raise ValueError(f'failed to parse time: {raw}')
	# Config times are treated as JST if timezone is omitted.
	return to_utc(ts, naive_tz='Asia/Tokyo')


def _get_event_origin_utc(ev: dict, *, event_json_path: Path) -> pd.Timestamp:
	origin_jst = ev.get('origin_time_jst')
	origin_other = ev.get('origin_time')
	origin_raw = origin_jst if origin_jst is not None else origin_other
	if origin_raw is None:
		raise ValueError(f'missing origin_time(_jst) in {event_json_path}')

	origin = pd.to_datetime(origin_raw)
	if pd.isna(origin):
		raise ValueError(f'failed to parse origin_time in {event_json_path}')

	# If origin_time_jst is present and timezone is omitted, interpret it as JST.
	# Otherwise (origin_time), interpret naive as UTC by default.
	naive_tz = 'Asia/Tokyo' if origin_jst is not None else 'UTC'
	return to_utc(origin, naive_tz=naive_tz)


def list_event_dirs_filtered(cfg: LokiWaveformStackingPipelineConfig) -> list[Path]:
	base = Path(cfg.base_input_dir)
	if not base.is_dir():
		raise FileNotFoundError(f'base_input_dir not found: {base}')

	t_start = _parse_cfg_time_utc(cfg.origin_time_start)
	t_end = _parse_cfg_time_utc(cfg.origin_time_end)
	if t_start is not None and t_end is not None and t_end < t_start:
		raise ValueError(
			f'origin_time_end must be >= origin_time_start: {t_end} < {t_start}'
		)

	candidates = sorted(base.glob(cfg.event_glob))

	dirs: list[Path] = []
	dropped = 0

	for p in candidates:
		event_json = p / 'event.json'
		if not (p.is_dir() and event_json.is_file()):
			continue

		ev = load_event_json(p)
		origin_utc = _get_event_origin_utc(ev, event_json_path=event_json)

		if t_start is not None and origin_utc < t_start:
			dropped += 1
			continue
		if t_end is not None and origin_utc > t_end:
			dropped += 1
			continue

		mag = None
		extra = ev.get('extra', {})
		if not isinstance(extra, dict):
			extra = {}

		for key in ('mag1', 'magnitude', 'mag'):
			if key in ev:
				mag = ev[key]
				break
			if key in extra:
				mag = extra[key]
				break

		if mag is None and (cfg.mag_min is not None or cfg.mag_max is not None):
			if cfg.drop_if_mag_missing:
				dropped += 1
				continue
		if mag is not None:
			mag_f = float(mag)
			if cfg.mag_min is not None and mag_f < float(cfg.mag_min):
				dropped += 1
				continue
			if cfg.mag_max is not None and mag_f > float(cfg.mag_max):
				dropped += 1
				continue

		dirs.append(p)

	if cfg.max_events is not None and cfg.max_events > 0:
		dirs = dirs[: int(cfg.max_events)]

	if not dirs:
		raise ValueError(f'No event dirs found under: {base} (glob={cfg.event_glob})')

	print(f'event filter: total={len(candidates)} kept={len(dirs)} dropped={dropped}')
	return dirs


def pipeline_loki_waveform_stacking(
	cfg: LokiWaveformStackingPipelineConfig, inputs: LokiWaveformStackingInputs
) -> None:
	cfg.loki_data_path.mkdir(parents=True, exist_ok=True)
	cfg.loki_output_path.mkdir(parents=True, exist_ok=True)

	event_dirs = list_event_dirs_filtered(cfg)
	streams_by_event: dict[str, Stream] = {}

	# 前処理specは inputs から作る（yaml管理）。属性が無い場合はデフォルト値を使用。
	pre_spec = DetrendBandpassSpec(
		detrend=getattr(inputs, 'pre_detrend', _DEFAULT_PRE['pre_detrend']),
		fstop_lo=getattr(inputs, 'pre_fstop_lo', _DEFAULT_PRE['pre_fstop_lo']),
		fpass_lo=getattr(inputs, 'pre_fpass_lo', _DEFAULT_PRE['pre_fpass_lo']),
		fpass_hi=getattr(inputs, 'pre_fpass_hi', _DEFAULT_PRE['pre_fpass_hi']),
		fstop_hi=getattr(inputs, 'pre_fstop_hi', _DEFAULT_PRE['pre_fstop_hi']),
		gpass=getattr(inputs, 'pre_gpass', _DEFAULT_PRE['pre_gpass']),
		gstop=getattr(inputs, 'pre_gstop', _DEFAULT_PRE['pre_gstop']),
		mad_scale=bool(getattr(inputs, 'pre_mad_scale', _DEFAULT_PRE['pre_mad_scale'])),
		mad_eps=float(getattr(inputs, 'pre_mad_eps', _DEFAULT_PRE['pre_mad_eps'])),
		mad_c=float(getattr(inputs, 'pre_mad_c', _DEFAULT_PRE['pre_mad_c'])),
	)
	fs_expected = float(inputs.base_sampling_rate_hz)

	for event_dir in event_dirs:
		event_name = event_dir.name
		(cfg.loki_data_path / event_name).mkdir(parents=True, exist_ok=True)

		st = build_stream_from_downloaded_win32(
			event_dir,
			base_sampling_rate_hz=int(inputs.base_sampling_rate_hz),
			components_order=('U', 'N', 'E'),
		)

		if inputs.pre_enable:
			preprocess_stream_detrend_bandpass(
				st,
				spec=pre_spec,
				fs_expected=fs_expected,
			)

		streams_by_event[event_name] = st
		print(
			f'prepared stream: event={event_name} n_traces={len(st)} dir={event_dir} '
			f'(pre={"on" if inputs.pre_enable else "off"})'
		)

	l1 = Loki(
		str(cfg.loki_data_path),
		str(cfg.loki_output_path),
		str(cfg.loki_db_path),
		str(cfg.loki_hdr_filename),
		mode='locator',
	)

	inputs_dict = asdict(inputs)
	# ★ 前処理キーは LOKI に渡さない
	loki_kwargs = {k: v for k, v in inputs_dict.items() if k not in _PRE_KEYS}
	from pathlib import Path

	from loki_tools.loki_parse import parse_loki_header

	header_path = Path(cfg.loki_db_path) / Path(cfg.loki_hdr_filename)
	header = parse_loki_header(header_path)
	db_stas = set(header.stations_df['station'].astype(str).tolist())

	print(f'[DBG] db stations: {len(db_stas)}')

	for evid, st in streams_by_event.items():
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

		print(
			f'[DBG] event={evid} traces={len(st)} '
			f'stations={len(st_stas)} overlap_db={len(overlap)} '
			f'chan_suffixes={ch_suf} cfg.comp={list(cfg.comp)}'
		)

		if len(overlap) == 0:
			# ここが真犯人候補
			print(f'[DBG]   example station names (stream): {st_stas[:10]}')
			print(f'[DBG]   example station names (db): {sorted(list(db_stas))[:10]}')
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
	"""EqTransformer の P/S 確率系列を LOKI に直接入力して震源推定する。

	- 各 station の (U,N,E) 波形から EqTransformer で P/S/D を推論
	- P を comp='P'（LOKI の V 側）, S を comp='S'（LOKI の H 側）として Stream 化
	- LOKI は comp=['P','S'] で起動

	注意:
		本関数は seisbench + torch が必要。
	"""

	def _trace_comp(tr) -> str:
		cha = getattr(tr.stats, 'channel', None)
		if cha is None:
			raise ValueError('trace.stats.channel missing')
		return str(cha)[-1]

	def _stream_to_station_zne(st: Stream) -> dict[str, np.ndarray]:
		by: dict[str, dict[str, object]] = {}
		for tr in st:
			sta = getattr(tr.stats, 'station', None)
			if sta is None:
				raise ValueError('trace.stats.station missing')
			comp = _trace_comp(tr)
			if comp not in ('U', 'N', 'E'):
				continue
			d = by.setdefault(str(sta), {})
			if comp in d:
				raise ValueError(f'duplicate trace for station={sta} comp={comp}')
			d[comp] = tr

		out: dict[str, np.ndarray] = {}
		for sta, d in by.items():
			missing = [c for c in ('U', 'N', 'E') if c not in d]
			if missing:
				raise ValueError(f'station={sta} missing components: {missing}')

			tru = d['U']
			tn = d['N']
			te = d['E']
			npts = int(tru.stats.npts)
			if int(tn.stats.npts) != npts or int(te.stats.npts) != npts:
				raise ValueError(f'station={sta} npts mismatch among components')
			if float(tn.stats.delta) != float(tru.stats.delta) or float(
				te.stats.delta
			) != float(tru.stats.delta):
				raise ValueError(f'station={sta} delta mismatch among components')
			if (
				tn.stats.starttime != tru.stats.starttime
				or te.stats.starttime != tru.stats.starttime
			):
				raise ValueError(f'station={sta} starttime mismatch among components')

			zne = np.vstack(
				[
					tru.data.astype(np.float32, copy=False),
					tn.data.astype(np.float32, copy=False),
					te.data.astype(np.float32, copy=False),
				]
			)
			out[sta] = zne
		return out

	_model_cache: dict[tuple[str, int], EQTransformer] = {}

	def _get_model(weights: str, in_samples: int) -> EQTransformer:
		key = (weights, int(in_samples))
		m = _model_cache.get(key)
		if m is not None:
			return m
		m = EQTransformer.from_pretrained(weights)
		if int(m.in_samples) != int(in_samples):
			raise ValueError(
				f'EqT in_samples mismatch: model={m.in_samples} requested={in_samples}'
			)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		m.eval().to(device)
		_model_cache[key] = m
		return m

	def _stack_overlap_max(dst: np.ndarray, src: np.ndarray, start: int) -> None:
		end = start + int(src.shape[-1])
		sl = slice(start, end)
		cur = dst[sl]
		cur_nan = np.isnan(cur)
		src_nan = np.isnan(src)
		if cur_nan.all():
			dst[sl] = src
			return
		out = cur.copy()
		use_src_only = cur_nan & ~src_nan
		out[use_src_only] = src[use_src_only]
		use_both = ~cur_nan & ~src_nan
		out[use_both] = np.maximum(out[use_both], src[use_both])
		dst[sl] = out

	def _infer_probs_1sta(
		zne: np.ndarray,
		*,
		fs: float,
		model: EQTransformer,
		in_samples: int,
		overlap: int,
		batch_size: int,
	) -> dict[str, np.ndarray]:
		fs_model = float(getattr(inputs, 'base_sampling_rate_hz', 100))
		if float(fs) != float(fs_model):
			raise ValueError(f'EqT pipeline expects fs={fs_model}, got fs={fs}')

		C, N = zne.shape
		if C != 3:
			raise ValueError(f'zne must be (3,N), got {zne.shape}')

		L = int(in_samples)
		H = L - int(overlap)
		if H <= 0:
			raise ValueError('eqt_overlap must be smaller than eqt_in_samples')

		det = np.full(N, np.nan, dtype=np.float32)
		p = np.full(N, np.nan, dtype=np.float32)
		s = np.full(N, np.nan, dtype=np.float32)

		device = next(model.parameters()).device

		def _to_tensor(w: np.ndarray) -> torch.Tensor:
			t = torch.from_numpy(w[None, :, :]).to(device)
			return zscore_channelwise(t, axis=-1, eps=1e-6)

		buf: list[tuple[int, torch.Tensor]] = []

		def _flush() -> None:
			if not buf:
				return
			starts, tensors = zip(*buf, strict=False)
			B = torch.cat(list(tensors), dim=0)
			y_det, y_p, y_s = model(B)
			y_det = y_det.detach().cpu().numpy()
			y_p = y_p.detach().cpu().numpy()
			y_s = y_s.detach().cpu().numpy()
			for s0, d0, p0, s0s in zip(starts, y_det, y_p, y_s, strict=False):
				s0i = int(s0)
				_stack_overlap_max(det, d0, s0i)
				_stack_overlap_max(p, p0, s0i)
				_stack_overlap_max(s, s0s, s0i)
			buf.clear()

		with torch.no_grad():
			if N < L:
				w = np.zeros((3, L), dtype=np.float32)
				w[:, :N] = zne[:, :N].astype(np.float32, copy=False)
				buf.append((0, _to_tensor(w)))
				_flush()
			else:
				for start in range(0, N - L + 1, H):
					w = zne[:, start : start + L].astype(np.float32, copy=False)
					buf.append((int(start), _to_tensor(w)))
					if len(buf) >= int(batch_size):
						_flush()
				# 末尾を必ずカバー
				last = N - L
				if last % H != 0:
					w = zne[:, last : last + L].astype(np.float32, copy=False)
					buf.append((int(last), _to_tensor(w)))
				_flush()

		det = np.nan_to_num(det, nan=0.0)
		p = np.nan_to_num(p, nan=0.0)
		s = np.nan_to_num(s, nan=0.0)
		return {'D': det, 'P': p, 'S': s}

	cfg.loki_data_path.mkdir(parents=True, exist_ok=True)
	cfg.loki_output_path.mkdir(parents=True, exist_ok=True)

	event_dirs = list_event_dirs_filtered(cfg)
	streams_by_event: dict[str, Stream] = {}

	pre_spec = DetrendBandpassSpec(
		detrend=getattr(inputs, 'pre_detrend', _DEFAULT_PRE['pre_detrend']),
		fstop_lo=getattr(inputs, 'pre_fstop_lo', _DEFAULT_PRE['pre_fstop_lo']),
		fpass_lo=getattr(inputs, 'pre_fpass_lo', _DEFAULT_PRE['pre_fpass_lo']),
		fpass_hi=getattr(inputs, 'pre_fpass_hi', _DEFAULT_PRE['pre_fpass_hi']),
		fstop_hi=getattr(inputs, 'pre_fstop_hi', _DEFAULT_PRE['pre_fstop_hi']),
		gpass=getattr(inputs, 'pre_gpass', _DEFAULT_PRE['pre_gpass']),
		gstop=getattr(inputs, 'pre_gstop', _DEFAULT_PRE['pre_gstop']),
		mad_scale=bool(getattr(inputs, 'pre_mad_scale', _DEFAULT_PRE['pre_mad_scale'])),
		mad_eps=float(getattr(inputs, 'pre_mad_eps', _DEFAULT_PRE['pre_mad_eps'])),
		mad_c=float(getattr(inputs, 'pre_mad_c', _DEFAULT_PRE['pre_mad_c'])),
	)
	fs_expected = float(inputs.base_sampling_rate_hz)

	model = _get_model(str(eqt_weights), int(eqt_in_samples))

	for event_dir in event_dirs:
		event_name = event_dir.name
		(cfg.loki_data_path / event_name).mkdir(parents=True, exist_ok=True)

		st = build_stream_from_downloaded_win32(
			event_dir,
			base_sampling_rate_hz=int(inputs.base_sampling_rate_hz),
			components_order=('U', 'N', 'E'),
		)

		if inputs.pre_enable:
			preprocess_stream_detrend_bandpass(
				st,
				spec=pre_spec,
				fs_expected=fs_expected,
			)

		zne_by_sta = _stream_to_station_zne(st)
		probs_by_sta: dict[str, dict[str, np.ndarray]] = {}
		for sta, zne in zne_by_sta.items():
			probs = _infer_probs_1sta(
				zne,
				fs=fs_expected,
				model=model,
				in_samples=int(eqt_in_samples),
				overlap=int(eqt_overlap),
				batch_size=int(eqt_batch_size),
			)
			probs_by_sta[sta] = {'P': probs['P'], 'S': probs['S']}

		st_prob = build_loki_ps_prob_stream(
			ref_stream=st,
			probs_by_station=probs_by_sta,
			channel_prefix=str(channel_prefix),
			require_both_ps=True,
		)

		streams_by_event[event_name] = st_prob
		print(
			f'prepared EqT prob stream: event={event_name} n_traces={len(st_prob)} dir={event_dir}'
		)

	l1 = Loki(
		str(cfg.loki_data_path),
		str(cfg.loki_output_path),
		str(cfg.loki_db_path),
		str(cfg.loki_hdr_filename),
		mode='locator',
	)

	inputs_dict = asdict(inputs)
	loki_kwargs = {k: v for k, v in inputs_dict.items() if k not in _PRE_KEYS}

	# EqT 確率は comp=['P','S'] に固定
	l1.location(
		extension=cfg.extension,
		comp=['P', 'S'],
		precision=cfg.precision,
		search=cfg.search,
		streams_by_event=streams_by_event,
		**loki_kwargs,
	)
