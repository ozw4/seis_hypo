# %%
"""Run Loki EqT direct-input locations for Izu2009 GaMMA event dirs."""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from obspy import Stream

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _REPO_ROOT / 'src'

for _path in (_REPO_ROOT, _SRC_DIR):
	_path_str = str(_path)
	if _path_str not in sys.path:
		sys.path.insert(0, _path_str)

from common.config import (  # noqa: E402
	LokiWaveformStackingInputs,
	LokiWaveformStackingPipelineConfig,
)
from common.core import load_event_json  # noqa: E402
from common.json_io import write_json  # noqa: E402
from io_util.stream import build_stream_from_downloaded_win32  # noqa: E402
from loki_tools.build_loki import build_loki_with_header  # noqa: E402
from loki_tools.prob_stream import build_loki_ps_prob_stream  # noqa: E402
from pick.eqt_probs import build_probs_by_station  # noqa: E402
from pipelines.loki_waveform_stacking_pipelines import (  # noqa: E402
	list_event_dirs_filtered,
	prepare_win32_stream,
)
from waveform.preprocess import spec_from_inputs  # noqa: E402

EVENTS_BASE_DIR = _REPO_ROOT / 'proc/izu2009/loki/events_from_gamma'
LOKI_DB_PATH = _REPO_ROOT / 'proc/izu2009/loki/traveltime/db'
LOKI_HDR_FILENAME = 'header.hdr'

LOKI_DATA_PATH = _REPO_ROOT / 'proc/izu2009/loki/loki_data_eqt_gamma'
LOKI_OUTPUT_PATH = _REPO_ROOT / 'proc/izu2009/loki/output_eqt_gamma'
RUN_CONFIG_JSON = _REPO_ROOT / 'proc/izu2009/loki/run_loki_eqt_from_gamma_config.json'

EVENT_GLOB = '[0-9]*'
# Keep the default cheap enough for smoke tests. Use --max-events 50 (or all) for
# the full configured run.
MAX_EVENTS: int | None = 1

EQT_WEIGHT = Path(
	'/workspace/model_weight/eqt/010_Train_EqT_FT-STEAD_rot30_Hinet_selftrain.pth'
)
EQT_IN_SAMPLES = 6000
EQT_OVERLAP = 3000
EQT_BATCH_SIZE = 64
EQT_CHANNEL_PREFIX = 'HH'

BASE_SAMPLING_RATE_HZ = 100
PRE_ENABLE = False

COMP = ('P', 'S')
PRECISION = 'single'
SEARCH = 'classic'
NPR = 2
NTRIAL = 1
MODEL = 'jma2001'
EXPECTED_MIN_TIME_BUF_COUNT = 94

_LOKI_WAVEFORMS_ORIGINAL: type | None = None
_LOKI_DIRECT_STREAMS: dict[object, Stream] = {}


def _repo_rel(path: Path) -> str:
	p = Path(path)
	try:
		return str(p.resolve().relative_to(_REPO_ROOT.resolve()))
	except ValueError:
		return str(p)


def _generated_at_utc() -> str:
	return (
		datetime.now(timezone.utc)
		.replace(microsecond=0)
		.isoformat()
		.replace('+00:00', 'Z')
	)


def _ensure_empty_or_missing_dir(path: Path, label: str) -> None:
	if not path.exists():
		return
	if not path.is_dir():
		raise NotADirectoryError(f'{label} exists but is not a directory: {path}')
	if any(path.iterdir()):
		raise FileExistsError(
			f'{label} already exists and is not empty; remove it explicitly: {path}'
		)


def _collect_time_buf_paths(db_path: Path) -> list[Path]:
	"""Collect Loki/NLL travel-time buffers from supported naming styles."""
	patterns = [
		'*.time.buf',
		'*.time.*.buf',
	]

	paths_by_resolved: dict[Path, Path] = {}
	for pattern in patterns:
		for path in db_path.glob(pattern):
			if path.is_file():
				paths_by_resolved[path.resolve()] = path

	return sorted(paths_by_resolved.values(), key=lambda path: path.name)


def _build_cfg(
	*, max_events: int | None = MAX_EVENTS
) -> LokiWaveformStackingPipelineConfig:
	if max_events is not None and int(max_events) <= 0:
		raise ValueError(f'max_events must be positive or None, got {max_events}')

	return LokiWaveformStackingPipelineConfig(
		base_input_dir=EVENTS_BASE_DIR,
		base_traveltime_dir=LOKI_DB_PATH.parent,
		loki_data_path=LOKI_DATA_PATH,
		loki_output_path=LOKI_OUTPUT_PATH,
		loki_db_path=LOKI_DB_PATH,
		loki_hdr_filename=LOKI_HDR_FILENAME,
		inputs_yaml=RUN_CONFIG_JSON,
		inputs_preset='inline',
		comp=COMP,
		precision=PRECISION,
		search=SEARCH,
		event_glob=EVENT_GLOB,
		max_events=max_events,
	)


def _build_inputs() -> LokiWaveformStackingInputs:
	return LokiWaveformStackingInputs(
		npr=NPR,
		ntrial=NTRIAL,
		model=MODEL,
		base_sampling_rate_hz=BASE_SAMPLING_RATE_HZ,
		pre_enable=PRE_ENABLE,
	)


def _candidate_event_dirs() -> list[Path]:
	if not EVENTS_BASE_DIR.is_dir():
		raise FileNotFoundError(f'events_from_gamma not found: {EVENTS_BASE_DIR}')

	candidates = sorted(p for p in EVENTS_BASE_DIR.glob(EVENT_GLOB) if p.is_dir())
	if not candidates:
		raise FileNotFoundError(
			f'no event directories found under: {EVENTS_BASE_DIR} (glob={EVENT_GLOB})'
		)

	missing_json = [p for p in candidates if not (p / 'event.json').is_file()]
	if missing_json:
		raise FileNotFoundError(
			'event.json missing for event dirs: '
			f'{[_repo_rel(p) for p in missing_json[:20]]}'
		)

	return candidates


def _validate_event_json_groups(event_dirs: list[Path]) -> None:
	for event_dir in event_dirs:
		ev = load_event_json(event_dir)
		win32 = ev.get('win32')
		if not isinstance(win32, dict):
			raise TypeError(f'event.json win32 must be an object: {event_dir}')
		if win32.get('format') != 'groups':
			raise ValueError(f'event.json win32.format must be "groups": {event_dir}')

		groups = win32.get('groups')
		if not isinstance(groups, list) or not groups:
			raise ValueError(
				f'event.json win32.groups must be a non-empty list: {event_dir}'
			)

		for i, group in enumerate(groups):
			if not isinstance(group, dict):
				raise TypeError(
					f'event.json win32.groups[{i}] must be an object: {event_dir}'
				)
			has_ch_file = bool(str(group.get('ch_file', '')).strip())
			ch_files = group.get('ch_files')
			has_ch_files = isinstance(ch_files, list) and bool(ch_files)
			if not (has_ch_file or has_ch_files):
				raise ValueError(
					'event.json win32 group must contain ch_file or ch_files: '
					f'{event_dir} group_index={i}'
				)


def _validate_traveltime_db() -> list[Path]:
	if not LOKI_DB_PATH.is_dir():
		raise FileNotFoundError(
			f'Loki travel-time db directory not found: {LOKI_DB_PATH}'
		)

	header_path = LOKI_DB_PATH / LOKI_HDR_FILENAME
	if not header_path.is_file():
		raise FileNotFoundError(f'Loki header not found: {header_path}')

	time_buf_paths = _collect_time_buf_paths(LOKI_DB_PATH)
	if not time_buf_paths:
		raise FileNotFoundError(
			f'no travel-time .buf files found in: {LOKI_DB_PATH}. '
			'Expected files matching *.time.buf or *.time.*.buf.'
		)
	if len(time_buf_paths) < int(EXPECTED_MIN_TIME_BUF_COUNT):
		raise FileNotFoundError(
			f'expected at least {EXPECTED_MIN_TIME_BUF_COUNT} travel-time buffers, '
			f'got {len(time_buf_paths)} in: {LOKI_DB_PATH}'
		)

	return time_buf_paths


def _validate_eqt_weight() -> None:
	if not EQT_WEIGHT.is_file():
		raise FileNotFoundError(f'EqT weight not found: {EQT_WEIGHT}')


def _validate_first_event_stream(first_event_dir: Path) -> None:
	st = build_stream_from_downloaded_win32(
		first_event_dir,
		base_sampling_rate_hz=BASE_SAMPLING_RATE_HZ,
	)
	if not st:
		raise ValueError(f'empty WIN32 Stream for first event: {first_event_dir}')
	if len(st) <= 0:
		raise ValueError(f'no traces in first event stream: {first_event_dir}')

	bad = [
		f'{tr.id}:{int(tr.stats.npts)}'
		for tr in st
		if int(getattr(tr.stats, 'npts', 0)) <= 0
	]
	if bad:
		raise ValueError(
			'first event stream has traces with npts <= 0: '
			f'{first_event_dir} {bad[:20]}'
		)


def _write_run_config(
	*, event_count: int, time_buf_count: int, max_events: int | None
) -> None:
	config: dict[str, Any] = {
		'generated_at_utc': _generated_at_utc(),
		'inputs': {
			'events_base_dir': _repo_rel(EVENTS_BASE_DIR),
			'loki_db_path': _repo_rel(LOKI_DB_PATH),
			'loki_hdr_filename': LOKI_HDR_FILENAME,
			'eqt_weight': _repo_rel(EQT_WEIGHT),
		},
		'outputs': {
			'loki_data_path': _repo_rel(LOKI_DATA_PATH),
			'loki_output_path': _repo_rel(LOKI_OUTPUT_PATH),
		},
		'event_selection': {
			'event_glob': EVENT_GLOB,
			'max_events': max_events,
			'event_count': int(event_count),
		},
		'eqt': {
			'in_samples': int(EQT_IN_SAMPLES),
			'overlap': int(EQT_OVERLAP),
			'batch_size': int(EQT_BATCH_SIZE),
			'channel_prefix': EQT_CHANNEL_PREFIX,
			'base_sampling_rate_hz': int(BASE_SAMPLING_RATE_HZ),
			'pre_enable': bool(PRE_ENABLE),
		},
		'loki': {
			'comp': list(COMP),
			'precision': PRECISION,
			'search': SEARCH,
			'npr': int(NPR),
			'ntrial': int(NTRIAL),
			'model': MODEL,
		},
		'validation': {
			'expected_min_time_buf_count': int(EXPECTED_MIN_TIME_BUF_COUNT),
			'time_buf_count': int(time_buf_count),
		},
	}
	write_json(RUN_CONFIG_JSON, config, ensure_ascii=False, indent=2)


def _reset_dir_empty(root: Path) -> None:
	root = Path(root)
	root.mkdir(parents=True, exist_ok=True)
	for p in root.iterdir():
		if p.is_dir():
			shutil.rmtree(p)
		else:
			p.unlink()


def _station_set(st: Stream) -> set[str]:
	return {
		str(tr.stats.station)
		for tr in st
		if getattr(tr.stats, 'station', None) is not None
	}


def _log_station_overlap(
	*,
	event_name: str,
	st_prob_ps: Stream,
	db_stations: set[str],
) -> None:
	stream_stations = sorted(_station_set(st_prob_ps))
	bad_gamma_ids = [sta for sta in stream_stations if '__' in sta]
	if bad_gamma_ids:
		raise ValueError(
			f'GaMMA station_id leaked into Loki probability Stream: '
			f'event={event_name} examples={bad_gamma_ids[:10]}'
		)

	overlap = sorted(set(stream_stations) & db_stations)
	print(
		f'station check: event={event_name} stream={len(stream_stations)} '
		f'header={len(db_stations)} overlap={len(overlap)} '
		f'stream_examples={stream_stations[:5]} header_examples={sorted(db_stations)[:5]}'
	)
	if not overlap:
		raise ValueError(
			f'no station overlap between Loki header and probability Stream: '
			f'event={event_name}'
		)


def _prepare_eqt_prob_stream(
	event_dir: Path,
	inputs: LokiWaveformStackingInputs,
	*,
	eqt_weights: str,
	eqt_in_samples: int,
	eqt_overlap: int,
	eqt_batch_size: int,
	channel_prefix: str,
) -> Stream:
	st = prepare_win32_stream(
		event_dir,
		base_fs_hz=int(inputs.base_sampling_rate_hz),
		pre_enable=bool(inputs.pre_enable),
		pre_spec=spec_from_inputs(inputs),
	)
	probs_by_station = build_probs_by_station(
		st,
		fs=float(inputs.base_sampling_rate_hz),
		eqt_weights=str(eqt_weights),
		eqt_in_samples=int(eqt_in_samples),
		eqt_overlap=int(eqt_overlap),
		eqt_batch_size=int(eqt_batch_size),
	)
	return build_loki_ps_prob_stream(
		ref_stream=st,
		probs_by_station=probs_by_station,
		channel_prefix=str(channel_prefix),
		require_both_ps=True,
	)


def _loki_kwargs(inputs: LokiWaveformStackingInputs) -> dict[str, object]:
	return {
		'npr': int(getattr(inputs, 'npr', 2)),
		'model': str(getattr(inputs, 'model', 'jma2001')),
	}


def _install_loki_direct_stream_patch(streams_by_event: dict[object, Stream]) -> None:
	"""Teach this Loki install to consume the direct Stream map for this runner.

	The repo pipelines already pass streams_by_event, but the installed Loki
	version in this environment still constructs Waveforms from files only. This
	local patch keeps the behavior scoped to keys present in streams_by_event and
	falls back to Loki's original file reader for every other path.
	"""
	import loki.waveforms as loki_waveforms

	global _LOKI_WAVEFORMS_ORIGINAL, _LOKI_DIRECT_STREAMS
	_LOKI_DIRECT_STREAMS = dict(streams_by_event)

	if _LOKI_WAVEFORMS_ORIGINAL is not None:
		return

	_LOKI_WAVEFORMS_ORIGINAL = loki_waveforms.Waveforms
	original_cls = _LOKI_WAVEFORMS_ORIGINAL

	class DirectStreamWaveforms(original_cls):  # type: ignore[misc, valid-type]
		def __init__(
			self,
			data_path,
			extension='*',
			comps=['E', 'N', 'Z'],
			freq=None,
			sds=False,
			tini=None,
			window=None,
			overlap=0.0,
			network='*',
		):
			stream = _LOKI_DIRECT_STREAMS.get(data_path)
			if stream is None:
				stream = _LOKI_DIRECT_STREAMS.get(str(data_path))
			if stream is None:
				super().__init__(
					data_path,
					extension=extension,
					comps=comps,
					freq=freq,
					sds=sds,
					tini=tini,
					window=window,
					overlap=overlap,
					network=network,
				)
				return

			self.stream = {}
			for comp in comps:
				self.stream[comp] = {}
				for tr in stream:
					if str(tr.stats.channel)[-1] != comp:
						continue
					start = tr.stats.starttime
					dtime = getattr(start, 'datetime', None)
					if dtime is None:
						dtime = start
					self.stream[comp][str(tr.stats.station)] = [
						dtime,
						float(tr.stats.delta),
						tr.data,
					]
			self.station_list()

	loki_waveforms.Waveforms = DirectStreamWaveforms


def _run_loki_for_event_direct(
	loki: object,
	cfg: LokiWaveformStackingPipelineConfig,
	*,
	event_name: str,
	stream_data_root: Path,
	st_prob_ps: Stream,
	loki_kwargs: dict[str, object],
) -> Path:
	_reset_dir_empty(stream_data_root)
	event_tmp_dir = stream_data_root / event_name
	event_tmp_dir.mkdir(parents=True, exist_ok=True)

	# LOKIの実装差（event名 or event_pathで引く）を吸収するため、キーを複数張る
	streams_by_event = {
		event_name: st_prob_ps,
		str(event_tmp_dir): st_prob_ps,
		event_tmp_dir: st_prob_ps,
	}
	_install_loki_direct_stream_patch(streams_by_event)

	# 念のため、LOKI側が保持するイベントリストを上書きできるなら1件に固定
	if hasattr(loki, 'data_tree'):
		loki.data_tree = [str(event_tmp_dir)]
	if hasattr(loki, 'events'):
		loki.events = [str(event_name)]

	loki.location(
		extension=cfg.extension,
		comp=['P', 'S'],
		precision=cfg.precision,
		search=cfg.search,
		streams_by_event=streams_by_event,
		**loki_kwargs,
	)
	return Path(cfg.loki_output_path) / event_name


def _assert_event_output_created(event_out_dir: Path) -> None:
	if not event_out_dir.is_dir():
		raise FileNotFoundError(
			f'Loki event output directory was not created: {event_out_dir}'
		)
	has_phs = bool(list(event_out_dir.glob('*.phs')))
	has_corr = (event_out_dir / 'corrmatrix_trial_0.npy').is_file()
	if not (has_phs or has_corr):
		raise FileNotFoundError(
			f'Loki output has no *.phs or corrmatrix_trial_0.npy: {event_out_dir}'
		)


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description='Run Loki EqT direct-input locations for Izu2009 GaMMA events.'
	)
	parser.add_argument(
		'--max-events',
		type=str,
		default=None,
		help='Override event limit. Use an integer or "all". Default uses MAX_EVENTS.',
	)
	args = parser.parse_args()
	if args.max_events is None:
		args.max_events_parsed = MAX_EVENTS
	elif args.max_events.lower() == 'all':
		args.max_events_parsed = None
	else:
		args.max_events_parsed = int(args.max_events)
	return args


def _validate_inputs_and_select_events(
	cfg: LokiWaveformStackingPipelineConfig,
) -> tuple[list[Path], list[Path]]:
	_candidate_event_dirs()
	time_buf_paths = _validate_traveltime_db()
	_validate_eqt_weight()
	_ensure_empty_or_missing_dir(LOKI_OUTPUT_PATH, 'LOKI_OUTPUT_PATH')

	event_dirs = list_event_dirs_filtered(cfg)
	if not event_dirs:
		raise ValueError('event count is 0 after MAX_EVENTS')

	_validate_event_json_groups(event_dirs)
	_validate_first_event_stream(event_dirs[0])
	return event_dirs, time_buf_paths


def main() -> None:
	"""Run the Izu2009 GaMMA event Loki EqT pipeline."""
	args = _parse_args()
	cfg = _build_cfg(max_events=args.max_events_parsed)
	inputs = _build_inputs()

	event_dirs, time_buf_paths = _validate_inputs_and_select_events(cfg)
	_write_run_config(
		event_count=len(event_dirs),
		time_buf_count=len(time_buf_paths),
		max_events=args.max_events_parsed,
	)

	LOKI_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
	stream_data_root = LOKI_DATA_PATH / '_streaming_direct_input'
	loki, header, _header_path = build_loki_with_header(
		cfg,
		data_path=stream_data_root,
		output_path=LOKI_OUTPUT_PATH,
	)
	_reset_dir_empty(stream_data_root)

	db_stations = set(header.stations_df['station'].astype(str).tolist())
	loki_kwargs = _loki_kwargs(inputs)

	processed = 0
	for event_dir in event_dirs:
		event_name = event_dir.name
		print(f'processing event: {event_name} dir={event_dir}')
		st_prob_ps = _prepare_eqt_prob_stream(
			event_dir,
			inputs,
			eqt_weights=str(EQT_WEIGHT),
			eqt_in_samples=int(EQT_IN_SAMPLES),
			eqt_overlap=int(EQT_OVERLAP),
			eqt_batch_size=int(EQT_BATCH_SIZE),
			channel_prefix=EQT_CHANNEL_PREFIX,
		)
		_log_station_overlap(
			event_name=event_name,
			st_prob_ps=st_prob_ps,
			db_stations=db_stations,
		)
		event_out_dir = _run_loki_for_event_direct(
			loki,
			cfg,
			event_name=event_name,
			stream_data_root=stream_data_root,
			st_prob_ps=st_prob_ps,
			loki_kwargs=loki_kwargs,
		)
		_assert_event_output_created(event_out_dir)
		processed += 1

	if not LOKI_OUTPUT_PATH.is_dir():
		raise FileNotFoundError(
			f'Loki output directory was not created: {LOKI_OUTPUT_PATH}'
		)
	if not LOKI_DATA_PATH.is_dir():
		raise FileNotFoundError(
			f'Loki data directory was not created: {LOKI_DATA_PATH}'
		)

	print(f'wrote run config: {RUN_CONFIG_JSON}')
	print(f'processed events: {processed}')
	print(f'Loki output: {LOKI_OUTPUT_PATH}')
	print(f'Loki data: {LOKI_DATA_PATH}')


if __name__ == '__main__':
	main()
