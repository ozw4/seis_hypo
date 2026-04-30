# %%
"""Run Loki EqT direct-input locations for Izu2009 GaMMA event dirs."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
from pipelines.loki_waveform_stacking_pipelines import (  # noqa: E402
	list_event_dirs_filtered,
	pipeline_loki_waveform_stacking_eqt,
)

EVENTS_BASE_DIR = _REPO_ROOT / 'proc/izu2009/loki/events_from_gamma'
LOKI_DB_PATH = _REPO_ROOT / 'proc/izu2009/loki/traveltime/db'
LOKI_HDR_FILENAME = 'header.hdr'

LOKI_DATA_PATH = _REPO_ROOT / 'proc/izu2009/loki/loki_data_eqt_gamma'
LOKI_OUTPUT_PATH = _REPO_ROOT / 'proc/izu2009/loki/output_eqt_gamma'
RUN_CONFIG_JSON = _REPO_ROOT / 'proc/izu2009/loki/run_loki_eqt_from_gamma_config.json'

EVENT_GLOB = '[0-9]*'
MAX_EVENTS: int | None = 50

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


def _build_cfg() -> LokiWaveformStackingPipelineConfig:
	if MAX_EVENTS is not None and int(MAX_EVENTS) <= 0:
		raise ValueError(f'MAX_EVENTS must be positive or None, got {MAX_EVENTS}')

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
		max_events=MAX_EVENTS,
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


def _write_run_config(*, event_count: int, time_buf_count: int) -> None:
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
			'max_events': MAX_EVENTS,
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


def _validate_inputs_and_select_events(
	cfg: LokiWaveformStackingPipelineConfig,
) -> tuple[list[Path], list[Path]]:
	_candidate_event_dirs()
	time_buf_paths = _validate_traveltime_db()
	_validate_eqt_weight()
	_ensure_empty_or_missing_dir(LOKI_OUTPUT_PATH, 'LOKI_OUTPUT_PATH')
	_ensure_empty_or_missing_dir(LOKI_DATA_PATH, 'LOKI_DATA_PATH')

	event_dirs = list_event_dirs_filtered(cfg)
	if not event_dirs:
		raise ValueError('event count is 0 after MAX_EVENTS')

	_validate_event_json_groups(event_dirs)
	_validate_first_event_stream(event_dirs[0])
	return event_dirs, time_buf_paths


def main() -> None:
	"""Run the Izu2009 GaMMA event Loki EqT pipeline."""
	cfg = _build_cfg()
	inputs = _build_inputs()

	event_dirs, time_buf_paths = _validate_inputs_and_select_events(cfg)
	_write_run_config(
		event_count=len(event_dirs),
		time_buf_count=len(time_buf_paths),
	)

	pipeline_loki_waveform_stacking_eqt(
		cfg,
		inputs,
		eqt_weights=str(EQT_WEIGHT),
		eqt_in_samples=int(EQT_IN_SAMPLES),
		eqt_overlap=int(EQT_OVERLAP),
		eqt_batch_size=int(EQT_BATCH_SIZE),
		channel_prefix=EQT_CHANNEL_PREFIX,
	)

	if not LOKI_OUTPUT_PATH.is_dir():
		raise FileNotFoundError(
			f'Loki output directory was not created: {LOKI_OUTPUT_PATH}'
		)
	if not LOKI_DATA_PATH.is_dir():
		raise FileNotFoundError(
			f'Loki data directory was not created: {LOKI_DATA_PATH}'
		)

	print(f'wrote run config: {RUN_CONFIG_JSON}')
	print(f'processed events: {len(event_dirs)}')
	print(f'Loki output: {LOKI_OUTPUT_PATH}')
	print(f'Loki data: {LOKI_DATA_PATH}')


if __name__ == '__main__':
	main()
