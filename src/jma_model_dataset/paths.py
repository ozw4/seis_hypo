from __future__ import annotations

from pathlib import Path

__all__ = [
	'flow_root',
	'raw_root',
	'active_dir',
	'missing_dir',
	'continuous_dir',
	'logs_dir',
	'done_dir',
	'export_dir',
	'active_ch_path',
	'missing_txt_path',
	'mapping_log_path',
	'continuous_done_path',
	'fill_to_48_done_path',
]


def _event_dir_path(event_dir: Path) -> Path:
	return Path(event_dir)


def _require_single_segment(value: str, *, field_name: str) -> str:
	value2 = str(value).strip()
	if value2 == '':
		raise ValueError(f'{field_name} must be non-empty')
	if value2 in {'.', '..'}:
		raise ValueError(f'{field_name} must not be {value2!r}')
	if '/' in value2 or '\\' in value2:
		raise ValueError(f'{field_name} must not contain path separators: {value2}')
	return value2


def flow_root(event_dir: Path) -> Path:
	return _event_dir_path(event_dir) / 'flows' / 'jma_model_dataset'


def raw_root(event_dir: Path) -> Path:
	return _event_dir_path(event_dir) / 'raw'


def active_dir(event_dir: Path) -> Path:
	return flow_root(event_dir) / 'active'


def missing_dir(event_dir: Path) -> Path:
	return flow_root(event_dir) / 'missing'


def continuous_dir(event_dir: Path) -> Path:
	return flow_root(event_dir) / 'continuous'


def logs_dir(event_dir: Path) -> Path:
	return flow_root(event_dir) / 'logs'


def done_dir(event_dir: Path) -> Path:
	return flow_root(event_dir) / 'done'


def export_dir(event_dir: Path) -> Path:
	return flow_root(event_dir) / 'export'


def active_ch_path(event_dir: Path, stem: str) -> Path:
	stem2 = _require_single_segment(stem, field_name='stem')
	return active_dir(event_dir) / f'{stem2}_active.ch'


def missing_txt_path(event_dir: Path, stem: str) -> Path:
	stem2 = _require_single_segment(stem, field_name='stem')
	return missing_dir(event_dir) / f'{stem2}_missing_continuous.txt'


def mapping_log_path(event_dir: Path, stem: str) -> Path:
	stem2 = _require_single_segment(stem, field_name='stem')
	return missing_dir(event_dir) / f'{stem2}_mapping_log.csv'


def continuous_done_path(
	event_dir: Path,
	stem: str,
	run_tag: str,
	network_code: str,
) -> Path:
	stem2 = _require_single_segment(stem, field_name='stem')
	run_tag2 = _require_single_segment(run_tag, field_name='run_tag')
	network_code2 = _require_single_segment(
		network_code, field_name='network_code'
	)
	return (
		done_dir(event_dir)
		/ f'{stem2}_continuous_done_{run_tag2}_{network_code2}.json'
	)


def fill_to_48_done_path(event_dir: Path, stem: str, run_tag: str) -> Path:
	stem2 = _require_single_segment(stem, field_name='stem')
	run_tag2 = _require_single_segment(run_tag, field_name='run_tag')
	return done_dir(event_dir) / f'{stem2}_fill_to_48_done_{run_tag2}.json'
