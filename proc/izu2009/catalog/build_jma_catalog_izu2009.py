# %%
"""Build the Izu 2009 JMA catalog CSVs without keeping intermediate files."""

# ruff: noqa: ANN001, ANN202, ANN401, ARG001, D101, D103, PLC0415, TC003

# file: proc/izu2009/catalog/build_jma_catalog_izu2009.py
#
# Purpose:
# - Download JMA arrival-time txt files from Hi-net into a temporary directory.
# - Parse and filter them using the existing repository utilities.
# - Write only the final Izu/time-window filtered catalog CSVs.
#
# Outputs:
# - proc/izu2009/catalog/out/jma_events_izu2009_20091217_20091220_r50km.csv
# - proc/izu2009/catalog/out/jma_measurements_izu2009_20091217_20091220_r50km.csv

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path
from typing import Any, TypedDict

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _REPO_ROOT / 'src'

for _path in (_REPO_ROOT, _SRC_DIR):
	_path_str = str(_path)
	if _path_str not in sys.path:
		sys.path.insert(0, _path_str)

DEFAULT_CONFIG = _REPO_ROOT / 'proc/izu2009/catalog/config/jma_catalog_izu2009.yaml'
VALID_LINE_ENDINGS = {'UNIX', 'DOS'}
VALID_LOG_LEVELS = {'DEBUG', 'INFO', 'WARNING', 'ERROR'}
LOGGER = logging.getLogger(__name__)


class DownloadConfig(TypedDict):
	start_date: date
	end_date: date
	span_days: int
	netrc_machine: str
	line_ending: str
	log_level: str


class CatalogFilterConfig(TypedDict):
	start_time_jst: str
	end_time_jst: str
	center_lat: float
	center_lon: float
	radius_km: float | None
	mag_min: float | None
	mag_max: float | None


class OutputConfig(TypedDict):
	events_csv: Path
	measurements_csv: Path
	overwrite: bool


class JmaCatalogConfig(TypedDict):
	download: DownloadConfig
	catalog_filter: CatalogFilterConfig
	output: OutputConfig


def _parse_date(value: Any, *, field: str) -> date:
	if isinstance(value, date):
		return value
	if isinstance(value, str):
		return date.fromisoformat(value)
	raise TypeError(f'{field} must be YYYY-MM-DD string, got {type(value).__name__}')


def _parse_optional_float(value: Any, *, field: str) -> float | None:
	if value is None:
		return None
	return float(value)


def _parse_path(value: Any, *, field: str) -> Path:
	if not isinstance(value, str):
		raise TypeError(f'{field} must be a path string, got {type(value).__name__}')
	path = Path(value)
	return path if path.is_absolute() else _REPO_ROOT / path


def _as_bool(value: Any, *, field: str) -> bool:
	if isinstance(value, bool):
		return value
	raise TypeError(f'{field} must be boolean, got {type(value).__name__}')


def _as_int(value: Any, *, field: str) -> int:
	if isinstance(value, bool) or not isinstance(value, int):
		raise TypeError(f'{field} must be integer, got {type(value).__name__}')
	return value


def _as_str(value: Any, *, field: str) -> str:
	if isinstance(value, str):
		return value
	raise TypeError(f'{field} must be string, got {type(value).__name__}')


def _require_mapping(obj: Any, *, field: str) -> dict[str, Any]:
	if not isinstance(obj, dict):
		raise TypeError(f'{field} must be a mapping')
	return obj


def _load_config(config_path: Path) -> JmaCatalogConfig:
	if not config_path.is_file():
		raise FileNotFoundError(f'config YAML not found: {config_path}')

	with config_path.open('r', encoding='utf-8') as f:
		obj = yaml.safe_load(f)

	root = _require_mapping(obj, field='config YAML root')
	download = _require_mapping(root.get('download'), field='download')
	catalog_filter = _require_mapping(
		root.get('catalog_filter'), field='catalog_filter'
	)
	output = _require_mapping(root.get('output'), field='output')

	for section_name, section, required_keys in [
		(
			'download',
			download,
			[
				'start_date',
				'end_date',
				'span_days',
				'netrc_machine',
				'line_ending',
				'log_level',
			],
		),
		(
			'catalog_filter',
			catalog_filter,
			[
				'start_time_jst',
				'end_time_jst',
				'center_lat',
				'center_lon',
				'radius_km',
				'mag_min',
				'mag_max',
			],
		),
		('output', output, ['events_csv', 'measurements_csv', 'overwrite']),
	]:
		missing = [key for key in required_keys if key not in section]
		if missing:
			raise ValueError(f'missing {section_name} config key(s): {missing}')

	start_date = _parse_date(download['start_date'], field='download.start_date')
	end_date = _parse_date(download['end_date'], field='download.end_date')
	span_days = _as_int(download['span_days'], field='download.span_days')
	line_ending = _as_str(download['line_ending'], field='download.line_ending').upper()
	log_level = _as_str(download['log_level'], field='download.log_level').upper()

	if start_date >= end_date:
		raise ValueError(
			f'download.start_date must be earlier than download.end_date: '
			f'{start_date} >= {end_date}'
		)
	if span_days <= 0:
		raise ValueError(f'download.span_days must be positive: {span_days}')
	if line_ending not in VALID_LINE_ENDINGS:
		raise ValueError(
			f'download.line_ending must be one of {sorted(VALID_LINE_ENDINGS)}'
		)
	if log_level not in VALID_LOG_LEVELS:
		raise ValueError(
			f'download.log_level must be one of {sorted(VALID_LOG_LEVELS)}'
		)

	events_csv = _parse_path(output['events_csv'], field='output.events_csv')
	measurements_csv = _parse_path(
		output['measurements_csv'],
		field='output.measurements_csv',
	)

	return {
		'download': {
			'start_date': start_date,
			'end_date': end_date,
			'span_days': span_days,
			'netrc_machine': _as_str(
				download['netrc_machine'],
				field='download.netrc_machine',
			),
			'line_ending': line_ending,
			'log_level': log_level,
		},
		'catalog_filter': {
			'start_time_jst': _as_str(
				catalog_filter['start_time_jst'],
				field='catalog_filter.start_time_jst',
			),
			'end_time_jst': _as_str(
				catalog_filter['end_time_jst'],
				field='catalog_filter.end_time_jst',
			),
			'center_lat': float(catalog_filter['center_lat']),
			'center_lon': float(catalog_filter['center_lon']),
			'radius_km': _parse_optional_float(
				catalog_filter['radius_km'],
				field='catalog_filter.radius_km',
			),
			'mag_min': _parse_optional_float(
				catalog_filter['mag_min'],
				field='catalog_filter.mag_min',
			),
			'mag_max': _parse_optional_float(
				catalog_filter['mag_max'],
				field='catalog_filter.mag_max',
			),
		},
		'output': {
			'events_csv': events_csv,
			'measurements_csv': measurements_csv,
			'overwrite': _as_bool(output['overwrite'], field='output.overwrite'),
		},
	}


def _iter_request_dates(start: date, end: date, span_days: int) -> Iterator[date]:
	step = timedelta(days=span_days)
	current = start
	while current < end:
		yield current
		current += step


def _build_temp_arrivetime_path(
	temp_output_dir: Path, start: date, span_days: int
) -> Path:
	return (
		temp_output_dir / f'{start.year}' / f'arrivetime_{start:%Y%m%d}_{span_days}.txt'
	)


def _planned_temp_paths(config: JmaCatalogConfig, temp_output_dir: Path) -> list[Path]:
	download = config['download']
	return [
		_build_temp_arrivetime_path(
			temp_output_dir, request_start, download['span_days']
		)
		for request_start in _iter_request_dates(
			download['start_date'],
			download['end_date'],
			download['span_days'],
		)
	]


def _require_output_paths_writable(config: JmaCatalogConfig) -> None:
	output = config['output']
	paths = [output['events_csv'], output['measurements_csv']]
	existing = [path for path in paths if path.exists()]
	if existing and not output['overwrite']:
		lines = '\n'.join(f'  - {path}' for path in existing)
		raise FileExistsError(
			f'final output already exists and output.overwrite=false:\n{lines}'
		)
	for path in paths:
		path.parent.mkdir(parents=True, exist_ok=True)


def _print_plan(config: JmaCatalogConfig) -> None:
	download = config['download']
	catalog_filter = config['catalog_filter']
	output = config['output']

	print('JMA Izu2009 final-catalog plan')
	print('  download:')
	print('    start_date    :', download['start_date'])
	print('    end_date      :', download['end_date'], '(exclusive)')
	print('    span_days     :', download['span_days'])
	print('    netrc_machine :', download['netrc_machine'])
	print('    line_ending   :', download['line_ending'])
	print('    temp files    : deleted automatically after parsing')
	print('  filter:')
	print('    start_time_jst:', catalog_filter['start_time_jst'])
	print('    end_time_jst  :', catalog_filter['end_time_jst'])
	print(
		'    center        :',
		catalog_filter['center_lat'],
		catalog_filter['center_lon'],
	)
	print('    radius_km     :', catalog_filter['radius_km'])
	print('    mag_min       :', catalog_filter['mag_min'])
	print('    mag_max       :', catalog_filter['mag_max'])
	print('  final outputs:')
	print('    events_csv      :', output['events_csv'])
	print('    measurements_csv:', output['measurements_csv'])
	print('    overwrite       :', output['overwrite'])
	print('  planned Hi-net requests:')
	for request_start in _iter_request_dates(
		download['start_date'],
		download['end_date'],
		download['span_days'],
	):
		print(f'    - start={request_start}')


def _download_to_temp(config: JmaCatalogConfig, temp_output_dir: Path) -> list[Path]:
	# Import Hi-net dependencies only when actually downloading. This keeps
	# --dry-run usable even in environments where HinetPy is not installed yet.
	from jma.download import create_hinet_client
	from jma.get_arrivetime import download_arrivaltime_range

	download = config['download']
	client = create_hinet_client(machine=download['netrc_machine'])
	download_arrivaltime_range(
		client,
		start=download['start_date'],
		end=download['end_date'],
		span_days=download['span_days'],
		output_dir=temp_output_dir,
		line_ending=download['line_ending'],
		overwrite=True,
	)

	input_paths = _planned_temp_paths(config, temp_output_dir)
	missing = [path for path in input_paths if not path.is_file()]
	if missing:
		lines = '\n'.join(f'  - {path}' for path in missing)
		raise FileNotFoundError(f'downloaded arrival-time file(s) not found:\n{lines}')
	return input_paths


def _build_raw_catalog_frames(input_paths: list[Path]):
	# Import pandas and parsing helpers only for the actual build step.
	import pandas as pd

	from jma.arrivetime_reader import (
		EVENT_FIELD_NAMES,
		MEAS_FIELD_NAMES,
		build_epicenter_row,
		iter_arrivetime_event_records,
	)

	epicenter_rows: list[list[str]] = []
	measurement_rows: list[list[str]] = []

	for event_id, epicenter_record, rows in iter_arrivetime_event_records(input_paths):
		epicenter_rows.append(build_epicenter_row(event_id, epicenter_record))
		measurement_rows.extend(rows)

	if not epicenter_rows:
		raise RuntimeError(
			'no epicenter records found in downloaded arrival-time files'
		)
	if not measurement_rows:
		raise RuntimeError(
			'no measurement records found in downloaded arrival-time files'
		)

	epic_df = pd.DataFrame(epicenter_rows, columns=EVENT_FIELD_NAMES)
	meas_df = pd.DataFrame(measurement_rows, columns=MEAS_FIELD_NAMES)

	# convert_epicenter_to_csv() normally writes blanks, then pd.read_csv() turns
	# them into NaN. Because this script builds DataFrames directly in memory,
	# reproduce that numeric coercion before calling catalog.selection.
	for col in [
		'event_id',
		'origin_time_std_s',
		'latitude_deg',
		'latitude_std_deg',
		'longitude_deg',
		'longitude_std_deg',
		'depth_km',
		'depth_std_km',
		'mag1',
		'mag2',
		'station_count',
	]:
		epic_df[col] = pd.to_numeric(epic_df[col].replace('', pd.NA), errors='coerce')
	meas_df['event_id'] = pd.to_numeric(
		meas_df['event_id'].replace('', pd.NA),
		errors='coerce',
	)

	return epic_df, meas_df


def _filter_catalog(epic_df, meas_df, config: JmaCatalogConfig):
	from catalog.selection import extract_events_in_region

	catalog_filter = config['catalog_filter']
	return extract_events_in_region(
		epic_df=epic_df,
		meas_df=meas_df,
		start_time=catalog_filter['start_time_jst'],
		end_time=catalog_filter['end_time_jst'],
		mag_min=catalog_filter['mag_min'],
		mag_max=catalog_filter['mag_max'],
		center_lat=catalog_filter['center_lat'],
		center_lon=catalog_filter['center_lon'],
		radius_km=catalog_filter['radius_km'],
	)


def _write_final_outputs(events_df, measurements_df, config: JmaCatalogConfig) -> None:
	output = config['output']
	events_df.to_csv(output['events_csv'], index=False)
	measurements_df.to_csv(output['measurements_csv'], index=False)


def build_final_jma_catalog(config: JmaCatalogConfig) -> tuple[int, int]:
	"""Download, parse, filter, and write final catalog CSVs only."""
	_require_output_paths_writable(config)
	with tempfile.TemporaryDirectory(prefix='izu2009_jma_arrivetime_') as temp_dir:
		temp_output_dir = Path(temp_dir)
		input_paths = _download_to_temp(config, temp_output_dir)
		epic_df, meas_df = _build_raw_catalog_frames(input_paths)
		events_df, measurements_df = _filter_catalog(epic_df, meas_df, config)
		_write_final_outputs(events_df, measurements_df, config)
		return int(events_df.shape[0]), int(measurements_df.shape[0])


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			'Download JMA arrival-time data for Izu 2009, discard intermediate '
			'files, and write only filtered final catalog CSVs.'
		)
	)
	parser.add_argument(
		'--config',
		type=Path,
		default=DEFAULT_CONFIG,
		help='YAML config path',
	)
	parser.add_argument(
		'--dry-run',
		action='store_true',
		help='print the plan without logging in, downloading, or writing outputs',
	)
	return parser


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()

	config = _load_config(args.config)
	logging.basicConfig(
		level=getattr(logging, config['download']['log_level']),
		format='%(asctime)s %(levelname)s %(message)s',
	)

	_print_plan(config)
	if args.dry_run:
		return

	event_count, measurement_count = build_final_jma_catalog(config)
	print('Wrote:', config['output']['events_csv'])
	print('Wrote:', config['output']['measurements_csv'])
	print('Events:', event_count)
	print('Measurements:', measurement_count)


if __name__ == '__main__':
	main()

# Examples:
# python proc/izu2009/catalog/build_jma_catalog_izu2009.py --dry-run
# python proc/izu2009/catalog/build_jma_catalog_izu2009.py
