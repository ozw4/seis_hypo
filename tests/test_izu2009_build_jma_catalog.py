from __future__ import annotations

# ruff: noqa: INP001, TC003
from datetime import date
from pathlib import Path

import pytest

from proc.izu2009.catalog.build_jma_catalog_izu2009 import (
	_load_config,
	_planned_temp_paths,
	_require_output_paths_writable,
)


def _config_text(events_csv: Path, measurements_csv: Path, *, overwrite: bool) -> str:
	return (
		'download:\n'
		"  start_date: '2009-12-17'\n"
		"  end_date: '2009-12-21'\n"
		'  span_days: 1\n'
		'  netrc_machine: hinet\n'
		'  line_ending: unix\n'
		'  log_level: info\n'
		'catalog_filter:\n'
		"  start_time_jst: '2009-12-17 00:00:00'\n"
		"  end_time_jst: '2009-12-20 23:59:59'\n"
		'  center_lat: 34.9700\n'
		'  center_lon: 139.1300\n'
		'  radius_km: 50.0\n'
		'  mag_min: null\n'
		'  mag_max: null\n'
		'output:\n'
		f'  events_csv: {events_csv}\n'
		f'  measurements_csv: {measurements_csv}\n'
		f'  overwrite: {str(overwrite).lower()}\n'
	)


def test_load_config_normalizes_paths_dates_and_uppercase_values(
	tmp_path: Path,
) -> None:
	config_path = tmp_path / 'jma_catalog.yaml'
	events_csv = tmp_path / 'out/events.csv'
	measurements_csv = tmp_path / 'out/measurements.csv'
	config_path.write_text(
		_config_text(events_csv, measurements_csv, overwrite=False),
		encoding='utf-8',
	)

	config = _load_config(config_path)

	assert config['download']['start_date'] == date(2009, 12, 17)
	assert config['download']['end_date'] == date(2009, 12, 21)
	assert config['download']['line_ending'] == 'UNIX'
	assert config['download']['log_level'] == 'INFO'
	assert config['catalog_filter']['radius_km'] == 50.0
	assert config['catalog_filter']['mag_min'] is None
	assert config['output']['events_csv'] == events_csv
	assert config['output']['measurements_csv'] == measurements_csv


def test_load_config_rejects_bool_span_days(tmp_path: Path) -> None:
	config_path = tmp_path / 'jma_catalog.yaml'
	config_path.write_text(
		_config_text(
			tmp_path / 'events.csv',
			tmp_path / 'measurements.csv',
			overwrite=False,
		).replace('span_days: 1', 'span_days: true'),
		encoding='utf-8',
	)

	with pytest.raises(TypeError, match=r'download\.span_days'):
		_load_config(config_path)


def test_planned_temp_paths_match_downloader_layout(tmp_path: Path) -> None:
	config_path = tmp_path / 'jma_catalog.yaml'
	config_path.write_text(
		_config_text(
			tmp_path / 'events.csv', tmp_path / 'measurements.csv', overwrite=False
		),
		encoding='utf-8',
	)
	config = _load_config(config_path)

	paths = _planned_temp_paths(config, tmp_path / 'download')

	assert paths == [
		tmp_path / 'download/2009/arrivetime_20091217_1.txt',
		tmp_path / 'download/2009/arrivetime_20091218_1.txt',
		tmp_path / 'download/2009/arrivetime_20091219_1.txt',
		tmp_path / 'download/2009/arrivetime_20091220_1.txt',
	]


def test_require_output_paths_writable_rejects_existing_when_no_overwrite(
	tmp_path: Path,
) -> None:
	events_csv = tmp_path / 'out/events.csv'
	measurements_csv = tmp_path / 'out/measurements.csv'
	events_csv.parent.mkdir()
	events_csv.write_text('event_id\n1\n', encoding='utf-8')
	config_path = tmp_path / 'jma_catalog.yaml'
	config_path.write_text(
		_config_text(events_csv, measurements_csv, overwrite=False),
		encoding='utf-8',
	)
	config = _load_config(config_path)

	with pytest.raises(FileExistsError, match=r'output\.overwrite=false'):
		_require_output_paths_writable(config)


def test_require_output_paths_writable_creates_parent_dirs_when_clear(
	tmp_path: Path,
) -> None:
	config_path = tmp_path / 'jma_catalog.yaml'
	events_csv = tmp_path / 'out/events.csv'
	measurements_csv = tmp_path / 'out/measurements.csv'
	config_path.write_text(
		_config_text(events_csv, measurements_csv, overwrite=False),
		encoding='utf-8',
	)
	config = _load_config(config_path)

	_require_output_paths_writable(config)

	assert events_csv.parent.is_dir()
