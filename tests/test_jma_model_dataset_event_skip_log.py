from __future__ import annotations

import csv
from pathlib import Path

from jma_model_dataset.event_skip_log import (
	append_event_skip_log,
	is_expected_missing_input_error,
)


def test_is_expected_missing_input_error_requires_event_local_path(
	tmp_path: Path,
) -> None:
	event_dir = tmp_path / 'event'
	raw_dir = event_dir / 'raw'

	assert is_expected_missing_input_error(
		NotADirectoryError(f'raw directory not found: {raw_dir}'),
		event_dir=event_dir,
	)
	assert is_expected_missing_input_error(
		ValueError(f'.evt must be exactly 1 in {raw_dir} (found 0): '),
		event_dir=event_dir,
	)
	assert is_expected_missing_input_error(
		ValueError(
			'no missing station pairs found in '
			f'{event_dir}/flows/jma_model_dataset/missing/event_missing_continuous.txt'
		),
		event_dir=event_dir,
	)

	assert not is_expected_missing_input_error(
		FileNotFoundError('/outside/event/raw/a.evt'),
		event_dir=event_dir,
	)
	assert not is_expected_missing_input_error(
		ValueError(f'.evt must be exactly 1 in {raw_dir} (found 2): a.evt, b.evt'),
		event_dir=event_dir,
	)
	assert not is_expected_missing_input_error(
		ValueError('target_fs_hz must be >= 1'),
		event_dir=event_dir,
	)


def test_append_event_skip_log_appends_header_once_and_evt_file(
	tmp_path: Path,
) -> None:
	event_dir = tmp_path / 'D20220101000000_00'
	raw_dir = event_dir / 'raw'
	raw_dir.mkdir(parents=True)
	(raw_dir / 'D20220101000000_00.evt').write_text('evt\n', encoding='utf-8')

	error = FileNotFoundError(
		'flow active .ch not found: '
		f'{event_dir}/flows/jma_model_dataset/active/a_active.ch'
	)
	log_path = append_event_skip_log(
		event_dir,
		step_name='06_export_100hz',
		reason='missing_input',
		error=error,
	)
	append_event_skip_log(
		event_dir,
		step_name='06_export_100hz',
		reason='missing_input',
		error=error,
	)

	lines = log_path.read_text(encoding='utf-8').splitlines()
	rows = list(csv.DictReader(lines))
	assert len(rows) == 2
	assert lines[0].startswith('time_utc,step_name,reason,event_dir')
	assert all(row['time_utc'].endswith('Z') for row in rows)
	assert all(row['step_name'] == '06_export_100hz' for row in rows)
	assert all(row['reason'] == 'missing_input' for row in rows)
	assert all(row['event_dir'] == str(event_dir.resolve()) for row in rows)
	assert all(row['event_name'] == event_dir.name for row in rows)
	assert all(row['evt_file'] == 'D20220101000000_00.evt' for row in rows)
	assert all(row['exception_class'] == 'FileNotFoundError' for row in rows)
	assert all(row['message'] == str(error) for row in rows)
