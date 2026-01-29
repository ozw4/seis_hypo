from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from hypo.synth_eval.io import write_station_csv


def test_write_station_csv_contract_and_casts(tmp_path: Path) -> None:
	out_csv = tmp_path / 'a' / 'b' / 'station_synth.csv'

	df = pd.DataFrame(
		{
			'Elevation_m': [10.0, 20.0],
			'foo': ['a', 'b'],
			'station_code': [1, 2],
			'receiver_index': [0, 1],
			'Longitude_deg': [140, 141],
			'bar': [3, 4],
			'Latitude_deg': [35, 36],
		}
	)

	write_station_csv(df, out_csv)
	assert out_csv.is_file()

	with out_csv.open('r', encoding='utf-8', newline='') as f:
		rows = list(csv.reader(f))

	header = rows[0]
	row1 = rows[1]

	assert header == [
		'station_code',
		'receiver_index',
		'Latitude_deg',
		'Longitude_deg',
		'Elevation_m',
		'foo',
		'bar',
	]

	assert row1[0] == '1'
	assert int(row1[1]) == 0
	assert float(row1[2]) == 35.0
	assert float(row1[3]) == 140.0
	assert int(row1[4]) == 10
	assert row1[5] == 'a'
	assert int(row1[6]) == 3


def test_write_station_csv_missing_required_columns(tmp_path: Path) -> None:
	df = pd.DataFrame(
		{'station_code': ['S0001'], 'Latitude_deg': [35.0], 'Longitude_deg': [140.0]}
	)

	with pytest.raises(ValueError):
		write_station_csv(df, tmp_path / 'station_synth.csv')
