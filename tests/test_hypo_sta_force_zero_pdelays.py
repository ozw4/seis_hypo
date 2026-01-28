from __future__ import annotations

from pathlib import Path

import pandas as pd

from hypo.sta import write_hypoinverse_sta


def _read_first_sta_line(path: Path) -> str:
	line = path.read_text(encoding='ascii').splitlines()[0]
	assert len(line) == 86
	return line


def test_write_hypoinverse_sta_force_zero_pdelays_overrides_csv(tmp_path: Path) -> None:
	station_csv = tmp_path / 'station.csv'
	out_sta = tmp_path / 'out.sta'

	pd.DataFrame(
		{
			'station_code': ['S0001'],
			'Latitude_deg': [35.0],
			'Longitude_deg': [140.0],
			'pdelay1': [1.23],
			'pdelay2': [4.56],
		}
	).to_csv(station_csv, index=False)

	write_hypoinverse_sta(station_csv, out_sta, force_zero_pdelays=True)

	line = _read_first_sta_line(out_sta)
	assert float(line[49:54]) == 0.0
	assert float(line[55:60]) == 0.0


def test_write_hypoinverse_sta_default_preserves_pdelays(tmp_path: Path) -> None:
	station_csv = tmp_path / 'station.csv'
	out_sta = tmp_path / 'out.sta'

	pd.DataFrame(
		{
			'station_code': ['S0001'],
			'Latitude_deg': [35.0],
			'Longitude_deg': [140.0],
			'pdelay1': [1.23],
			'pdelay2': [4.56],
		}
	).to_csv(station_csv, index=False)

	write_hypoinverse_sta(station_csv, out_sta, force_zero_pdelays=False)

	line = _read_first_sta_line(out_sta)
	assert float(line[49:54]) == 1.23
	assert float(line[55:60]) == 4.56
