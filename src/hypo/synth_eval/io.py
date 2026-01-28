from __future__ import annotations

from pathlib import Path

import pandas as pd

from common.core import validate_columns


def write_station_csv(df_station: pd.DataFrame, out_csv: Path) -> None:
	"""station DataFrame を synth station CSV として書き出す。

	必須列:
	- station_code
	- Latitude_deg
	- Longitude_deg
	- Elevation_m
	"""
	validate_columns(
		df_station,
		['station_code', 'Latitude_deg', 'Longitude_deg', 'Elevation_m'],
		'station DataFrame',
	)

	df = df_station.copy()
	df['station_code'] = df['station_code'].astype(str)
	df['Latitude_deg'] = df['Latitude_deg'].astype(float)
	df['Longitude_deg'] = df['Longitude_deg'].astype(float)
	df['Elevation_m'] = df['Elevation_m'].astype(int)

	required = ['station_code', 'Latitude_deg', 'Longitude_deg', 'Elevation_m']
	cols = required + [c for c in df.columns if c not in required]
	df = df[cols]

	out = Path(out_csv)
	out.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(out, index=False)
