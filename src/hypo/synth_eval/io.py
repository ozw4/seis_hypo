from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common.core import validate_columns


def write_station_csv(df_station: pd.DataFrame, out_csv: Path) -> None:
	"""station DataFrame を synth station CSV として書き出す。

	必須列:
	- station_code
	- receiver_index (0-based)
	- Latitude_deg
	- Longitude_deg
	- Elevation_m
	"""
	validate_columns(
		df_station,
		[
			'station_code',
			'receiver_index',
			'Latitude_deg',
			'Longitude_deg',
			'Elevation_m',
		],
		'station DataFrame',
	)

	df = df_station.copy()
	df['station_code'] = df['station_code'].astype(str)

	ri = pd.to_numeric(df['receiver_index'], errors='raise')
	if ri.isna().any():
		raise ValueError('receiver_index has missing values')
	vals = ri.to_numpy(float)
	if not np.isfinite(vals).all():
		raise ValueError('receiver_index has non-finite values')
	if not np.equal(vals, np.round(vals)).all():
		raise ValueError('receiver_index must be integer-valued')
	df['receiver_index'] = vals.astype(int)

	df['Latitude_deg'] = df['Latitude_deg'].astype(float)
	df['Longitude_deg'] = df['Longitude_deg'].astype(float)
	df['Elevation_m'] = df['Elevation_m'].astype(int)

	required = [
		'station_code',
		'receiver_index',
		'Latitude_deg',
		'Longitude_deg',
		'Elevation_m',
	]
	cols = required + [c for c in df.columns if c not in required]
	df = df[cols]

	out = Path(out_csv)
	out.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(out, index=False)
