from __future__ import annotations

from pathlib import Path

import pandas as pd

from common.core import validate_columns

ELEVATION_COLUMN_CANDIDATES: tuple[str, ...] = ('elevation_m', 'elev_m', 'elevation')


def normalize_station_rows(
	stations_df: pd.DataFrame,
	*,
	require_elevation: bool = False,
) -> pd.DataFrame:
	"""Return station-unique rows with consistent lat/lon/elevation columns."""
	if stations_df.empty:
		raise ValueError('stations_df is empty')

	validate_columns(stations_df, ['station', 'lat', 'lon'], 'stations_df')

	df = stations_df.copy()

	elev_col: str | None = None
	for cand in ELEVATION_COLUMN_CANDIDATES:
		if cand in df.columns:
			elev_col = cand
			break

	if require_elevation and elev_col is None:
		raise ValueError(
			f'stations_df must contain one of {ELEVATION_COLUMN_CANDIDATES} when '
			'require_elevation=True'
		)

	if elev_col is None:
		df['elevation_m'] = 0.0
	else:
		if elev_col != 'elevation_m':
			df['elevation_m'] = df[elev_col]
		df['elevation_m'] = df['elevation_m'].fillna(0.0)

	df['lat'] = df['lat'].astype(float)
	df['lon'] = df['lon'].astype(float)
	df['elevation_m'] = df['elevation_m'].astype(float)

	df = (
		df[['station', 'lat', 'lon', 'elevation_m']]
		.groupby('station', as_index=False)
		.first()
		.sort_values('station')
		.reset_index(drop=True)
	)

	if df.empty:
		raise ValueError('stations_df is empty after station grouping')

	return df


def read_forge_stations_portal_depth(path: Path) -> pd.DataFrame:
	if not path.is_file():
		raise FileNotFoundError(f'stations csv not found: {path}')

	df = pd.read_csv(path)
	if df.empty:
		raise ValueError(f'stations csv is empty: {path}')

	if 'station' not in df.columns:
		if 'station_id' in df.columns:
			df = df.copy()
			df['station'] = df['station_id'].astype(str)
		else:
			raise ValueError("stations csv must contain 'station' or 'station_id'")

	if 'lat' not in df.columns or 'lon' not in df.columns:
		raise ValueError("stations csv must contain 'lat' and 'lon'")

	if 'depth_m' not in df.columns:
		raise ValueError(
			"stations csv must contain 'depth_m' (portal-based, meters, positive downward)"
		)

	df = df.copy()
	df['station'] = df['station'].astype(str)
	if df['station'].isna().any():
		raise ValueError('station contains NaN')

	df['lat'] = df['lat'].astype(float)
	df['lon'] = df['lon'].astype(float)
	df['depth_m'] = pd.to_numeric(df['depth_m'], errors='coerce')

	if df['depth_m'].isna().any():
		raise ValueError('depth_m has NaN; fix station metadata')
	if (df['depth_m'] < 0).any():
		raise ValueError('depth_m must be non-negative (portal-based depth)')

	df['elevation_m'] = -df['depth_m'].astype(float)
	return df[['station', 'lat', 'lon', 'depth_m', 'elevation_m']].copy()
