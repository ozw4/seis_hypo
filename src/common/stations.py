from __future__ import annotations

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
