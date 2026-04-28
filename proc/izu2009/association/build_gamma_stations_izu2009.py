# %%
"""Build GaMMA stations CSV for the Izu 2009 selected station table."""

# file: proc/izu2009/association/build_gamma_stations_izu2009.py
#
# Purpose:
# - Convert proc/izu2009/prepare_data/profile/stations47/stations_47.csv
#   to a GaMMA stations CSV.
# - Use the same station_id rule as build_gamma_picks_izu2009.py:
#   '{network_code}__{station_code}'.
# - Write the median latitude/longitude origin used by the local XY projection.

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _REPO_ROOT / 'src'

for _path in (_REPO_ROOT, _SRC_DIR):
	_path_str = str(_path)
	if _path_str not in sys.path:
		sys.path.insert(0, _path_str)

core = importlib.import_module('common.core')
geo = importlib.import_module('common.geo')
json_io = importlib.import_module('common.json_io')

validate_columns = core.validate_columns
latlon_to_local_xy_km = geo.latlon_to_local_xy_km
write_json = json_io.write_json

IN_STATIONS_CSV = (
	_REPO_ROOT / 'proc/izu2009/prepare_data/profile/stations47/stations_47.csv'
)
OUT_DIR = _REPO_ROOT / 'proc/izu2009/association/in'
OUT_GAMMA_STATIONS_CSV = OUT_DIR / 'gamma_stations.csv'
OUT_ORIGIN_LATLON_JSON = OUT_DIR / 'origin_latlon.json'

NETWORK_STATION_SEPARATOR = '__'

REQUIRED_COLUMNS = ['network_code', 'station', 'lat', 'lon', 'elevation_m']
OUT_COLUMNS = ['id', 'x(km)', 'y(km)', 'z(km)']
EXTRA_COLUMNS = [
	'lat',
	'lon',
	'elevation_m',
	'network_code',
	'station_code',
	'station_name',
	'dist_km',
]


def _require_non_empty_string_column(df: pd.DataFrame, column: str, label: str) -> None:
	values = df[column].astype('string').str.strip()
	if values.isna().any() or (values == '').any():
		raise ValueError(f'{label}: empty {column} found')


def _require_finite_numeric_column(df: pd.DataFrame, column: str, label: str) -> None:
	values = pd.to_numeric(df[column], errors='raise').astype('float64')
	if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
		raise ValueError(f'{label}: non-finite {column} found')
	df[column] = values


def _normalize_station_table(stations_csv: Path) -> pd.DataFrame:
	if not stations_csv.is_file():
		raise FileNotFoundError(f'stations CSV not found: {stations_csv}')

	stations = pd.read_csv(
		stations_csv,
		dtype={'network_code': 'string', 'station': 'string'},
	)
	validate_columns(stations, REQUIRED_COLUMNS, f'stations CSV: {stations_csv}')
	if stations.empty:
		raise ValueError(f'stations CSV is empty: {stations_csv}')

	stations = stations.copy()
	stations['network_code'] = stations['network_code'].astype('string').str.strip()
	stations['station_code'] = stations['station'].astype('string').str.strip()

	_require_non_empty_string_column(stations, 'network_code', str(stations_csv))
	_require_non_empty_string_column(stations, 'station_code', str(stations_csv))

	for column in ['lat', 'lon', 'elevation_m']:
		_require_finite_numeric_column(stations, column, str(stations_csv))

	if 'dist_km' in stations.columns:
		_require_finite_numeric_column(stations, 'dist_km', str(stations_csv))

	return stations


def _build_station_id(stations: pd.DataFrame) -> pd.Series:
	sep = NETWORK_STATION_SEPARATOR
	for column in ['network_code', 'station_code']:
		has_separator = stations[column].astype('string').str.contains(sep, regex=False)
		if has_separator.any():
			examples = stations.loc[has_separator, column].head(20).tolist()
			raise ValueError(
				f'{column} contains the station-id separator {sep!r}: {examples}'
			)

	return stations['network_code'] + sep + stations['station_code']


def build_gamma_stations_izu2009(
	stations_csv: Path = IN_STATIONS_CSV,
) -> tuple[pd.DataFrame, dict[str, float]]:
	"""Create Izu 2009 GaMMA stations table and median-origin metadata."""
	stations = _normalize_station_table(Path(stations_csv))
	stations['id'] = _build_station_id(stations)

	dup = stations.loc[stations['id'].duplicated(keep=False)]
	if not dup.empty:
		dup_rows = (
			dup[['id', 'network_code', 'station_code']]
			.drop_duplicates()
			.sort_values(['id', 'network_code', 'station_code'])
		)
		raise ValueError(
			'duplicated station id detected. '
			f'examples={dup_rows.head(20).to_dict(orient="records")}'
		)

	lat0 = float(stations['lat'].median())
	lon0 = float(stations['lon'].median())

	x_km, y_km = latlon_to_local_xy_km(
		stations['lat'].to_numpy(dtype=float),
		stations['lon'].to_numpy(dtype=float),
		lat0_deg=lat0,
		lon0_deg=lon0,
	)

	out = pd.DataFrame(
		{
			'id': stations['id'],
			'x(km)': x_km.astype(float),
			'y(km)': y_km.astype(float),
			'z(km)': 0.0,
		}
	)

	present_extra_columns = [
		column for column in EXTRA_COLUMNS if column in stations.columns
	]
	validate_columns(stations, present_extra_columns, 'extra station columns source')
	out = pd.concat([out, stations[present_extra_columns].copy()], axis=1)

	out = out.sort_values(['id'], kind='mergesort').reset_index(drop=True)
	validate_columns(out, OUT_COLUMNS, 'output GaMMA stations CSV')

	origin = {'lat0_deg': lat0, 'lon0_deg': lon0}
	return out, origin


def main() -> None:
	"""Run Izu 2009 station conversion using the constants above."""
	OUT_DIR.mkdir(parents=True, exist_ok=True)

	stations_df, origin = build_gamma_stations_izu2009(IN_STATIONS_CSV)
	stations_df.to_csv(OUT_GAMMA_STATIONS_CSV, index=False)
	write_json(OUT_ORIGIN_LATLON_JSON, origin, ensure_ascii=False, indent=2)

	print('Wrote:', OUT_GAMMA_STATIONS_CSV)
	print('Rows:', int(stations_df.shape[0]))
	print('Networks:', int(stations_df['network_code'].nunique()))
	print('Origin:', OUT_ORIGIN_LATLON_JSON, origin)


if __name__ == '__main__':
	main()

# 実行例:
# python proc/izu2009/association/build_gamma_stations_izu2009.py
