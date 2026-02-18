# %%
"""Build GaMMA stations CSV from representative Hi-net .ch files."""

# file: proc/prepare_data/jma/build_gamma_stations_from_ch.py
#
# Purpose:
# - Build GaMMA stations CSV from representative Hi-net .ch files by network.
#
# Input:
#   CH_TABLE_BY_NETWORK: {network_code: /path/to/representative.ch}
#
# Output GaMMA core columns:
#   id      : station key (default '{network_code}.{station_code}')
#   x(km)   : local East [km] from origin (lat0/lon0 median)
#   y(km)   : local North [km] from origin (lat0/lon0 median)
#   z(km)   : depth [km] (currently fixed 0.0)
#
# Extra output columns (optional):
#   lat, lon, elevation_m, network_code, station_code

from __future__ import annotations

from pathlib import Path

import pandas as pd

from common.core import validate_columns
from common.geo import latlon_to_local_xy_km
from common.json_io import write_json
from jma.station_reader import read_hinet_channel_table

# =========================
# Parameters (edit here)
# =========================
CH_TABLE_BY_NETWORK: dict[str, Path] = {
	'0101': Path(
		'/workspace/proc/prepare_data/izu2009/download_continuous/continuous_ch47/0101/win_0101_200912170000_10m_aa3c27a4.ch'
	),
	'0203': Path(
		'/workspace/proc/prepare_data/izu2009/download_continuous/continuous_ch47/0203/win_0203_200912170000_10m_9a3c463f.ch'
	),
	'0207': Path(
		'/workspace/proc/prepare_data/izu2009/download_continuous/continuous_ch47/0207/win_0207_200912170000_10m_1c7df708.ch'
	),
	'0301': Path(
		'/workspace/proc/prepare_data/izu2009/download_continuous/continuous_ch47/0301/win_0301_200912170000_10m_4dd999af.ch'
	),
}

OUT_DIR = Path('/workspace/proc/run_continuous/association/jma/out')
OUT_GAMMA_STATIONS_CSV = OUT_DIR / 'gamma_stations.csv'
OUT_ORIGIN_LATLON_JSON = OUT_DIR / 'origin_latlon.json'

# station id mode (must match picks-side setting)
STATION_ID_MODE = 'network_station'

OUT_COLUMNS = ['id', 'x(km)', 'y(km)', 'z(km)']

INCLUDE_EXTRA_COLUMNS = True
EXTRA_COLUMNS = ['lat', 'lon', 'elevation_m', 'network_code', 'station_code']


def _normalize_station_rows(
	network_code: str,
	ch_path: Path,
) -> pd.DataFrame:
	ch_df = read_hinet_channel_table(ch_path)
	validate_columns(
		ch_df,
		['station', 'lat', 'lon', 'elevation_m'],
		f'.ch table: {ch_path}',
	)

	st = ch_df[['station', 'lat', 'lon', 'elevation_m']].copy()
	st['station'] = st['station'].astype('string').str.strip()
	if st['station'].isna().any() or (st['station'] == '').any():
		raise ValueError(f'empty station code found in {ch_path}')

	st['lat'] = pd.to_numeric(st['lat'], errors='raise').astype('float64')
	st['lon'] = pd.to_numeric(st['lon'], errors='raise').astype('float64')
	st['elevation_m'] = pd.to_numeric(st['elevation_m'], errors='raise').astype(
		'float64'
	)

	# Drop component duplicates by station.
	st = (
		st.groupby('station', as_index=False)
		.agg({'lat': 'first', 'lon': 'first', 'elevation_m': 'first'})
		.sort_values('station')
		.reset_index(drop=True)
	)

	st = st.rename(columns={'station': 'station_code'})
	st['network_code'] = str(network_code).strip()

	return st[['network_code', 'station_code', 'lat', 'lon', 'elevation_m']]


def _build_station_id(stations: pd.DataFrame) -> pd.Series:
	if STATION_ID_MODE == 'network_station':
		return stations['network_code'] + '.' + stations['station_code']

	if STATION_ID_MODE == 'station_only':
		net_per_station = (
			stations[['station_code', 'network_code']]
			.drop_duplicates()
			.groupby('station_code', as_index=False)['network_code']
			.nunique()
		)
		conflict = net_per_station.loc[net_per_station['network_code'] > 1]
		if not conflict.empty:
			examples = conflict['station_code'].head(20).tolist()
			raise ValueError(
				'station_only mode causes id collisions across network_code. '
				f'conflicting station_code examples={examples}'
			)
		return stations['station_code'].copy()

	raise ValueError(
		'STATION_ID_MODE must be '
		f"'network_station' or 'station_only', got {STATION_ID_MODE}"
	)


def build_gamma_stations_from_ch(
	ch_table_by_network: dict[str, Path],
) -> tuple[pd.DataFrame, dict[str, float]]:
	"""Create GaMMA stations table and median-origin metadata."""
	if not ch_table_by_network:
		raise ValueError('CH_TABLE_BY_NETWORK is empty')

	rows: list[pd.DataFrame] = []
	for network_code in sorted(ch_table_by_network):
		net = str(network_code).strip()
		if not net:
			raise ValueError('network_code must not be empty')

		ch_path = Path(ch_table_by_network[network_code])
		if not ch_path.is_file():
			raise FileNotFoundError(f'.ch not found for network_code={net}: {ch_path}')

		rows.append(_normalize_station_rows(net, ch_path))

	stations = pd.concat(rows, axis=0, ignore_index=True)
	if stations.empty:
		raise ValueError('no station rows extracted from .ch files')

	stations = stations.copy()
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

	if INCLUDE_EXTRA_COLUMNS:
		validate_columns(stations, EXTRA_COLUMNS, 'extra station columns source')
		out = pd.concat([out, stations[EXTRA_COLUMNS].copy()], axis=1)

	out = out.sort_values(['id'], kind='mergesort').reset_index(drop=True)
	validate_columns(out, OUT_COLUMNS, 'output GaMMA stations CSV')

	origin = {'lat0_deg': lat0, 'lon0_deg': lon0}
	return out, origin


def main() -> None:
	"""Run conversion using top-of-file constants."""
	OUT_DIR.mkdir(parents=True, exist_ok=True)

	stations_df, origin = build_gamma_stations_from_ch(CH_TABLE_BY_NETWORK)
	stations_df.to_csv(OUT_GAMMA_STATIONS_CSV, index=False)
	write_json(OUT_ORIGIN_LATLON_JSON, origin, ensure_ascii=False, indent=2)

	print('Wrote:', OUT_GAMMA_STATIONS_CSV)
	print('Rows:', int(stations_df.shape[0]))
	print('Networks:', int(stations_df['network_code'].nunique()))
	print('Origin:', OUT_ORIGIN_LATLON_JSON, origin)


if __name__ == '__main__':
	main()

# 実行例:
# export PYTHONPATH="$PWD/src"
# python proc/prepare_data/jma/build_gamma_stations_from_ch.py
# 入力.ch例:
#   /workspace/proc/prepare_data/izu2009/download_continuous/continuous_ch47/0101/
#   win_0101_200912170000_10m_aa3c27a4.ch
