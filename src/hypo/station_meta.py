from pathlib import Path

import pandas as pd
from common.core import validate_columns


def parse_station_code(code: str) -> tuple[str | None, str | None]:
	s = str(code).strip().upper()
	if not s:
		return None, None
	sta = s[:5]
	if not sta:
		return None, None
	net = s[5] if len(s) == 6 else ''
	return sta, net


def build_station_meta(station_csv: str | Path) -> dict[str, dict]:
	station_df = pd.read_csv(station_csv)
	validate_columns(
		station_df,
		['station_code', 'Latitude_deg', 'Longitude_deg'],
		'station CSV',
	)
	meta: dict[str, dict] = {}
	for _, row in station_df.iterrows():
		code = str(row['station_code']).strip()
		if not code:
			continue
		sta, net = parse_station_code(code)
		if sta is None:
			continue

		lat = row['Latitude_deg']
		lon = row['Longitude_deg']
		lat_val = float(lat) if pd.notna(lat) else None
		lon_val = float(lon) if pd.notna(lon) else None

		meta[code] = {
			'net': net,
			'sta': sta,
			'lat': lat_val,
			'lon': lon_val,
		}
	return meta
