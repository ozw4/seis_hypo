# src/jma/monthly_presence_selection.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common.geo import haversine_distance_km
from jma.stationcode_common import month_columns, normalize_network_code


@dataclass(frozen=True)
class MonthRange:
	month_cols: list[str]


def _month_range_cols(
	df: pd.DataFrame, *, start_time: str | pd.Timestamp, end_time: str | pd.Timestamp
) -> MonthRange:
	all_month_cols = month_columns(df)
	if not all_month_cols:
		raise ValueError('monthly_presence: no YYYY-MM columns found')

	t0 = pd.to_datetime(start_time)
	t1 = pd.to_datetime(end_time)
	if t0 > t1:
		raise ValueError('start_time > end_time')

	want = []
	cur = pd.Timestamp(year=t0.year, month=t0.month, day=1)
	last = pd.Timestamp(year=t1.year, month=t1.month, day=1)
	while cur <= last:
		want.append(f'{cur.year:04d}-{cur.month:02d}')
		cur = cur + pd.offsets.MonthBegin(1)

	missing = [c for c in want if c not in df.columns]
	if missing:
		raise ValueError(f'monthly_presence missing month columns: {missing}')

	return MonthRange(month_cols=want)


def load_active_stations_from_monthly_presence(
	monthly_presence_csv: str | Path,
	*,
	start_time: str | pd.Timestamp,
	end_time: str | pd.Timestamp,
) -> pd.DataFrame:
	"""monthly_presence.csv から「対象期間内（ざっくり=該当月のどれかが1）」の観測点を返す。
	lat/lon 等のメタ列は monthly_presence 側のものをそのまま使う。
	"""
	p = Path(monthly_presence_csv)
	if not p.is_file():
		raise FileNotFoundError(p)

	df = pd.read_csv(p, low_memory=False)

	required = {'network_code', 'station', 'lat', 'lon'}
	missing = required.difference(df.columns)
	if missing:
		raise ValueError(f'monthly_presence missing columns: {sorted(missing)}')

	mr = _month_range_cols(df, start_time=start_time, end_time=end_time)

	# 稼働判定: 対象月のどれかが 1
	active = np.zeros(len(df), dtype=bool)
	for c in mr.month_cols:
		active |= df[c].fillna(0).astype(int).to_numpy() == 1

	out_cols = []
	for c in [
		'network_code',
		'station',
		'station_name',
		'lat',
		'lon',
		'elevation_m',
		'components',
		'n_components',
	]:
		if c in df.columns:
			out_cols.append(c)

	out = df.loc[active, out_cols].copy()
	out['network_code'] = out['network_code'].map(normalize_network_code)
	out['station'] = out['station'].astype(str).str.strip()

	out['lat'] = pd.to_numeric(out['lat'], errors='raise')
	out['lon'] = pd.to_numeric(out['lon'], errors='raise')
	if 'elevation_m' in out.columns:
		out['elevation_m'] = pd.to_numeric(out['elevation_m'], errors='coerce')

	if out.empty:
		raise RuntimeError('no active stations in the specified month range')

	return out.reset_index(drop=True)


def stations_within_radius_from_monthly_presence(
	monthly_presence_csv: str | Path,
	*,
	start_time: str | pd.Timestamp,
	end_time: str | pd.Timestamp,
	center_lat: float,
	center_lon: float,
	radius_km: float,
) -> pd.DataFrame:
	"""monthly_presence.csv → 期間内稼働 → (center_lat, center_lon) 半径 radius_km 以内
	返り値は station 行（network_code, station, lat, lon, ... , dist_km）
	"""
	df = load_active_stations_from_monthly_presence(
		monthly_presence_csv, start_time=start_time, end_time=end_time
	)

	lat_arr = df['lat'].to_numpy(dtype=float)
	lon_arr = df['lon'].to_numpy(dtype=float)

	dist_km = haversine_distance_km(
		lat0_deg=float(center_lat),
		lon0_deg=float(center_lon),
		lat_deg=lat_arr,
		lon_deg=lon_arr,
	)
	df = df.copy()
	df['dist_km'] = dist_km
	df = df[df['dist_km'] <= float(radius_km)].copy()
	if df.empty:
		raise RuntimeError('no stations found within the specified radius')

	# 同一(network_code, station)重複がある場合は最短距離を代表に
	df = df.sort_values(['dist_km', 'network_code', 'station']).reset_index(drop=True)
	df = df.drop_duplicates(
		subset=['network_code', 'station'], keep='first'
	).reset_index(drop=True)
	return df


def write_station_lists(
	stations_df: pd.DataFrame,
	*,
	out_dir: str | Path,
	pick_n: int | None = None,
) -> None:
	"""LOKI/HinetPy入力用の局リストを書き出す（argparseなし運用を想定）。
	- stations_selected.csv
	- stations_selected.txt（stationのみ）
	- stations_<network_code>.txt
	"""
	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	df = stations_df.copy()
	if pick_n is not None and int(pick_n) > 0 and len(df) > int(pick_n):
		df = df.sort_values('dist_km').head(int(pick_n)).copy()

	df.to_csv(out_dir / 'stations_selected.csv', index=False, encoding='utf-8')

	(out_dir / 'stations_selected.txt').write_text(
		'\n'.join(df['station'].astype(str).tolist()) + '\n', encoding='utf-8'
	)

	for net, sub in df.groupby('network_code', as_index=False):
		(out_dir / f'stations_{net}.txt').write_text(
			'\n'.join(sorted(sub['station'].astype(str).unique().tolist())) + '\n',
			encoding='utf-8',
		)
