# file: proc/loki_hypo/compare_df.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common.core import load_event_json
from common.geo import haversine_distance_km, local_xy_km_to_latlon
from common.time_util import get_event_origin_utc, to_utc
from loki_tools.loki_parse import (
	infer_event_origin_time_from_loki_result,
	list_event_dirs,
	parse_header_origin,
	parse_loki_event_dir,
)


def load_jma_event_jsons(
	base_input_dir: Path,
	event_glob: str = '[0-9]*',
	allowed_event_ids: set[str] | None = None,
) -> pd.DataFrame:
	if not base_input_dir.is_dir():
		raise FileNotFoundError(f'base_input_dir not found: {base_input_dir}')

	rows: list[dict] = []
	for evdir in sorted([p for p in base_input_dir.glob(event_glob) if p.is_dir()]):
		ev = load_event_json(evdir)

		event_id = str(ev.get('event_id', evdir.name))
		if allowed_event_ids is not None and event_id not in allowed_event_ids:
			continue

		# 時刻キーは運用で揺れてOK（origin_time_jst を優先）
		origin_time = ev.get('origin_time_jst', None)
		if origin_time is None:
			origin_time = ev.get('origin_time', None)

		# 震源情報はトップ or ev["extra"] のどっちでも受ける
		lat = ev.get('latitude_deg', None)
		lon = ev.get('longitude_deg', None)
		dep = ev.get('depth_km', None)

		# マグ（任意）: top / extra のどちらでも拾う。無ければ NaN
		mag = ev.get('mag1', None)

		extra = ev.get('extra', None)
		if isinstance(extra, dict):
			if lat is None:
				lat = extra.get('latitude_deg', None)
			if lon is None:
				lon = extra.get('longitude_deg', None)
			if dep is None:
				dep = extra.get('depth_km', None)
			if mag is None:
				mag = extra.get('mag1', None)

		if origin_time is None or lat is None or lon is None or dep is None:
			raise ValueError(
				f'event.json missing required keys: {evdir / "event.json"} '
				f'(need origin_time[_jst], latitude_deg, longitude_deg, depth_km; '
				f'lat/lon/dep may be under extra)'
			)

		origin_time_utc = get_event_origin_utc(ev, event_json_path=evdir / 'event.json')

		rows.append(
			{
				'event_id': event_id,
				'origin_time_jma': str(origin_time),
				'origin_time_jma_utc': origin_time_utc,
				'lat_jma': float(lat),
				'lon_jma': float(lon),
				'depth_km_jma': float(dep),
				'mag_jma': np.nan if mag is None else float(mag),
			}
		)

	df = pd.DataFrame(rows)
	df['origin_time_jma'] = pd.to_datetime(df['origin_time_jma'])
	df['origin_time_jma_utc'] = pd.to_datetime(df['origin_time_jma_utc'], utc=True)
	return df


def load_loki_loc_by_event_id(
	loki_output_dir: Path,
	event_glob: str = '[0-9]*',
	allowed_event_ids: set[str] | None = None,
) -> pd.DataFrame:
	evdirs = list_event_dirs(loki_output_dir, event_glob=event_glob)
	if not evdirs:
		raise ValueError(f"no event dirs in {loki_output_dir} with glob '{event_glob}'")

	rows: list[dict] = []
	for evdir in evdirs:
		if allowed_event_ids is not None and evdir.name not in allowed_event_ids:
			continue
		res = parse_loki_event_dir(evdir)
		origin_loki = infer_event_origin_time_from_loki_result(res)
		# ntrial==1 前提（厳密）
		if len(res.loc_rows) != 1:
			raise ValueError(
				f'.loc must have exactly 1 row for now: {res.loc_path} rows={len(res.loc_rows)}'
			)

		r = res.loc_rows[0]
		rows.append(
			{
				'event_id': res.event_name,
				'origin_time_loki': str(origin_loki),
				'x_km_loki': float(r.x_km),
				'y_km_loki': float(r.y_km),
				'z_km_loki': float(r.z_km),
				'cmax': float(r.cmax),
				'loc_path': str(res.loc_path),
			}
		)

	return pd.DataFrame(rows)


def build_compare_df(
	*,
	base_input_dir: str | Path,
	loki_output_dir: str | Path,
	header_path: str | Path,
	event_glob: str = '[0-9]*',
	allowed_event_ids: set[str] | None = None,
) -> pd.DataFrame:
	base_input_dir = Path(base_input_dir)
	loki_output_dir = Path(loki_output_dir)
	header_path = Path(header_path)

	jma = load_jma_event_jsons(
		base_input_dir, event_glob=event_glob, allowed_event_ids=allowed_event_ids
	)
	loki = load_loki_loc_by_event_id(
		loki_output_dir, event_glob=event_glob, allowed_event_ids=allowed_event_ids
	)

	origin = parse_header_origin(header_path)

	# LOKIのx/y[km]が「lat0/lon0基準」のローカルXY前提なので、geoの逆変換を使う
	x = (loki['x_km_loki'].to_numpy(float)).astype(float)
	y = (loki['y_km_loki'].to_numpy(float)).astype(float)
	lat_arr, lon_arr = local_xy_km_to_latlon(
		x, y, lat0_deg=origin.lat0_deg, lon0_deg=origin.lon0_deg
	)
	loki['lat_loki'] = lat_arr
	loki['lon_loki'] = lon_arr

	df = jma.merge(loki, on='event_id', how='inner')
	if df.empty:
		raise ValueError('no matched events between JMA and LOKI (event_id mismatch?)')

	# ---- origin time error (LOKI - JMA) ----
	df['origin_time_jma'] = pd.to_datetime(df['origin_time_jma'])
	df['origin_time_jma_utc'] = pd.to_datetime(df['origin_time_jma_utc'], utc=True)
	df['origin_time_loki'] = pd.to_datetime(df['origin_time_loki'])

	dt_sec: list[float] = []
	for r in df.itertuples(index=False):
		j = to_utc(pd.Timestamp(r.origin_time_jma_utc), naive_tz='UTC')
		l = to_utc(pd.Timestamp(r.origin_time_loki), naive_tz='UTC')
		dt_sec.append(float((l - j).total_seconds()))
	df['dt_origin_sec'] = dt_sec

	# haversine_distance_km は「基準点→点群」仕様なので、ペア距離は行ごとに計算（厳密優先）
	dh: list[float] = []
	for r in df.itertuples(index=False):
		d = haversine_distance_km(
			float(r.lat_jma),
			float(r.lon_jma),
			np.asarray([float(r.lat_loki)]),
			np.asarray([float(r.lon_loki)]),
		)[0]
		dh.append(float(d))
	df['dh_km'] = dh

	df['dz_km'] = df['z_km_loki'] - df['depth_km_jma']

	dh_arr = df['dh_km'].to_numpy(float)
	dz_arr = df['dz_km'].to_numpy(float)
	df['e3d_km'] = np.sqrt(dh_arr * dh_arr + dz_arr * dz_arr)
	wz = 0.5
	df['e_w3d_km'] = np.sqrt(dh_arr * dh_arr + (wz * dz_arr) * (wz * dz_arr))

	# リンク線生成に便利な列（plot側で pairs 作りやすい）
	df['lon_loki_from_xy'] = df['lon_loki']
	df['lat_loki_from_xy'] = df['lat_loki']
	df['dep_loki_km'] = df['z_km_loki']

	return df
