from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common.geo import local_xy_km_to_latlon


def _event_code_to_int(event_code: str) -> int:
	s = str(event_code).strip()
	if not s.startswith('ev_'):
		raise ValueError(f'unexpected event_id format: {event_code}')
	tail = s.split('_', 1)[1]
	if not tail.isdigit():
		raise ValueError(f'unexpected event_id format: {event_code}')
	return int(tail)


def build_station_df(
	recv_xyz_m: np.ndarray,
	receiver_indices: np.ndarray,
	station_codes_all: list[str] | np.ndarray,
	lat0: float,
	lon0: float,
	*,
	z_is_depth_positive: bool,
) -> pd.DataFrame:
	if recv_xyz_m.ndim != 2 or recv_xyz_m.shape[1] != 3:
		raise ValueError(f'recv geometry must be (N,3), got {recv_xyz_m.shape}')

	n = int(recv_xyz_m.shape[0])

	idx = np.asarray(receiver_indices)
	if idx.ndim != 1:
		raise ValueError('receiver_indices must be a 1D array')
	if idx.size == 0:
		raise ValueError('no stations selected')
	if np.any(np.isnan(idx)):
		raise ValueError('receiver_indices contains NaN')
	if idx.dtype.kind not in ('i', 'u'):
		raise ValueError('receiver_indices must be an integer array')
	idx = idx.astype(int, copy=False)
	if idx.min() < 0 or idx.max() >= n:
		raise ValueError(
			f'receiver index out of range: min={int(idx.min())} max={int(idx.max())} n={n}'
		)
	if np.unique(idx).size != idx.size:
		raise ValueError('receiver_indices has duplicates')

	codes_all = np.asarray(station_codes_all)
	if codes_all.ndim != 1:
		raise ValueError('station_codes_all must be a 1D sequence')
	if codes_all.size != n:
		raise ValueError(
			'station_codes_all length mismatch: '
			f'len={int(codes_all.size)} expected={n}'
		)

	xyz = recv_xyz_m[idx]
	x_km = xyz[:, 0] / 1000.0
	y_km = xyz[:, 1] / 1000.0
	z_m = xyz[:, 2].astype(float)

	# Hypoinverse station file の Elevation は「標高 (positive up)」。
	# 合成データが (x,y,z)=(E,N,depth[+down]) の場合は Elevation_m = -depth_m。
	# (x,y,z)=(E,N,up[+up]) の場合は Elevation_m = +z_m。
	if z_is_depth_positive:
		elevation_m = (-z_m).round().astype(int)
	else:
		elevation_m = (z_m).round().astype(int)

	lat_deg, lon_deg = local_xy_km_to_latlon(x_km, y_km, lat0_deg=lat0, lon0_deg=lon0)

	codes = [str(s) for s in codes_all[idx].tolist()]

	return pd.DataFrame(
		{
			'station_code': codes,
			'receiver_index': idx.astype(int),
			'Latitude_deg': lat_deg.astype(float),
			'Longitude_deg': lon_deg.astype(float),
			'Elevation_m': elevation_m.astype(int),
			'channel': 'HHZ',
			'comp1': 'Z',
			'weight_code': ' ',
			'default_period': 1.0,
		}
	)


def build_truth_df(
	index_csv: Path,
	lat0: float,
	lon0: float,
	origin0: pd.Timestamp,
	dt_sec: float,
	max_events: int,
) -> pd.DataFrame:
	df = pd.read_csv(index_csv)
	for c in ['event_id', 'x_m', 'y_m', 'z_m']:
		if c not in df.columns:
			raise ValueError(f'index.csv missing column: {c}')

	df['event_int'] = df['event_id'].map(_event_code_to_int).astype(int)
	df = df.sort_values('event_int').reset_index(drop=True)

	if max_events > 0:
		df = df.iloc[:max_events].copy()

	x_km = df['x_m'].astype(float).to_numpy() / 1000.0
	y_km = df['y_m'].astype(float).to_numpy() / 1000.0
	z_km = df['z_m'].astype(float).to_numpy() / 1000.0

	lat_deg, lon_deg = local_xy_km_to_latlon(x_km, y_km, lat0_deg=lat0, lon0_deg=lon0)
	origin_times = [origin0 + pd.Timedelta(seconds=i * dt_sec) for i in range(len(df))]

	return pd.DataFrame(
		{
			'event_id_str': df['event_id'].astype(str),
			'event_id': df['event_int'].astype(int),
			'origin_time': origin_times,
			'x_m_true': df['x_m'].astype(float),
			'y_m_true': df['y_m'].astype(float),
			'z_m_true': df['z_m'].astype(float),
			'lat_deg_true': lat_deg.astype(float),
			'lon_deg_true': lon_deg.astype(float),
			'depth_km_true': z_km.astype(float),
		}
	)


def build_epic_df(truth_df: pd.DataFrame, default_depth_km: float) -> pd.DataFrame:
	n = len(truth_df)
	return pd.DataFrame(
		{
			'event_id': truth_df['event_id'].astype(int),
			'origin_time': truth_df['origin_time'],
			'latitude_deg': np.full(n, np.nan),
			'longitude_deg': np.full(n, np.nan),
			'depth_km': np.full(n, float(default_depth_km)),
		}
	)


def build_meas_df(
	events_dir: Path,
	truth_df: pd.DataFrame,
	station_df: pd.DataFrame,
) -> pd.DataFrame:
	nsta = len(station_df)
	idx = station_df['receiver_index'].to_numpy(dtype=int)
	if idx.ndim != 1:
		raise ValueError('station_df.receiver_index must be 1D')
	if idx.size != nsta:
		raise ValueError('station_df.receiver_index length mismatch')

	rows: list[dict] = []
	for _, ev in truth_df.iterrows():
		eid = int(ev['event_id'])
		origin = pd.to_datetime(ev['origin_time'])

		ev_dir = events_dir / str(ev['event_id_str'])
		tt_p_path = ev_dir / 'tt_p_first_true.npy'
		tt_s_path = ev_dir / 'tt_s_first_true.npy'
		if not tt_p_path.is_file():
			raise FileNotFoundError(f'missing: {tt_p_path}')
		if not tt_s_path.is_file():
			raise FileNotFoundError(f'missing: {tt_s_path}')

		tt_p = np.load(tt_p_path).astype(float)
		tt_s = np.load(tt_s_path).astype(float)
		if tt_p.ndim != 1 or tt_s.ndim != 1:
			raise ValueError(f'tt arrays must be 1D: event_id={eid}')
		if idx.min() < 0 or idx.max() >= tt_p.size or idx.max() >= tt_s.size:
			raise ValueError(
				f'tt index out of range: event_id={eid} '
				f'min={int(idx.min())} max={int(idx.max())} '
				f'tt_p={int(tt_p.size)} tt_s={int(tt_s.size)}'
			)
		tt_p = tt_p[idx]
		tt_s = tt_s[idx]

		if len(tt_p) != nsta or len(tt_s) != nsta:
			raise ValueError(f'tt length mismatch: event_id={eid} nsta={nsta}')

		for i in range(nsta):
			code = str(station_df.iloc[i]['station_code'])
			p = tt_p[i]
			s = tt_s[i]

			p_time = (
				origin + pd.Timedelta(seconds=float(p)) if np.isfinite(p) else pd.NaT
			)
			s_time = (
				origin + pd.Timedelta(seconds=float(s)) if np.isfinite(s) else pd.NaT
			)

			rows.append(
				{
					'event_id': eid,
					'station_code': code,
					'phase_name_1': 'P',
					'phase_name_2': 'S',
					'phase1_time': p_time,
					'phase2_time': s_time,
					'pick_flag_1': 'M',
					'pick_flag_2': 'M',
					'pick_flag_3': 'M',
					'pick_flag_4': 'M',
				}
			)

	return pd.DataFrame(rows)
