"""Normalization helpers for GaMMA input tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from common.core import validate_columns


def normalize_picks(df: pd.DataFrame) -> pd.DataFrame:
	"""Normalize picks DataFrame into GaMMA expected columns."""
	picks = df.copy()

	if 'phase_time' in picks.columns:
		picks['phase_time'] = pd.to_datetime(
			picks['phase_time'], utc=True, errors='raise'
		)
		rename_map = {
			'station_id': 'id',
			'phase_time': 'timestamp',
			'phase_type': 'type',
			'phase_score': 'prob',
			'phase_amplitude': 'amp',
		}
		for src, dst in rename_map.items():
			if src in picks.columns:
				picks = picks.rename(columns={src: dst})

	validate_columns(picks, ['id', 'timestamp', 'type'], 'picks')

	if 'prob' not in picks.columns:
		picks['prob'] = 1.0
	if 'amp' not in picks.columns:
		picks['amp'] = -1.0

	picks['type'] = picks['type'].astype(str).str.upper()
	picks = picks[picks['type'].isin(['P', 'S'])].reset_index(drop=True)

	return picks[['id', 'timestamp', 'type', 'prob', 'amp']]


def _prepare_stations(df: pd.DataFrame) -> pd.DataFrame:
	sta = df.copy()

	if 'station_id' in sta.columns:
		sta = sta.rename(columns={'station_id': 'id'})

	if 'x(km)' not in sta.columns:
		if 'x_km' in sta.columns:
			sta['x(km)'] = sta['x_km'].astype(float)
		else:
			raise ValueError('stations needs x(km) or x_km')
	if 'y(km)' not in sta.columns:
		if 'y_km' in sta.columns:
			sta['y(km)'] = sta['y_km'].astype(float)
		else:
			raise ValueError('stations needs y(km) or y_km')
	if 'z(km)' not in sta.columns:
		if 'z_depth_km' in sta.columns:
			sta['z(km)'] = sta['z_depth_km'].astype(float)
		else:
			raise ValueError('stations needs z(km) or z_depth_km')

	validate_columns(sta, ['id', 'x(km)', 'y(km)', 'z(km)'], 'stations')
	return sta


def normalize_stations(df: pd.DataFrame) -> pd.DataFrame:
	"""Normalize stations DataFrame into unique station coordinate rows."""
	sta = _prepare_stations(df)
	return (
		sta[['id', 'x(km)', 'y(km)', 'z(km)']]
		.drop_duplicates('id')
		.reset_index(drop=True)
	)


def estimate_station_origin_m(df: pd.DataFrame) -> tuple[float, float]:
	"""Estimate meter-grid origin offsets from station coordinates."""
	sta = _prepare_stations(df)

	if 'E_m' in sta.columns and 'N_m' in sta.columns:
		origin_e_m = float(
			np.median(
				sta['E_m'].to_numpy(dtype=float)
				- sta['x(km)'].to_numpy(dtype=float) * 1000.0
			)
		)
		origin_n_m = float(
			np.median(
				sta['N_m'].to_numpy(dtype=float)
				- sta['y(km)'].to_numpy(dtype=float) * 1000.0
			)
		)
	else:
		origin_e_m = float('nan')
		origin_n_m = float('nan')

	return origin_e_m, origin_n_m
