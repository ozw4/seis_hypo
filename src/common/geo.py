# src/common/geo.py
from __future__ import annotations

from math import cos, radians
from typing import Any

import numpy as np
import pandas as pd


def haversine_distance_km(
	lat0_deg: float,
	lon0_deg: float,
	lat_deg: np.ndarray,
	lon_deg: np.ndarray,
) -> np.ndarray:
	"""(lat0, lon0) と (lat, lon) 配列の大円距離[km]."""
	r_km = 6371.0
	lat0 = np.deg2rad(float(lat0_deg))
	lon0 = np.deg2rad(float(lon0_deg))
	lat = np.deg2rad(lat_deg.astype(float))
	lon = np.deg2rad(lon_deg.astype(float))

	dlat = lat - lat0
	dlon = lon - lon0
	a = np.sin(dlat / 2.0) ** 2 + np.cos(lat0) * np.cos(lat) * np.sin(dlon / 2.0) ** 2
	c = 2.0 * np.arcsin(np.sqrt(a))
	return r_km * c


def _latlon_to_xy(lat, lon):
	lat = np.asarray(lat, float)
	lon = np.asarray(lon, float)
	R = 6371000.0
	lat0 = np.deg2rad(np.nanmean(lat))
	x = R * np.deg2rad(lon - np.nanmean(lon)) * np.cos(lat0)
	y = R * np.deg2rad(lat - np.nanmean(lat))
	return x, y  # x: East, y: North


def latlon_to_local_xy_km(
	lat_deg: Any,
	lon_deg: Any,
	*,
	lat0_deg: float,
	lon0_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
	"""緯度経度 (deg) を基準点 (lat0_deg, lon0_deg) まわりのローカルXY (km) に変換する。

	変換モデル
	----------
	equirectangular（等距離円筒）近似による簡易変換。
	- y は緯度差に比例
	- x は経度差に cos(lat0) を掛けたものに比例

	想定用途
	----------
	- LOKI/NonLinLoc のグリッド設定やQCなど、地域スケールの座標整理。
	- 厳密な測地変換ではないため、広域（数百kmを大きく超えるなど）では誤差が増える。

	座標系
	----------
	- x: East（東向きが正）
	- y: North（北向きが正）

	引数
	----------
	lat_deg, lon_deg:
	    array-like（list/np.ndarray/pd.Series 等）
	lat0_deg, lon0_deg:
	    基準点の緯度経度 [deg]

	戻り値
	----------
	(x_km, y_km):
	    numpy.ndarray
	"""
	lat = np.asarray(lat_deg, dtype=float)
	lon = np.asarray(lon_deg, dtype=float)

	lat0_rad = radians(lat0_deg)
	km_per_deg_lat = 111.32
	km_per_deg_lon = 111.32 * cos(lat0_rad)

	x_km = (lon - lon0_deg) * km_per_deg_lon
	y_km = (lat - lat0_deg) * km_per_deg_lat
	return x_km, y_km


def local_xy_km_to_latlon(
	x_km: Any,
	y_km: Any,
	*,
	lat0_deg: float,
	lon0_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
	"""基準点 (lat0_deg, lon0_deg) まわりのローカルXY (km) を緯度経度 (deg) に逆変換する。

	変換モデル
	----------
	latlon_to_local_xy_km と同じ equirectangular 近似に基づく逆変換。

	座標系
	----------
	- x: East（東向きが正）
	- y: North（北向きが正）

	引数
	----------
	x_km, y_km:
	    array-like（list/np.ndarray/pd.Series 等）
	lat0_deg, lon0_deg:
	    基準点の緯度経度 [deg]

	戻り値
	----------
	(lat_deg, lon_deg):
	    numpy.ndarray
	"""
	x = np.asarray(x_km, dtype=float)
	y = np.asarray(y_km, dtype=float)

	lat0_rad = radians(lat0_deg)
	km_per_deg_lat = 111.32
	km_per_deg_lon = 111.32 * cos(lat0_rad)

	lat_deg = lat0_deg + (y / km_per_deg_lat)
	lon_deg = lon0_deg + (x / km_per_deg_lon)
	return lat_deg, lon_deg


# ---- 並び順 ----
def compute_station_order(
	station_df: pd.DataFrame, mode: str = 'pca', azimuth_deg: float | None = None
) -> np.ndarray:
	lat = station_df['lat'].to_numpy(float)
	lon = station_df['lon'].to_numpy(float)
	x, y = _latlon_to_xy(lat, lon)

	if mode == 'lat':
		order = np.argsort(y)
	elif mode == 'lon':
		order = np.argsort(x)
	elif mode == 'azimuth':
		if azimuth_deg is None:
			raise ValueError(
				"mode='azimuth' では azimuth_deg を指定してください（0°=北, 90°=東）"
			)
		th = np.deg2rad(azimuth_deg)
		ux, uy = np.sin(th), np.cos(th)
		s = x * ux + y * uy
		order = np.argsort(s)
	elif mode == 'pca':
		XY = np.column_stack([x - x.mean(), y - y.mean()])
		_, _, Vt = np.linalg.svd(XY, full_matrices=False)
		v = Vt[0]
		s = XY @ v
		order = np.argsort(s)
	else:
		raise ValueError(f'unknown mode: {mode}')
	return order
