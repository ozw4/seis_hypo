# src/common/geo.py
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
