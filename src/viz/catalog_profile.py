# src/viz/catalog_profile.py
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.geo import latlon_to_local_xy_km, local_xy_km_to_latlon
from common.load_config import load_config
from viz.core.fig_io import save_figure
from viz.events_map import plot_events_map_and_sections
from viz.plot_config import PlotConfig


def circle_lonlat_points(
	*, center_lat: float, center_lon: float, radius_km: float, step_deg: float = 2.0
) -> list[tuple[float, float]]:
	theta = np.deg2rad(np.arange(0.0, 360.0 + step_deg, step_deg))
	x_km = float(radius_km) * np.cos(theta)
	y_km = float(radius_km) * np.sin(theta)
	lat, lon = local_xy_km_to_latlon(
		x_km, y_km, lat0_deg=float(center_lat), lon0_deg=float(center_lon)
	)
	return [(float(lo), float(la)) for lo, la in zip(lon, lat, strict=True)]


def plot_catalog_events_3view(
	events_df: pd.DataFrame,
	*,
	plot_config_yaml: str | Path,
	prefecture_shp: str | Path,
	out_png: str | Path,
	plot_setting: str,
	station_radius_km: float | None = None,
) -> None:
	"""plot_config.yaml のレンジ設定に基づき、
	(1) lon-lat map (2) lon-depth (3) lat-depth を描く（mobara と同じ系）
	"""
	params = load_config(PlotConfig, plot_config_yaml, plot_setting)

	if params.well_coord is None:
		raise ValueError('plot_config: well_coord is required')
	center_lat, center_lon = float(params.well_coord[0]), float(params.well_coord[1])

	extras = [
		{
			'label': 'center',
			'xy': [(center_lon, center_lat)],
			'marker': 'o',
			'color': 'royalblue',
			'size': 30,
			'annotate': False,
		}
	]
	if station_radius_km is not None:
		extras.append(
			{
				'label': f'{station_radius_km:g} km',
				'xy': circle_lonlat_points(
					center_lat=center_lat,
					center_lon=center_lon,
					radius_km=float(station_radius_km),
				),
				'marker': '.',
				'color': 'black',
				'size': 6,
				'annotate': False,
			}
		)

	plot_events_map_and_sections(
		df=events_df,
		prefecture_shp=prefecture_shp,
		out_png=out_png,
		mag_col='mag1',
		depth_col='depth_km',
		markersize=3,
		min_mag=params.min_mag,
		max_mag=params.max_mag,
		lon_range=params.lon_range,
		lat_range=params.lat_range,
		depth_range=params.depth_range,
		extras_xy=extras,
	)


def plot_catalog_stations_map_simple(
	stations_df: pd.DataFrame,
	*,
	plot_config_yaml: str | Path,
	prefecture_shp: str | Path,
	out_png: str | Path,
	plot_setting: str,
	show_station_labels: bool = True,
	station_label_max: int = 60,
) -> None:
	"""都道府県境界 + station散布図（plot_config の lon/lat レンジで固定）"""
	params = load_config(PlotConfig, plot_config_yaml, plot_setting)

	req = {'lat', 'lon', 'station'}
	missing = req.difference(stations_df.columns)
	if missing:
		raise ValueError(f'stations_df missing columns: {sorted(missing)}')

	pref = gpd.read_file(prefecture_shp)
	if pref.crs is None:
		pref = pref.set_crs('EPSG:4326')
	elif str(pref.crs).lower() != 'epsg:4326':
		pref = pref.to_crs('EPSG:4326')

	fig, ax = plt.subplots(figsize=(7.0, 7.0))
	pref.boundary.plot(ax=ax, linewidth=0.6, color='black')

	ax.scatter(
		stations_df['lon'].to_numpy(float),
		stations_df['lat'].to_numpy(float),
		s=40.0,
		marker='^',
		edgecolors='black',
		facecolors='white',
		linewidths=0.8,
		zorder=3,
	)

	if show_station_labels:
		n = len(stations_df)
		step = max(1, int(np.ceil(n / max(1, int(station_label_max)))))
		df = stations_df.reset_index(drop=True)
		for i in range(0, n, step):
			ax.text(
				float(df.at[i, 'lon']) + 0.005,
				float(df.at[i, 'lat']) + 0.005,
				str(df.at[i, 'station']),
				fontsize=7,
				zorder=4,
			)

	ax.set_xlim(float(params.lon_range[0]), float(params.lon_range[1]))
	ax.set_ylim(float(params.lat_range[0]), float(params.lat_range[1]))
	ax.set_xlabel('Longitude')
	ax.set_ylabel('Latitude')
	ax.set_title('Stations')
	ax.grid(True, linewidth=0.5)

	save_figure(fig, out_png, dpi=200, bbox_inches='tight')


def plot_catalog_localxy_events_stations(
	events_df: pd.DataFrame,
	stations_df: pd.DataFrame,
	*,
	center_lat: float,
	center_lon: float,
	out_png: str | Path,
	circle_km: float | None = None,
	zoom_km: float | None = None,
) -> None:
	"""Center を原点とした local XY(km) で events/stations を同時表示"""
	evx, evy = latlon_to_local_xy_km(
		events_df['latitude_deg'].to_numpy(float),
		events_df['longitude_deg'].to_numpy(float),
		lat0_deg=float(center_lat),
		lon0_deg=float(center_lon),
	)
	stx, sty = latlon_to_local_xy_km(
		stations_df['lat'].to_numpy(float),
		stations_df['lon'].to_numpy(float),
		lat0_deg=float(center_lat),
		lon0_deg=float(center_lon),
	)

	fig, ax = plt.subplots(figsize=(7.0, 7.0))
	ax.scatter(evx, evy, s=6.0, alpha=0.5, label='events')
	ax.scatter(stx, sty, s=30.0, marker='^', label='stations')

	for x, y, sta in zip(
		stx, sty, stations_df['station'].astype(str).to_numpy(), strict=False
	):
		ax.text(float(x) + 0.2, float(y) + 0.2, sta, fontsize=7)

	if circle_km is not None:
		th = np.deg2rad(np.arange(0.0, 361.0, 1.0))
		ax.plot(
			float(circle_km) * np.cos(th),
			float(circle_km) * np.sin(th),
			linewidth=1.0,
			label=f'{circle_km:g} km',
		)

	ax.set_aspect('equal', adjustable='box')
	ax.set_xlabel('East [km]')
	ax.set_ylabel('North [km]')
	ax.grid(True, linewidth=0.5)
	ax.legend()

	if zoom_km is not None:
		z = float(zoom_km)
		ax.set_xlim(-z, z)
		ax.set_ylim(-z, z)

	save_figure(fig, out_png, dpi=200, bbox_inches='tight')
