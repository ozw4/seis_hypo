# %%
# proc/data_profiling/izu2009/vis_izu2009.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

from catalog.selection import extract_events_in_region
from common.load_config import load_config
from jma.monthly_presence_selection import (
	stations_within_radius_from_monthly_presence,
	write_station_lists,
)
from viz.catalog_profile import (
	plot_catalog_events_3view,
	plot_catalog_localxy_events_stations,
	plot_catalog_stations_map_simple,
)
from viz.plot_config import PlotConfig

if __name__ == '__main__':
	# ---- 入力 ----
	plot_config_yaml = '/workspace/data/config/plot_config.yaml'
	plot_setting = 'izu_default'

	epic_csv = '/workspace/data/arrivetime/JMA/arrivetime_epicenters_2009.0.csv'
	monthly_presence_csv = '/workspace/proc/prepare_data/jma/stationcode_match/v1/snapshots/monthly/monthly_presence.csv'
	prefecture_shp = '/workspace/data/N03-20240101_GML/N03-20240101_prefecture.shp'

	start_time = '2009-12-17 00:00:00'
	end_time = '2009-12-20 23:59:59'

	# extract_events_in_region 用（半径はあなたが固定したい値にする）
	event_radius_km = 50

	# station selection（50 km & 34点）
	station_radius_km = 50.0
	pick_n_stations = 34

	out_dir = Path('./profile')
	out_dir.mkdir(parents=True, exist_ok=True)

	# ---- config ----
	params = load_config(PlotConfig, plot_config_yaml, plot_setting)
	if params.well_coord is None:
		raise ValueError('plot_config: well_coord is required')
	center_lat = float(params.well_coord[0])
	center_lon = float(params.well_coord[1])

	# ---- epic_df ----
	epic_df = pd.read_csv(epic_csv)

	# ---- events: extract_events_in_region（指定どおりこれを使う）----
	events_df, _ = extract_events_in_region(
		epic_df=epic_df,
		meas_df=None,
		start_time=start_time,
		end_time=end_time,
		mag_min=params.min_mag,
		mag_max=params.max_mag,
		center_lat=center_lat,
		center_lon=center_lon,
		radius_km=event_radius_km,
	)
	events_df.to_csv(out_dir / 'events_selected.csv', index=False, encoding='utf-8')

	# ---- stations: monthly_presence → 期間内稼働 → 50 km ----
	stations_all = stations_within_radius_from_monthly_presence(
		monthly_presence_csv,
		start_time=start_time,
		end_time=end_time,
		center_lat=center_lat,
		center_lon=center_lon,
		radius_km=station_radius_km,
	)
	stations_all.to_csv(
		out_dir / 'stations_active_within50km_all.csv', index=False, encoding='utf-8'
	)

	stations_34 = stations_all.sort_values('dist_km').head(pick_n_stations).copy()
	stations_34.to_csv(out_dir / 'stations_target34.csv', index=False, encoding='utf-8')

	write_station_lists(stations_34, out_dir=out_dir / 'stations_lists', pick_n=None)

	# ---- viz: mobara 方式（plot_config.yaml の範囲で固定）----
	plot_catalog_events_3view(
		events_df,
		plot_config_yaml=plot_config_yaml,
		plot_setting=plot_setting,
		prefecture_shp=prefecture_shp,
		out_png=out_dir / 'Figure_Events_Izu2009_3view.png',
		station_radius_km=station_radius_km,
	)

	plot_catalog_stations_map_simple(
		stations_34,
		plot_config_yaml=plot_config_yaml,
		plot_setting=plot_setting,
		prefecture_shp=prefecture_shp,
		out_png=out_dir / 'Figure_Stations_target34_simple.png',
		show_station_labels=True,
		station_label_max=60,
	)

	plot_catalog_localxy_events_stations(
		events_df,
		stations_34,
		center_lat=center_lat,
		center_lon=center_lon,
		out_png=out_dir / 'Figure_LocalXY_events_stations.png',
		circle_km=station_radius_km,
		zoom_km=None,
	)
