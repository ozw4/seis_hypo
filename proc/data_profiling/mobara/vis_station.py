# %%
from __future__ import annotations

from viz.stations_map import plot_stations_by_affiliation_from_station_csv

if __name__ == '__main__':
	plot_stations_by_affiliation_from_station_csv(
		station_csv='/workspace/data/station/jma/station.csv',
		prefecture_shp='/workspace/data/N03-20240101_GML/N03-20240101_prefecture.shp',
		out_png='img/Figure_Stations_all.png',
		station_codes=None,  # サブセットにしたいときは ["ABASH2", ...] を渡す
		marker='^',
		markersize=10,
		fontsize=8,
		label_dlat=0.03,
		cmap_name='tab20',
		affiliation_colors={
			'NIED': 'tab:gray',
			'NIED F-net': 'tab:orange',
			'JMA': 'tab:blue',
		},
		show_station_labels=False,  # 全局描画時は False 推奨
	)

	plot_stations_by_affiliation_from_station_csv(
		station_csv='/workspace/data/station/jma/station.csv',
		prefecture_shp='/workspace/data/N03-20240101_GML/N03-20240101_prefecture.shp',
		out_png='img/Figure_Stations_NIED.png',
		station_codes=None,  # サブセットにしたいときは ["ABASH2", ...] を渡す
		marker='^',
		markersize=10,
		fontsize=8,
		label_dlat=0.03,
		cmap_name='tab20',
		affiliation_colors={'NIED': 'tab:gray', 'NIED F-net': 'tab:orange'},
		affiliation_filter=['NIED', 'NIED F-net'],
		show_station_labels=False,  # 全局描画時は False 推奨
	)

	plot_stations_by_affiliation_from_station_csv(
		station_csv='/workspace/data/station/jma/station.csv',
		prefecture_shp='/workspace/data/N03-20240101_GML/N03-20240101_prefecture.shp',
		out_png='img/Figure_Stations_JMA.png',
		station_codes=None,  # サブセットにしたいときは ["ABASH2", ...] を渡す
		marker='^',
		markersize=10,
		fontsize=8,
		label_dlat=0.03,
		cmap_name='tab20',
		affiliation_colors={'JMA': 'tab:blue'},
		affiliation_filter=['JMA', 'JMA Intensity'],
		show_station_labels=False,  # 全局描画時は False 推奨
	)

	plot_stations_by_affiliation_from_station_csv(
		station_csv='/workspace/data/station/jma/station.csv',
		prefecture_shp='/workspace/data/N03-20240101_GML/N03-20240101_prefecture.shp',
		out_png='img/Figure_Stations_NIED.png',
		station_codes=None,  # サブセットにしたいときは ["ABASH2", ...] を渡す
		marker='^',
		markersize=10,
		fontsize=8,
		label_dlat=0.03,
		cmap_name='tab20',
		affiliation_colors={'NIED': 'tab:gray', 'NIED F-net': 'tab:orange'},
		affiliation_filter=['NIED', 'NIED F-net'],
		show_station_labels=False,  # 全局描画時は False 推奨
		config_name='mobara_default',
	)
