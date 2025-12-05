# %%
from __future__ import annotations

import pandas as pd
from extract_data import filter_by_das_score
from vis import plot_events_map_and_sections

from common.load_config import load_plot_preset

if __name__ == '__main__':
	csv_path = '/workspace/data/arrivetime/arrivetime_epicenters_mobara2020.csv'
	prefecture_shp = '/workspace/util/N03-20240101_GML/N03-20240101_prefecture.shp'

	epics_df = pd.read_csv(csv_path)
	out_png = './img/Figure_Events_Mobara2020_all.png'
	params = load_plot_preset(
		'/workspace/data/config/plot_config.yaml', 'mobara_default'
	)

	lon_min, lon_max = params['lon_range']
	lat_min, lat_max = params['lat_range']
	depth_min, depth_max = params['depth_range']
	well_coord = params.get('well_coord')  # [lat, lon] or None
	min_mag = params.get('min_mag')
	max_mag = params.get('max_mag')

	extras = []
	if well_coord is not None:
		extras.append(
			{
				'label': 'mobara site',
				'xy': [(well_coord[1], well_coord[0])],  # (lon, lat)
				'marker': 'o',
				'color': 'royalblue',
				'size': 30,
				'annotate': False,
			}
		)

	plot_events_map_and_sections(
		df=epics_df,
		prefecture_shp=prefecture_shp,
		out_png=out_png,
		mag_col='mag1',
		depth_col='depth_km',
		markersize=3,  # 中間マグニチュードの基準サイズ
		min_mag=min_mag,
		max_mag=max_mag,
		lon_range=(lon_min, lon_max),
		lat_range=(lat_min, lat_max),
		depth_range=(depth_min, depth_max),
		extras_xy=extras,
	)
	for das_score in [1, 2, 3, 4]:
		flt_epics_df, _ = filter_by_das_score(
			epics_df, das_min=das_score, das_max=das_score
		)
		out_png = f'./img/Figure_Events_Mobara2020_das{das_score}.png'
		plot_events_map_and_sections(
			df=flt_epics_df,
			prefecture_shp=prefecture_shp,
			out_png=out_png,
			mag_col='mag1',
			depth_col='depth_km',
			markersize=3,
			min_mag=min_mag,
			max_mag=max_mag,
			lon_range=(lon_min, lon_max),
			lat_range=(lat_min, lat_max),
			depth_range=(depth_min, depth_max),
			extras_xy=extras,  # 中間マグニチュードの基準サイズ
		)
# %
