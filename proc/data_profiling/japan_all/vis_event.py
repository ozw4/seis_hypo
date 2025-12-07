# %%
from __future__ import annotations

import pandas as pd

from common.load_config import load_plot_preset
from viz.events_map import plot_events_map_and_sections

if __name__ == '__main__':
	csv_path = '/workspace/data/arrivetime/arrivetime_epicenters.csv'
	prefecture_shp = '/workspace/data/N03-20240101_GML/N03-20240101_prefecture.shp'
	params = load_plot_preset(
		'/workspace/data/config/plot_config.yaml', 'japan_default'
	)
	lon_min, lon_max = params['lon_range']
	lat_min, lat_max = params['lat_range']
	depth_min, depth_max = params['depth_range']
	well_coord = params.get('well_coord')  # [lat, lon] or None
	min_mag = params.get('min_mag')
	max_mag = params.get('max_mag')

	# 2002-06 〜 2025-10 を 1 ヶ月ごとにループ
	months = pd.period_range(start='2002-06', end='2025-10', freq='M')
	df = pd.read_csv(csv_path)
	for ym in months:
		start_time = ym.to_timestamp(how='start')  # その月の1日 00:00
		end_time = (ym + 1).to_timestamp(how='start')  # 翌月1日 00:00（上限）

		out_png = f'./img/event/Figure_Events_{start_time:%Y%m}.png'
		print(f'Processing {start_time:%Y-%m} -> {out_png}')

		plot_events_map_and_sections(
			df=df,
			prefecture_shp=prefecture_shp,
			out_png=out_png,
			mag_col='mag1',
			depth_col='depth_km',
			start_time=start_time,
			end_time=end_time,
			markersize=3,  # 中間マグニチュードの基準サイズ
			min_mag=min_mag,
			max_mag=max_mag,
			lon_range=(lon_min, lon_max),
			lat_range=(lat_min, lat_max),
			depth_range=(depth_min, depth_max),
		)
# %%
