# %%
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from common.load_config import load_config
from viz.core.fig_io import save_figure
from viz.core.sections3 import make_3view_axes, scatter_points_3view, sync_xyz_ranges
from viz.plot_config import PlotConfig


def load_hypoinverse_catalog(csv_path: str | Path) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	cols = ['lon_deg_hyp', 'lat_deg_hyp', 'depth_km_hyp', 'origin_time_hyp']
	missing = [c for c in cols if c not in df.columns]
	if missing:
		raise ValueError(f'hypoinverse csv に必要な列がありません: {missing}')

	keep = cols + (
		['passed_plot_quality_filter']
		if 'passed_plot_quality_filter' in df.columns
		else []
	)
	df = df.loc[:, keep].copy()
	if 'passed_plot_quality_filter' in df.columns:
		df = df[df['passed_plot_quality_filter'].astype(bool)]

	df = df.rename(
		columns={
			'lon_deg_hyp': 'longitude_deg',
			'lat_deg_hyp': 'latitude_deg',
			'depth_km_hyp': 'depth_km',
			'origin_time_hyp': 'origin_time',
		}
	)
	df['origin_time'] = pd.to_datetime(df['origin_time'])
	df['catalog'] = 'HypoInverse'
	return df[['longitude_deg', 'latitude_deg', 'depth_km', 'origin_time', 'catalog']]


def load_jma_catalog(xlsx_path: str | Path, sheet_name: str | int = 0) -> pd.DataFrame:
	df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=0, skiprows=[1])
	cols = ['経度(Lon)', '緯度(Lat)', '深度(km)', '発生日', '発生時刻']
	missing = [c for c in cols if c not in df.columns]
	if missing:
		raise ValueError(f'JMA xlsx に必要な列がありません: {missing}')

	keep = cols + (['Mj'] if 'Mj' in df.columns else [])
	df = df.loc[:, keep].copy()
	df = df.rename(
		columns={
			'経度(Lon)': 'longitude_deg',
			'緯度(Lat)': 'latitude_deg',
			'深度(km)': 'depth_km',
		}
	)
	df['origin_time'] = pd.to_datetime(
		df['発生日'].astype(str).str.strip()
		+ ' '
		+ df['発生時刻'].astype(str).str.strip()
	)
	df['catalog'] = 'JMA unified catalog'
	return df[['longitude_deg', 'latitude_deg', 'depth_km', 'origin_time', 'catalog']]


def plot_two_catalogs(
	df_hyp: pd.DataFrame,
	df_jma: pd.DataFrame,
	*,
	prefecture_shp: str | Path,
	out_png: str | Path,
	lon_range: tuple[float, float],
	lat_range: tuple[float, float],
	depth_range: tuple[float, float],
	well_coord: tuple[float, float] | None = None,
) -> None:
	plt.rcParams['font.family'] = 'Arial'
	plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.5})

	fig, ax_xy, ax_xz, ax_yz, _ = make_3view_axes(
		figsize=(10, 10),
		width_ratios=(3.0, 1.5),
		height_ratios=(3.0, 1.5),
	)

	pref = gpd.read_file(prefecture_shp)
	if pref.crs is None or pref.crs.to_string().upper() != 'EPSG:4326':
		pref = pref.to_crs('EPSG:4326')
	pref.plot(
		ax=ax_xy,
		facecolor='whitesmoke',
		edgecolor='gray',
		linewidth=0.6,
		zorder=1,
	)

	styles = {
		'HypoInverse': {
			'marker': 'o',
			'facecolors': 'none',
			'edgecolors': 'crimson',
			'linewidths': 0.9,
			's': 28,
			'alpha': 0.95,
			'zorder': 4,
		},
		'JMA unified catalog': {
			'marker': '^',
			'facecolors': 'none',
			'edgecolors': 'dimgray',
			'linewidths': 0.7,
			's': 18,
			'alpha': 0.55,
			'zorder': 3,
		},
	}

	for label, df in [('JMA unified catalog', df_jma), ('HypoInverse', df_hyp)]:
		if df.empty:
			continue
		scatter_points_3view(
			ax_xy,
			ax_xz,
			ax_yz,
			x=df['longitude_deg'].to_numpy(),
			y=df['latitude_deg'].to_numpy(),
			z=df['depth_km'].to_numpy(),
			yz_mode='z-y',
			label=f'{label} (n={len(df)})',
			**styles[label],
		)

	if well_coord is not None:
		site_lat = float(well_coord[0])
		site_lon = float(well_coord[1])
		ax_xy.scatter(
			[site_lon],
			[site_lat],
			marker='*',
			s=120,
			color='royalblue',
			edgecolors='k',
			linewidths=0.5,
			zorder=5,
			label='Mobara site',
		)
		ax_xz.scatter(
			[site_lon],
			[0.0],
			marker='*',
			s=120,
			color='royalblue',
			edgecolors='k',
			linewidths=0.5,
			zorder=5,
		)
		ax_yz.scatter(
			[0.0],
			[site_lat],
			marker='*',
			s=120,
			color='royalblue',
			edgecolors='k',
			linewidths=0.5,
			zorder=5,
		)

	ax_xy.set_ylabel('Latitude', fontsize=10)
	ax_xy.set_aspect('auto')
	ax_xz.set_xlabel('Longitude', fontsize=10)
	ax_xz.set_ylabel('Depth (km)', fontsize=10)
	ax_yz.set_xlabel('Depth (km)', fontsize=10)

	sync_xyz_ranges(
		ax_xy,
		ax_xz,
		ax_yz,
		x_range=lon_range,
		y_range=lat_range,
		z_range=depth_range,
		invert_z=True,
		yz_mode='z-y',
	)

	if df_hyp.empty and df_jma.empty:
		raise RuntimeError('両方のカタログが空です。')
	if df_hyp.empty:
		t_min = df_jma['origin_time'].min()
		t_max = df_jma['origin_time'].max()
	elif df_jma.empty:
		t_min = df_hyp['origin_time'].min()
		t_max = df_hyp['origin_time'].max()
	else:
		t_min = min(df_hyp['origin_time'].min(), df_jma['origin_time'].min())
		t_max = max(df_hyp['origin_time'].max(), df_jma['origin_time'].max())

	fig.suptitle(
		'Earthquake Events comparison\n'
		f'{t_min:%Y-%m-%d %H:%M:%S} - {t_max:%Y-%m-%d %H:%M:%S}',
		fontsize=10,
		y=ax_xy.get_position().y1 + 0.05,
	)
	ax_xy.legend(loc='lower right', fontsize=8, framealpha=0.8)
	fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95], pad=0.2)
	save_figure(fig, out_png, dpi=300, bbox_inches='tight', close=True)


if __name__ == '__main__':
	hyp_csv = '/workspace/proc/hypocenter_determination/jma_mobara_hypoinverse/result/test_mobara2025_jma_fpevent/hypoinverse_events_after_plot_quality_filter.csv'
	jma_xlsx = '/workspace/data/waveform/気象庁一元化震源カタログ抽出_2025年観測データ_茂原試験サイト周辺140km四方限定.xlsx'
	prefecture_shp = '/workspace/data/N03-20240101_GML/N03-20240101_prefecture.shp'
	out_png = './img/Figure_Events_Mobara_compare_hypoinverse_vs_jma.png'

	params = load_config(
		PlotConfig,
		'/workspace/data/config/plot_config.yaml',
		'mobara_default',
	)

	df_hyp = load_hypoinverse_catalog(hyp_csv)
	df_jma = load_jma_catalog(jma_xlsx)

	plot_two_catalogs(
		df_hyp,
		df_jma,
		prefecture_shp=prefecture_shp,
		out_png=out_png,
		lon_range=params.lon_range,
		lat_range=params.lat_range,
		depth_range=params.depth_range,
		well_coord=params.well_coord,
	)
