# %%
# file: proc/loki_hypo/plot_loki_with_existing_event_plot.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

from loki_tools.loki_parse import load_loki_catalogue_as_events_df

# 既存の描画関数（ユーザー提示のもの）を import して使う想定
# 例: from your_module import plot_events_map_and_sections
from viz.events_map import plot_events_map_and_sections


def _load_plot_ranges(
	plot_config_yaml: str | Path,
	plot_setting: str,
) -> tuple[
	tuple[float, float], tuple[float, float], tuple[float, float]
]:
	from common.load_config import load_config
	from viz.plot_config import PlotConfig

	plot_config_yaml = Path(plot_config_yaml)
	if not plot_config_yaml.is_file():
		raise FileNotFoundError(f'plot_config_yaml not found: {plot_config_yaml}')

	params = load_config(PlotConfig, plot_config_yaml, plot_setting)
	return (
		tuple(params.lat_range),
		tuple(params.lon_range),
		tuple(params.depth_range),
	)


def make_loki_plot_df(compare_df: pd.DataFrame) -> pd.DataFrame:
	req = {
		'event_id',
		'origin_time_jma',
		'lat_loki',
		'lon_loki',
		'dep_loki_km',
		'cmax',
		'mag_jma',
	}
	missing = req - set(compare_df.columns)
	if missing:
		raise ValueError(f'compare_df missing columns: {sorted(missing)}')

	return pd.DataFrame(
		{
			'event_id': compare_df['event_id'].astype(str),
			'origin_time': pd.to_datetime(compare_df['origin_time_jma']),
			'latitude_deg': compare_df['lat_loki'].astype(float),
			'longitude_deg': compare_df['lon_loki'].astype(float),
			'depth_km': compare_df['dep_loki_km'].astype(float),
			'cmax': compare_df['cmax'].astype(float),
			'mag_jma': compare_df['mag_jma'].astype(float),
		}
	)


def make_extras_lld_jma(compare_df: pd.DataFrame) -> list[dict]:
	req = {'lon_jma', 'lat_jma', 'depth_km_jma', 'mag_jma'}
	missing = req - set(compare_df.columns)
	if missing:
		raise ValueError(
			f'compare_df missing columns for JMA extras: {sorted(missing)}'
		)

	lld = list(
		zip(
			compare_df['lon_jma'].astype(float).to_list(),
			compare_df['lat_jma'].astype(float).to_list(),
			compare_df['depth_km_jma'].astype(float).to_list(),
			strict=True,
		)
	)
	mags = compare_df['mag_jma'].astype(float).to_list()
	return [
		{
			'label': 'JMA',
			'marker': 'o',
			'color': 'lightcoral',
			'alpha': 0.6,
			'size': 90.0,
			'mag': mags,
			'annotate': False,
			'names': None,
			'lld': lld,
		}
	]


def make_links_lld(compare_df: pd.DataFrame) -> list[dict]:
	req = {'lon_loki', 'lat_loki', 'dep_loki_km', 'lon_jma', 'lat_jma', 'depth_km_jma'}
	missing = req - set(compare_df.columns)
	if missing:
		raise ValueError(f'compare_df missing columns for links: {sorted(missing)}')

	pairs: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
	for r in compare_df.itertuples(index=False):
		pairs.append(
			(
				(
					float(r.lon_loki),
					float(r.lat_loki),
					float(r.dep_loki_km),
				),
				(
					float(r.lon_jma),
					float(r.lat_jma),
					float(r.depth_km_jma),
				),
			)
		)

	return [
		{
			'label': 'LOKI→JMA',
			'pairs': pairs,
			'color': 'black',
			'linewidth': 0.6,
			'alpha': 0.35,
		}
	]


def plot_loki_results_quickcheck(
	*,
	loki_output_dir: str | Path,
	prefecture_shp: str | Path,
	out_png: str | Path,
	lat_range: tuple[float, float] | None = None,
	lon_range: tuple[float, float] | None = None,
	depth_range: tuple[float, float] | None = None,
	start_time: str | None = None,
	end_time: str | None = None,
	min_cmax: float | None = None,
	max_cmax: float | None = None,
) -> None:
	df = load_loki_catalogue_as_events_df(loki_output_dir)

	if start_time is not None:
		df = df[df['origin_time'] >= pd.to_datetime(start_time)]
	if end_time is not None:
		df = df[df['origin_time'] <= pd.to_datetime(end_time)]

	if min_cmax is not None:
		df = df[df['cmax'] >= float(min_cmax)]
	if max_cmax is not None:
		df = df[df['cmax'] <= float(max_cmax)]

	if df.empty:
		raise RuntimeError('no rows to plot after filtering')

	plot_events_map_and_sections(
		df,
		prefecture_shp=prefecture_shp,
		out_png=out_png,
		mag_col='cmax',  # ← magnitude の代わりに coherence を色/サイズに使う
		depth_col='depth_km',
		origin_time_col='origin_time',
		lat_col='latitude_deg',
		lon_col='longitude_deg',
		lat_range=lat_range,
		lon_range=lon_range,
		depth_range=depth_range,
		start_time=start_time,
		end_time=end_time,
	)


if __name__ == '__main__':
	loki_output_dir = '/workspace/proc/loki_hypo/loki_output_mobara/mobara'
	plot_setting = 'mobara_default'
	lat_range, lon_range, depth_range = _load_plot_ranges(
		'/workspace/data/config/plot_config.yaml',
		plot_setting,
	)
	plot_loki_results_quickcheck(
		loki_output_dir=loki_output_dir,
		prefecture_shp='/workspace/data/N03-20240101_GML/N03-20240101_prefecture.shp',
		out_png=f'{loki_output_dir}/loki_quickcheck.png',
		lat_range=lat_range,
		lon_range=lon_range,
		depth_range=depth_range,
	)
