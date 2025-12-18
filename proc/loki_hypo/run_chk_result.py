# %%
#!/usr/bin/env python3
# proc/loki_hypo/run_plot_compare_jma_loki.py
#
# JMA(event.json) と LOKI(.loc) を比較して、
# - LOKI点群（cmaxで色/サイズ）
# - JMA点群（extras_lld）
# - 対応線（links_lld）
# を plot_events_map_and_sections で1枚に描く（直書き運用版）

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.load_config import load_config
from loki_tools.compare_df import build_compare_df
from loki_tools.plot import (
	make_extras_lld_jma,
	make_links_lld,
	make_loki_plot_df,
)
from loki_tools.plot_error_stats import (
	iqr_outliers,
	make_coherence_bins,
	make_mag_bins,
	pick_mag_column,
	plot_box,
	plot_hist,
)
from viz.events_map import plot_events_map_and_sections
from viz.plot_config import PlotConfig


def main() -> None:
	# =========================
	# ここを環境に合わせて直書き
	# =========================
	USE_BUILD_COMPARE_DF = True

	base_input_dir = Path('/workspace/data/waveform')
	loki_output_dir = Path('/workspace/proc/loki_hypo/loki_output_mobara')
	header_path = Path('/workspace/proc/loki_hypo/mobara_traveltime/db/header.hdr')
	event_glob = '[0-9]*'

	compare_csv_in = loki_output_dir / 'compare_jma_vs_loki.csv'

	out_png = loki_output_dir / 'loki_vs_jma.png'
	out_csv = compare_csv_in

	error_out_dir = loki_output_dir / 'error_stats'
	coherence_col = 'cmax'
	n_coh_bins = 4
	top_n = 10

	plot_setting = 'mobara_default'
	plot_config_yaml = Path('/workspace/data/config/plot_config.yaml')

	# =========================

	params = load_config(PlotConfig, plot_config_yaml, plot_setting)
	lon_range = tuple(params.lon_range)
	lat_range = tuple(params.lat_range)
	depth_range = tuple(params.depth_range)

	compare_df = build_compare_df(
		base_input_dir=base_input_dir,
		loki_output_dir=loki_output_dir,
		header_path=header_path,
		event_glob=event_glob,
	)
	out_csv.parent.mkdir(parents=True, exist_ok=True)
	compare_df.to_csv(out_csv, index=False)

	# ---- map/sections ----
	df_loki_plot = make_loki_plot_df(compare_df)
	extras_lld = make_extras_lld_jma(compare_df)
	links_lld = make_links_lld(compare_df)

	out_png.parent.mkdir(parents=True, exist_ok=True)
	plot_events_map_and_sections(
		df_loki_plot,
		out_png=out_png,
		mag_col=coherence_col,
		depth_col='depth_km',
		origin_time_col='origin_time',
		lat_col='latitude_deg',
		lon_col='longitude_deg',
		lon_range=lon_range,
		lat_range=lat_range,
		depth_range=depth_range,
		extras_lld=extras_lld,
		links_lld=links_lld,
	)

	# ---- error stats ----
	df = compare_df.copy()
	print(df.columns.tolist())
	required = {'event_id', 'origin_time_jma', 'dh_km', 'dz_km', coherence_col}
	missing = required - set(df.columns)
	if missing:
		raise ValueError(f'compare_df missing columns: {sorted(missing)}')

	df['origin_time_jma'] = pd.to_datetime(df['origin_time_jma'])
	df['dh_km'] = df['dh_km'].astype(float)
	df['dz_km'] = df['dz_km'].astype(float)
	df[coherence_col] = df[coherence_col].astype(float)

	error_out_dir.mkdir(parents=True, exist_ok=True)

	plot_hist(
		df['dh_km'].to_numpy(),
		title='Horizontal error dh_km (LOKI vs JMA)',
		xlabel='dh_km [km]',
		out_png=error_out_dir / 'dh_km_hist.png',
		bins=20,
	)
	plot_hist(
		df['dz_km'].to_numpy(),
		title='Depth error dz_km (LOKI - JMA)',
		xlabel='dz_km [km]',
		out_png=error_out_dir / 'dz_km_hist.png',
		bins=20,
	)

	plot_box(
		[df['dh_km'].to_numpy()],
		labels=['all'],
		title='dh_km boxplot (all)',
		ylabel='dh_km [km]',
		out_png=error_out_dir / 'dh_km_box_all.png',
	)
	plot_box(
		[df['dz_km'].to_numpy()],
		labels=['all'],
		title='dz_km boxplot (all)',
		ylabel='dz_km [km]',
		out_png=error_out_dir / 'dz_km_box_all.png',
	)

	df['coh_bin'] = make_coherence_bins(df, coherence_col, n_bins=n_coh_bins)
	labels = sorted(df['coh_bin'].unique())
	dh_by = [df.loc[df['coh_bin'] == b, 'dh_km'].to_numpy() for b in labels]
	dz_by = [df.loc[df['coh_bin'] == b, 'dz_km'].to_numpy() for b in labels]

	plot_box(
		dh_by,
		labels=labels,
		title=f'dh_km by coherence bins ({coherence_col})',
		ylabel='dh_km [km]',
		out_png=error_out_dir / f'dh_km_box_by_{coherence_col}.png',
	)
	plot_box(
		dz_by,
		labels=labels,
		title=f'dz_km by coherence bins ({coherence_col})',
		ylabel='dz_km [km]',
		out_png=error_out_dir / f'dz_km_box_by_{coherence_col}.png',
	)

	dh_out = iqr_outliers(df, 'dh_km')
	dz_out = iqr_outliers(df, 'dz_km')

	dh_top = df.sort_values('dh_km', ascending=False).head(top_n)
	dz_top_abs = (
		df.assign(dz_abs=np.abs(df['dz_km']))
		.sort_values('dz_abs', ascending=False)
		.head(top_n)
	)

	dh_out.sort_values('dh_km', ascending=False).to_csv(
		error_out_dir / 'outliers_dh_km_iqr.csv', index=False
	)
	dz_out.assign(dz_abs=np.abs(dz_out['dz_km'])).sort_values(
		'dz_abs', ascending=False
	).to_csv(
		error_out_dir / 'outliers_dz_km_iqr.csv',
		index=False,
	)
	dh_top.to_csv(error_out_dir / f'top{top_n}_dh_km.csv', index=False)
	dz_top_abs.to_csv(error_out_dir / f'top{top_n}_abs_dz_km.csv', index=False)

	print(f'Wrote: {out_png}')
	print(f'Wrote: {out_csv}')
	print(f'Wrote plots/csv under: {error_out_dir}')

	mag_col = pick_mag_column(df)
	df['mag_jma'] = pd.to_numeric(df[mag_col], errors='raise').astype(float)

	# NaNマグは落とす（誤差解析に使えないので）
	df_mag = df.dropna(subset=['mag_jma']).copy()

	df_mag['mag_bin'], mag_labels = make_mag_bins(df_mag['mag_jma'])

	# --- 箱ひげ（マグ別）---
	dh_by_mag = [
		df_mag.loc[df_mag['mag_bin'] == b, 'dh_km'].to_numpy() for b in mag_labels
	]
	dz_by_mag = [
		df_mag.loc[df_mag['mag_bin'] == b, 'dz_km'].to_numpy() for b in mag_labels
	]

	# 空bin除外
	nz = [(b, a, z) for b, a, z in zip(mag_labels, dh_by_mag, dz_by_mag) if len(a) > 0]
	mag_labels_nz = [b for b, _, _ in nz]
	dh_by_mag_nz = [a for _, a, _ in nz]
	dz_by_mag_nz = [z for _, _, z in nz]

	plot_box(
		dh_by_mag_nz,
		labels=mag_labels_nz,
		title='dh_km by JMA magnitude bins',
		ylabel='dh_km [km]',
		out_png=error_out_dir / 'dh_km_box_by_mag.png',
	)
	plot_box(
		dz_by_mag_nz,
		labels=mag_labels_nz,
		title='dz_km by JMA magnitude bins',
		ylabel='dz_km [km]',
		out_png=error_out_dir / 'dz_km_box_by_mag.png',
	)

	# --- 散布図（マグ vs 誤差）---
	fig = plt.figure(figsize=(9, 4.8))
	ax = fig.add_subplot(111)
	ax.scatter(df_mag['mag_jma'].to_numpy(), df_mag['dh_km'].to_numpy())
	ax.set_title('dh_km vs JMA magnitude')
	ax.set_xlabel('JMA magnitude')
	ax.set_ylabel('dh_km [km]')
	fig.tight_layout()
	fig.savefig(error_out_dir / 'dh_km_vs_mag.png', dpi=200)
	plt.close(fig)

	fig = plt.figure(figsize=(9, 4.8))
	ax = fig.add_subplot(111)
	ax.scatter(df_mag['mag_jma'].to_numpy(), df_mag['dz_km'].to_numpy())
	ax.set_title('dz_km vs JMA magnitude')
	ax.set_xlabel('JMA magnitude')
	ax.set_ylabel('dz_km [km]')
	fig.tight_layout()
	fig.savefig(error_out_dir / 'dz_km_vs_mag.png', dpi=200)
	plt.close(fig)

	# --- 集計CSV（マグ別に count/median/mean 等）---
	summary = df_mag.groupby('mag_bin', as_index=False).agg(
		n=('event_id', 'count'),
		mag_min=('mag_jma', 'min'),
		mag_max=('mag_jma', 'max'),
		dh_median=('dh_km', 'median'),
		dh_mean=('dh_km', 'mean'),
		dh_p90=('dh_km', lambda x: float(np.quantile(x.astype(float), 0.9))),
		dz_median=('dz_km', 'median'),
		dz_mean=('dz_km', 'mean'),
		dz_p90_abs=(
			'dz_km',
			lambda x: float(np.quantile(np.abs(x.astype(float)), 0.9)),
		),
	)
	summary.to_csv(error_out_dir / 'summary_by_mag.csv', index=False)


if __name__ == '__main__':
	main()
