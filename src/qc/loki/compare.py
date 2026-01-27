from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from loki_tools.compare_df import build_compare_df
from loki_tools.plot import make_extras_lld_jma, make_links_lld, make_loki_plot_df
from loki_tools.plot_error_stats import (
	iqr_outliers,
	make_coherence_bins,
	make_mag_bins,
	pick_mag_column,
	plot_box,
	plot_hist,
)
from viz.events_map import plot_events_map_and_sections
from viz.loki.compare import plot_hist_overlay, save_scatter
from viz.plot_config import PlotConfig


def _finite_1d(x: np.ndarray) -> np.ndarray:
	a = np.asarray(x, dtype=float).ravel()
	return a[np.isfinite(a)]


def _stats_1d(x: np.ndarray) -> tuple[int, float, float, float]:
	v = _finite_1d(x)
	if v.size == 0:
		return (0, float('nan'), float('nan'), float('nan'))
	mean_v = float(np.mean(v))
	median_v = float(np.median(v))
	rmse_v = float(np.sqrt(np.mean(v * v)))
	return (int(v.size), mean_v, median_v, rmse_v)


def _d3_km_from_df(df: pd.DataFrame, dh_col: str, dz_col: str) -> np.ndarray:
	dh = np.asarray(df[dh_col].to_numpy(), dtype=float).ravel()
	dz = np.asarray(df[dz_col].to_numpy(), dtype=float).ravel()
	if dh.size != dz.size:
		raise ValueError('dh and dz length mismatch')
	mask = np.isfinite(dh) & np.isfinite(dz)
	dh2 = dh[mask]
	dz2 = dz[mask]
	return np.sqrt(dh2 * dh2 + dz2 * dz2)


def print_error_summary(
	dfs: list[pd.DataFrame],
	labels: list[str],
	*,
	dh_col: str = 'dh_km',
	dz_col: str = 'dz_km',
) -> None:
	if len(dfs) != len(labels):
		raise ValueError('len(dfs) must match len(labels)')
	print('=== Error summary (LOKI vs JMA) ===')
	for df, lbl in zip(dfs, labels, strict=True):
		n_dh, mean_dh, med_dh, rmse_dh = _stats_1d(df[dh_col].to_numpy())
		dz = np.asarray(df[dz_col].to_numpy(), dtype=float)
		dz_abs = np.abs(dz)
		n_dz_abs, mae_dz, med_abs_dz, rmse_abs_dz = _stats_1d(dz_abs)
		d3 = _d3_km_from_df(df, dh_col, dz_col)
		n_d3, mean_d3, med_d3, rmse_d3 = _stats_1d(d3)
		print(
			f'[{lbl}] '
			f'dh_km: n={n_dh}, mean={mean_dh:.3f}, median={med_dh:.3f}, '
			f'RMSE={rmse_dh:.3f} | '
			f'|dz|_km: n={n_dz_abs}, MAE={mae_dz:.3f}, median={med_abs_dz:.3f}, '
			f'RMSE={rmse_abs_dz:.3f} | '
			f'd3_km: n={n_d3}, mean={mean_d3:.3f}, median={med_d3:.3f}, '
			f'RMSE={rmse_d3:.3f}'
		)


def compare_error_hists_from_compare_csvs(
	compare_csvs: list[Path],
	*,
	labels: list[str] | None = None,
	out_dir: Path,
	bins_dh: int = 24,
	bins_dz: int = 25,
	bins_dt: int = 25,
	density: bool = False,
	print_stats: bool = True,
	min_cmax: float | None = None,
	max_cmax: float | None = None,
) -> None:
	if len(compare_csvs) == 0:
		raise ValueError('compare_csvs must be non-empty')

	csvs = [Path(p) for p in compare_csvs]
	for p in csvs:
		if not p.is_file():
			raise FileNotFoundError(f'compare_csv not found: {p}')

	if labels is None:
		lbls = [p.parent.name for p in csvs]
	else:
		lbls = list(labels)
		if len(lbls) != len(csvs):
			raise ValueError('len(labels) must match len(compare_csvs)')

	required = {'dh_km', 'dz_km', 'dt_origin_sec', 'cmax'}
	dfs: list[pd.DataFrame] = []
	for p in csvs:
		df = pd.read_csv(p)
		missing = required - set(df.columns)
		if missing:
			raise ValueError(f'compare_df missing columns in {p}: {sorted(missing)}')
		df = df.copy()
		df['dh_km'] = df['dh_km'].astype(float)
		df['dz_km'] = df['dz_km'].astype(float)
		df['dt_origin_sec'] = df['dt_origin_sec'].astype(float)
		df['cmax'] = df['cmax'].astype(float)
		if min_cmax is not None:
			df = df[df['cmax'] >= float(min_cmax)]
		if max_cmax is not None:
			df = df[df['cmax'] <= float(max_cmax)]
		dfs.append(df)

	if print_stats:
		print_error_summary(dfs, lbls)

	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	plot_hist_overlay(
		[d['dh_km'].to_numpy() for d in dfs],
		lbls,
		title='Horizontal error dh_km (LOKI vs JMA)',
		xlabel='dh_km [km]',
		out_png=out_dir / 'dh_km_hist_compare.png',
		bins=int(bins_dh),
		density=bool(density),
	)
	plot_hist_overlay(
		[d['dz_km'].to_numpy() for d in dfs],
		lbls,
		title='Depth error dz_km (LOKI - JMA)',
		xlabel='dz_km [km]',
		out_png=out_dir / 'dz_km_hist_compare.png',
		bins=int(bins_dz),
		density=bool(density),
	)
	plot_hist_overlay(
		[d['dt_origin_sec'].to_numpy() for d in dfs],
		lbls,
		title='Origin time error dt_origin_sec (LOKI - JMA)',
		xlabel='dt_origin_sec [sec]',
		out_png=out_dir / 'dt_origin_sec_hist_compare.png',
		bins=int(bins_dt),
		density=bool(density),
	)


def run_loki_vs_jma_qc(
	base_input_dir: Path,
	loki_output_dir: Path,
	header_path: Path,
	event_glob: str,
	plot_cfg: PlotConfig,
	*,
	use_build_compare_df: bool = True,
	compare_csv_out: Path | None = None,
	compare_csv_in: Path | None = None,
	out_png: Path | None = None,
	allowed_event_ids: set[str] | None = None,
) -> pd.DataFrame:
	compare_csv = (
		loki_output_dir / 'compare_jma_vs_loki.csv'
		if compare_csv_out is None
		else Path(compare_csv_out)
	)
	out_png = loki_output_dir / 'loki_vs_jma.png' if out_png is None else Path(out_png)

	if use_build_compare_df:
		compare_df = build_compare_df(
			base_input_dir=base_input_dir,
			loki_output_dir=loki_output_dir,
			header_path=header_path,
			event_glob=event_glob,
			allowed_event_ids=allowed_event_ids,
		)
	else:
		read_csv = compare_csv if compare_csv_in is None else Path(compare_csv_in)
		if not read_csv.is_file():
			raise FileNotFoundError(f'compare_csv not found: {read_csv}')
		compare_df = pd.read_csv(read_csv)

	compare_csv.parent.mkdir(parents=True, exist_ok=True)
	compare_df.to_csv(compare_csv, index=False)

	# ---- map/sections ----
	df_loki_plot = make_loki_plot_df(compare_df)
	extras_lld = make_extras_lld_jma(compare_df)
	links_lld = make_links_lld(compare_df)

	out_png.parent.mkdir(parents=True, exist_ok=True)
	plot_events_map_and_sections(
		df_loki_plot,
		out_png=out_png,
		mag_col='mag_jma',
		size_col='mag_jma',
		depth_col='depth_km',
		origin_time_col='origin_time',
		lat_col='latitude_deg',
		lon_col='longitude_deg',
		min_mag=2,
		lon_range=tuple(plot_cfg.lon_range),
		lat_range=tuple(plot_cfg.lat_range),
		depth_range=tuple(plot_cfg.depth_range),
		extras_lld=extras_lld,
		links_lld=links_lld,
	)

	# ---- error stats ----
	df = compare_df.copy()
	required = {
		'event_id',
		'origin_time_jma',
		'origin_time_loki',
		'dt_origin_sec',
		'dh_km',
		'dz_km',
		'e_w3d_km',
		'cmax',
	}
	missing = required - set(df.columns)
	if missing:
		raise ValueError(f'compare_df missing columns: {sorted(missing)}')

	df['origin_time_jma'] = pd.to_datetime(df['origin_time_jma'])
	df['origin_time_loki'] = pd.to_datetime(df['origin_time_loki'])
	df['dt_origin_sec'] = df['dt_origin_sec'].astype(float)
	df['dh_km'] = df['dh_km'].astype(float)
	df['dz_km'] = df['dz_km'].astype(float)
	df['e_w3d_km'] = df['e_w3d_km'].astype(float)
	df['cmax'] = df['cmax'].astype(float)

	error_out_dir = loki_output_dir / 'error_stats'
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
	plot_hist(
		df['dt_origin_sec'].to_numpy(),
		title='Origin time error dt_origin_sec (LOKI - JMA)',
		xlabel='dt_origin_sec [sec]',
		out_png=error_out_dir / 'dt_origin_sec_hist.png',
		bins=30,
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
	plot_box(
		[df['dt_origin_sec'].to_numpy()],
		labels=['all'],
		title='dt_origin_sec boxplot (all)',
		ylabel='dt_origin_sec [sec]',
		out_png=error_out_dir / 'dt_origin_sec_box_all.png',
	)
	df['coh_bin'] = make_coherence_bins(df, 'cmax', n_bins=4)
	labels = sorted(df['coh_bin'].unique())
	dh_by = [df.loc[df['coh_bin'] == b, 'dh_km'].to_numpy() for b in labels]
	dz_by = [df.loc[df['coh_bin'] == b, 'dz_km'].to_numpy() for b in labels]

	plot_box(
		dh_by,
		labels=labels,
		title='dh_km by coherence bins (cmax)',
		ylabel='dh_km [km]',
		out_png=error_out_dir / 'dh_km_box_by_cmax.png',
	)
	plot_box(
		dz_by,
		labels=labels,
		title='dz_km by coherence bins (cmax)',
		ylabel='dz_km [km]',
		out_png=error_out_dir / 'dz_km_box_by_cmax.png',
	)

	dh_out = iqr_outliers(df, 'dh_km')
	dz_out = iqr_outliers(df, 'dz_km')

	dh_top = df.sort_values('dh_km', ascending=False).head(10)
	dz_top_abs = (
		df.assign(dz_abs=np.abs(df['dz_km']))
		.sort_values('dz_abs', ascending=False)
		.head(10)
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
	dh_top.to_csv(error_out_dir / 'top10_dh_km.csv', index=False)
	dz_top_abs.to_csv(error_out_dir / 'top10_abs_dz_km.csv', index=False)

	mag_col = pick_mag_column(df)
	df['mag_jma'] = pd.to_numeric(df[mag_col], errors='raise').astype(float)

	df_mag = df.dropna(subset=['mag_jma']).copy()
	df_mag['mag_bin'], mag_labels = make_mag_bins(df_mag['mag_jma'])

	dh_by_mag = [
		df_mag.loc[df_mag['mag_bin'] == b, 'dh_km'].to_numpy() for b in mag_labels
	]
	dz_by_mag = [
		df_mag.loc[df_mag['mag_bin'] == b, 'dz_km'].to_numpy() for b in mag_labels
	]

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

	save_scatter(
		df_mag['mag_jma'].to_numpy(),
		df_mag['dh_km'].to_numpy(),
		title='dh_km vs JMA magnitude',
		xlabel='JMA magnitude',
		ylabel='dh_km [km]',
		out_png=error_out_dir / 'dh_km_vs_mag.png',
	)

	save_scatter(
		df_mag['mag_jma'].to_numpy(),
		df_mag['dz_km'].to_numpy(),
		title='dz_km vs JMA magnitude',
		xlabel='JMA magnitude',
		ylabel='dz_km [km]',
		out_png=error_out_dir / 'dz_km_vs_mag.png',
	)

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

	return compare_df
