# %%
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from viz.core.qc_plot import (
	binned_mean,
	binned_stats,
	hexbin_with_color_mode,
	require_any_files,
	require_file,
)

# =========================
# Hard-coded paths & params
# =========================
# You can point DT_INPUT to either:
#  - a single dt_table.csv file, or
#  - a directory that contains one or more dt_table.csv files under it.
DT_INPUT = Path(
	'/workspace/proc/prepare_data/jma/runs/dt_stalta_p_u_YYYYMMDD_01/dt_table.csv'
)
OUT_DIR = Path('/workspace/proc/prepare_data/jma/runs/dt_pick_error_analysis_out')


# =========================
# Optional filters
# =========================
RUN_ID_ALLOWLIST: list[str] | None = None
PHASE_ALLOWLIST: list[str] | None = None
COMPONENT_ALLOWLIST: list[str] | None = None


# =========================
# Plot ranges
# =========================
MIN_DIST_KM = 5.0
MAX_DIST_KM = 250.0

MAG_BINS = 80
LOGDIST_BINS = 85

EPS = 1e-12


# =========================
# Distance x-axis mode
# =========================
# 'log10' | 'linear'
DIST_XMODE = 'log10'
DIST_BINS_LINEAR = 80


# =========================
# Hexbin visualization mode
# =========================
# 'xnorm'      : normalize counts within each x-bin (p(y | x-bin) style)
# 'log_counts' : raw counts but shown as log10(counts) via hexbin(bins='log')
# 'counts'     : raw counts (linear)
HEX_MODE = 'log_counts'

HEX_CMAP = 'viridis'
HEX_GRIDSIZE_MAG = 70
HEX_GRIDSIZE_DIST = 75
HEX_MINCNT_MAG = 1
HEX_MINCNT_DIST = 1

# xnormモード時：x-binの総数が少ない列はノイジーなので隠す（0なら無効）
XBIN_MIN_COLSUM = 5.0


# =========================
# Color scale control (vmin/vmax)
# =========================
# Noneなら自動スケール
HEX_VMIN_COUNTS: float | None = None
HEX_VMAX_COUNTS: float | None = None

HEX_VMIN_LOGCOUNTS: float | None = None
HEX_VMAX_LOGCOUNTS: float | None = 3000

HEX_VMIN_XNORM = 0.0
HEX_VMAX_XNORM = 0.10


# =========================
# Outlier trimming for plots
# =========================
# We keep outliers for summary metrics, but trim them for hexbin plots
# to avoid making everything unreadable.
PLOT_TRIM_ABS_DT_Q = 0.99


# =========================
# Y-axis limits (view only)
# =========================
# Noneなら自動スケール
YLIM_ABS_DT: tuple[float, float] | None = (0.0, 1.0)  # |dt|表示用
YLIM_DT: tuple[float, float] | None = (-1.0, 1.0)

TOPK_BAR = 30


def _clim_for_mode(mode: str) -> tuple[float | None, float | None]:
	mode2 = str(mode).strip().lower()
	if mode2 == 'counts':
		return HEX_VMIN_COUNTS, HEX_VMAX_COUNTS
	if mode2 == 'log_counts':
		return HEX_VMIN_LOGCOUNTS, HEX_VMAX_LOGCOUNTS
	if mode2 == 'xnorm':
		return HEX_VMIN_XNORM, HEX_VMAX_XNORM
	raise ValueError(f'unknown HEX_MODE: {mode}')


def _to_bool(s: pd.Series) -> pd.Series:
	if s.dtype == bool:
		return s
	if np.issubdtype(s.dtype, np.number):
		return s.astype(float).fillna(0.0) != 0.0
	v = s.astype(str).str.strip().str.lower()
	return v.isin(['1', 'true', 't', 'yes', 'y'])


def _read_dt_tables(dt_input: Path) -> pd.DataFrame:
	dt_input = Path(dt_input)
	if dt_input.is_dir():
		paths = sorted(dt_input.glob('**/dt_table.csv'))
		require_any_files(
			paths,
			hint=(
				f'Edit DT_INPUT at the top of this script.\n'
				f'Current DT_INPUT is a directory with no dt_table.csv under it: {dt_input}'
			),
		)
		usecols = _usecols_for_analysis()
		dfs = [pd.read_csv(p, usecols=usecols, low_memory=False) for p in paths]
		out = pd.concat(dfs, ignore_index=True)
		out['_source_csv'] = np.repeat([p.name for p in paths], [len(x) for x in dfs])
		return out

	require_file(
		dt_input,
		hint=(
			'Edit DT_INPUT at the top of this script to the correct dt_table.csv path.'
		),
	)
	usecols = _usecols_for_analysis()
	out = pd.read_csv(dt_input, usecols=usecols, low_memory=False)
	out['_source_csv'] = dt_input.name
	return out


def _usecols_for_analysis() -> list[str]:
	cols = [
		'run_id',
		'phase',
		'component',
		'event_id',
		'station',
		'mag',
		'distance_hypo_km',
		't0_iso',
		'found_peak',
		'dt_sec',
		'good_0p05',
		'good_0p10',
		'good_0p20',
		'fail_reason',
	]
	return cols


def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
	if RUN_ID_ALLOWLIST is not None:
		allow = {str(x) for x in RUN_ID_ALLOWLIST}
		df = df[df['run_id'].astype(str).isin(sorted(allow))].copy()
	if PHASE_ALLOWLIST is not None:
		allow = {str(x) for x in PHASE_ALLOWLIST}
		df = df[df['phase'].astype(str).isin(sorted(allow))].copy()
	if COMPONENT_ALLOWLIST is not None:
		allow = {str(x) for x in COMPONENT_ALLOWLIST}
		df = df[df['component'].astype(str).isin(sorted(allow))].copy()
	return df


def _prep_numeric(df: pd.DataFrame) -> pd.DataFrame:
	df2 = df.copy()
	df2['mag'] = df2['mag'].astype(float)
	df2['distance_hypo_km'] = df2['distance_hypo_km'].astype(float)
	df2['dt_sec'] = df2['dt_sec'].astype(float)
	df2['found_peak'] = _to_bool(df2['found_peak'])
	for c in ['good_0p05', 'good_0p10', 'good_0p20']:
		if c in df2.columns:
			df2[c] = _to_bool(df2[c]).astype(int)
	df2['abs_dt_sec'] = np.abs(df2['dt_sec'].to_numpy(dtype=float))
	return df2


def _distance_x_and_bins(
	dist_km: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str, np.ndarray | None, list[str] | None]:
	dist_km = np.asarray(dist_km, dtype=float)
	mode = str(DIST_XMODE).strip().lower()

	if mode == 'log10':
		x = np.log10(np.maximum(dist_km, EPS))
		xmin = float(np.nanmin(x))
		xmax = float(np.nanmax(x))
		if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
			raise ValueError('distance(log10) range invalid for binning')
		bins = np.linspace(xmin, xmax, LOGDIST_BINS + 1)
		xlabel = 'log10(distance_hypo_km)'

		tick_km = np.array([5, 10, 20, 50, 100, 200, 500], dtype=float)
		tick_km = tick_km[
			(tick_km >= np.nanmin(dist_km)) & (tick_km <= np.nanmax(dist_km))
		]
		if tick_km.size:
			xticks = np.log10(tick_km)
			xticklabels = [str(int(v)) for v in tick_km]
			return x, bins, xlabel, xticks, xticklabels
		return x, bins, xlabel, None, None

	if mode == 'linear':
		x = dist_km
		xmin = float(np.nanmin(x))
		xmax = float(np.nanmax(x))
		if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
			raise ValueError('distance(linear) range invalid for binning')
		bins = np.linspace(xmin, xmax, int(DIST_BINS_LINEAR) + 1)
		xlabel = 'distance_hypo_km (km)'

		tick_km = np.array([5, 10, 20, 50, 100, 200, 300], dtype=float)
		tick_km = tick_km[(tick_km >= xmin) & (tick_km <= xmax)]
		if tick_km.size:
			xticks = tick_km
			xticklabels = [str(int(v)) for v in tick_km]
			return x, bins, xlabel, xticks, xticklabels
		return x, bins, xlabel, None, None

	raise ValueError(f'unknown DIST_XMODE: {DIST_XMODE} (use log10/linear)')


def _global_summary(df: pd.DataFrame) -> dict[str, object]:
	base: dict[str, object] = {
		'n_rows_total': len(df),
		'unique_runs': int(df['run_id'].nunique()) if 'run_id' in df.columns else None,
		'unique_events': int(df['event_id'].nunique())
		if 'event_id' in df.columns
		else None,
		'unique_stations': int(df['station'].nunique())
		if 'station' in df.columns
		else None,
	}

	found = df['found_peak'].to_numpy(dtype=bool)
	base['found_rate'] = float(np.mean(found)) if found.size else float('nan')

	df_ok = df[found].copy()
	if len(df_ok) == 0:
		base['dt_metrics'] = None
		return base

	abs_dt = df_ok['abs_dt_sec'].to_numpy(dtype=float)
	dt = df_ok['dt_sec'].to_numpy(dtype=float)

	dt_metrics: dict[str, object] = {
		'median_dt_sec': float(np.nanmedian(dt)),
		'median_abs_dt_sec': float(np.nanmedian(abs_dt)),
		'p90_abs_dt_sec': float(np.nanquantile(abs_dt, 0.90)),
		'p99_abs_dt_sec': float(np.nanquantile(abs_dt, 0.99)),
		'p999_abs_dt_sec': float(np.nanquantile(abs_dt, 0.999)),
	}
	for c in ['good_0p05', 'good_0p10', 'good_0p20']:
		if c in df_ok.columns:
			dt_metrics[f'{c}_rate'] = float(np.mean(df_ok[c].to_numpy(dtype=float)))

	base['dt_metrics'] = dt_metrics
	return base


def _make_hist_and_cdf(df_ok: pd.DataFrame, out_dir: Path) -> dict[str, str]:
	abs_dt = df_ok['abs_dt_sec'].to_numpy(dtype=float)
	abs_dt = abs_dt[np.isfinite(abs_dt)]
	abs_dt = abs_dt[abs_dt >= 0]
	if abs_dt.size == 0:
		raise ValueError('no finite abs_dt_sec values after filtering')

	fig = plt.figure(figsize=(9, 5))
	ax = fig.add_subplot(111)
	ax.hist(abs_dt, bins=200)
	ax.set_xlabel('|dt_sec| (s)')
	ax.set_ylabel('count')
	ax.set_title('Pick error |dt| histogram')
	ax.grid(True)
	fig.tight_layout()
	out_hist = out_dir / 'abs_dt_hist.png'
	fig.savefig(out_hist, dpi=200)
	plt.close(fig)

	s = np.sort(abs_dt)
	y = np.linspace(0.0, 1.0, num=s.size, endpoint=True)
	fig = plt.figure(figsize=(9, 5))
	ax = fig.add_subplot(111)
	ax.plot(s, y, linewidth=2.0)
	ax.set_xlabel('|dt_sec| (s)')
	ax.set_ylabel('CDF')
	ax.set_title('Pick error |dt| CDF')
	ax.grid(True)
	fig.tight_layout()
	out_cdf = out_dir / 'abs_dt_cdf.png'
	fig.savefig(out_cdf, dpi=200)
	plt.close(fig)

	return {'abs_dt_hist': str(out_hist), 'abs_dt_cdf': str(out_cdf)}


def _make_mag_plot(
	df_ok: pd.DataFrame,
	*,
	y_col: str,
	out_png: Path,
	out_bins_csv: Path,
	title: str,
	y_label: str,
	y_clip: float | None,
	clip_by_abs: bool,
	ylim: tuple[float, float] | None,
) -> None:
	x = df_ok['mag'].to_numpy(dtype=float)
	y = df_ok[y_col].to_numpy(dtype=float)

	ok = np.isfinite(x) & np.isfinite(y)
	x = x[ok]
	y = y[ok]
	if y_clip is not None:
		m = np.abs(y) <= float(y_clip) if bool(clip_by_abs) else (y <= float(y_clip))
		x = x[m]
		y = y[m]

	xmin = float(np.nanmin(x))
	xmax = float(np.nanmax(x))
	if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
		raise ValueError('mag range invalid for binning')

	bins = np.linspace(xmin, xmax, MAG_BINS + 1)
	stats = binned_stats(x, y, bins)
	stats.to_csv(out_bins_csv, index=False)

	vmin, vmax = _clim_for_mode(HEX_MODE)
	fig = plt.figure(figsize=(9, 6))
	ax = fig.add_subplot(111)

	hexbin_with_color_mode(
		fig,
		ax,
		x=x,
		y=y,
		gridsize=HEX_GRIDSIZE_MAG,
		mincnt=HEX_MINCNT_MAG,
		cmap=HEX_CMAP,
		mode=HEX_MODE,
		xedges_for_xnorm=bins,
		xbin_min_colsum=float(XBIN_MIN_COLSUM),
		vmin=vmin,
		vmax=vmax,
	)

	ok2 = stats['n'].to_numpy(dtype=float) > 0
	ax.plot(
		stats.loc[ok2, 'bin_center'].to_numpy(),
		stats.loc[ok2, 'median'].to_numpy(),
		marker='o',
		linewidth=2.0,
		label='binned median',
	)
	ax.fill_between(
		stats.loc[ok2, 'bin_center'].to_numpy(),
		stats.loc[ok2, 'p16'].to_numpy(),
		stats.loc[ok2, 'p84'].to_numpy(),
		alpha=0.4,
		label='p16–p84',
	)

	ax.set_xlabel('mag')
	ax.set_ylabel(y_label)
	ax.set_title(title)
	if ylim is not None:
		ax.set_ylim(float(ylim[0]), float(ylim[1]))
	ax.grid(True)
	ax.legend()

	fig.tight_layout()
	fig.savefig(out_png, dpi=200)
	plt.close(fig)


def _make_distance_plot(
	df_ok: pd.DataFrame,
	*,
	y_col: str,
	out_png: Path,
	out_bins_csv: Path,
	title: str,
	y_label: str,
	y_clip: float | None,
	clip_by_abs: bool,
	ylim: tuple[float, float] | None,
) -> None:
	dist = df_ok['distance_hypo_km'].to_numpy(dtype=float)
	y = df_ok[y_col].to_numpy(dtype=float)

	ok = np.isfinite(dist) & np.isfinite(y)
	dist = dist[ok]
	y = y[ok]
	if y_clip is not None:
		m = np.abs(y) <= float(y_clip) if bool(clip_by_abs) else (y <= float(y_clip))
		dist = dist[m]
		y = y[m]

	x, bins, xlabel, xticks, xticklabels = _distance_x_and_bins(dist)

	stats = binned_stats(x, y, bins)
	stats.to_csv(out_bins_csv, index=False)

	vmin, vmax = _clim_for_mode(HEX_MODE)
	fig = plt.figure(figsize=(9, 6))
	ax = fig.add_subplot(111)

	hexbin_with_color_mode(
		fig,
		ax,
		x=x,
		y=y,
		gridsize=HEX_GRIDSIZE_DIST,
		mincnt=HEX_MINCNT_DIST,
		cmap=HEX_CMAP,
		mode=HEX_MODE,
		xedges_for_xnorm=bins,
		xbin_min_colsum=float(XBIN_MIN_COLSUM),
		vmin=vmin,
		vmax=vmax,
	)

	ok2 = stats['n'].to_numpy(dtype=float) > 0
	ax.plot(
		stats.loc[ok2, 'bin_center'].to_numpy(),
		stats.loc[ok2, 'median'].to_numpy(),
		marker='o',
		linewidth=2.0,
		label='binned median',
	)
	ax.fill_between(
		stats.loc[ok2, 'bin_center'].to_numpy(),
		stats.loc[ok2, 'p16'].to_numpy(),
		stats.loc[ok2, 'p84'].to_numpy(),
		alpha=0.4,
		label='p16–p84',
	)

	ax.set_xlabel(xlabel)
	ax.set_ylabel(y_label)
	ax.set_title(title)
	if xticks is not None and xticklabels is not None:
		ax.set_xticks(xticks)
		ax.set_xticklabels(xticklabels)
	if ylim is not None:
		ax.set_ylim(float(ylim[0]), float(ylim[1]))

	ax.grid(True)
	ax.legend()

	fig.tight_layout()
	fig.savefig(out_png, dpi=200)
	plt.close(fig)


def _make_goodrate_plot(
	df_ok: pd.DataFrame,
	*,
	x_kind: str,  # 'mag' | 'distance'
	out_png: Path,
	title: str,
) -> None:
	kind = str(x_kind).strip().lower()
	if kind == 'mag':
		x = df_ok['mag'].to_numpy(dtype=float)
		xmin = float(np.nanmin(x))
		xmax = float(np.nanmax(x))
		if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
			raise ValueError('mag range invalid for binning')
		bins = np.linspace(xmin, xmax, MAG_BINS + 1)
		xlabel = 'mag'
		xticks = None
		xticklabels = None
	elif kind == 'distance':
		dist = df_ok['distance_hypo_km'].to_numpy(dtype=float)
		x, bins, xlabel, xticks, xticklabels = _distance_x_and_bins(dist)
	else:
		raise ValueError("x_kind must be 'mag' or 'distance'")

	fig = plt.figure(figsize=(9, 5))
	ax = fig.add_subplot(111)

	for c in ['good_0p05', 'good_0p10', 'good_0p20']:
		if c not in df_ok.columns:
			continue
		stats = binned_mean(x, df_ok[c].to_numpy(dtype=float), bins)
		ok = stats['n'].to_numpy(dtype=float) > 0
		ax.plot(
			stats.loc[ok, 'bin_center'].to_numpy(),
			stats.loc[ok, 'mean'].to_numpy(),
			marker='o',
			linewidth=2.0,
			label=c,
		)

	ax.set_xlabel(xlabel)
	ax.set_ylabel('rate')
	ax.set_ylim(0.0, 1.0)
	ax.set_title(title)
	ax.grid(True)
	ax.legend()

	if xticks is not None and xticklabels is not None:
		ax.set_xticks(xticks)
		ax.set_xticklabels(xticklabels)

	fig.tight_layout()
	fig.savefig(out_png, dpi=200)
	plt.close(fig)


def _group_summary(
	df_ok: pd.DataFrame,
	*,
	key: str,
) -> pd.DataFrame:
	if key not in df_ok.columns:
		raise ValueError(f'missing key column: {key}')

	g = df_ok.groupby(key, dropna=False)
	p90 = g['abs_dt_sec'].quantile(0.90)
	out = pd.DataFrame(
		{
			key: g.size().index,
			'n': g.size().to_numpy(dtype=int),
			'median_dt_sec': g['dt_sec'].median().to_numpy(dtype=float),
			'median_abs_dt_sec': g['abs_dt_sec'].median().to_numpy(dtype=float),
			'p90_abs_dt_sec': p90.to_numpy(dtype=float),
		}
	)
	for c in ['good_0p05', 'good_0p10', 'good_0p20']:
		if c in df_ok.columns:
			out[f'{c}_rate'] = g[c].mean().to_numpy(dtype=float)
	out = out.sort_values('n', ascending=False).reset_index(drop=True)
	return out


def _plot_top_barh(
	sum_df: pd.DataFrame,
	*,
	key: str,
	metric: str,
	out_png: Path,
	title: str,
	n_top: int,
	ascending: bool,
) -> None:
	if metric not in sum_df.columns:
		raise ValueError(f'missing metric column: {metric}')

	df = sum_df.sort_values(metric, ascending=bool(ascending)).head(int(n_top)).copy()
	df = df.iloc[::-1].reset_index(drop=True)
	labels = df[key].astype(str).to_numpy()
	vals = df[metric].to_numpy(dtype=float)

	fig = plt.figure(figsize=(10, max(4, 0.28 * len(df) + 1)))
	ax = fig.add_subplot(111)
	ax.barh(labels, vals)
	ax.set_xlabel(metric)
	ax.set_title(title)
	ax.grid(True)
	fig.tight_layout()
	fig.savefig(out_png, dpi=200)
	plt.close(fig)


def _make_daily_drift_plot(df_ok: pd.DataFrame, out_png: Path) -> None:
	if 't0_iso' not in df_ok.columns:
		return

	ts = pd.to_datetime(df_ok['t0_iso'].astype(str), format='ISO8601', errors='raise')
	df = df_ok.copy()
	df['_date'] = ts.dt.date
	g = df.groupby('_date', dropna=False)
	daily = pd.DataFrame(
		{
			'date': g.size().index,
			'n': g.size().to_numpy(dtype=int),
			'median_dt_sec': g['dt_sec'].median().to_numpy(dtype=float),
			'median_abs_dt_sec': g['abs_dt_sec'].median().to_numpy(dtype=float),
		}
	)

	daily = daily.sort_values('date').reset_index(drop=True)
	x = np.arange(len(daily), dtype=float)
	fig = plt.figure(figsize=(11, 5))
	ax = fig.add_subplot(111)
	ax.plot(x, daily['median_dt_sec'].to_numpy(dtype=float), marker='o', linewidth=2.0)
	ax.set_xticks(x)
	ax.set_xticklabels(
		[str(d) for d in daily['date'].to_list()], rotation=45, ha='right'
	)
	ax.set_ylabel('median dt_sec (s)')
	ax.set_title('Daily median pick error drift (signed)')
	ax.grid(True)
	fig.tight_layout()
	fig.savefig(out_png, dpi=200)
	plt.close(fig)


def main() -> None:
	OUT_DIR.mkdir(parents=True, exist_ok=True)

	df0 = _read_dt_tables(DT_INPUT)
	df0 = _apply_filters(df0)

	req = {
		'run_id',
		'event_id',
		'station',
		'mag',
		'distance_hypo_km',
		'found_peak',
		'dt_sec',
	}
	miss = sorted(req - set(df0.columns))
	if miss:
		raise ValueError(f'dt_table missing required columns: {miss}')

	df = _prep_numeric(df0)

	summary: dict[str, object] = {
		'dt_input': str(DT_INPUT),
		'out_dir': str(OUT_DIR),
		'filters': {
			'run_id_allowlist': RUN_ID_ALLOWLIST,
			'phase_allowlist': PHASE_ALLOWLIST,
			'component_allowlist': COMPONENT_ALLOWLIST,
			'min_dist_km': float(MIN_DIST_KM),
			'max_dist_km': float(MAX_DIST_KM),
		},
		'distance_xmode': DIST_XMODE,
		'hex': {
			'mode': HEX_MODE,
			'cmap': HEX_CMAP,
			'xbin_min_colsum': float(XBIN_MIN_COLSUM),
			'clim': {
				'counts': [HEX_VMIN_COUNTS, HEX_VMAX_COUNTS],
				'log_counts': [HEX_VMIN_LOGCOUNTS, HEX_VMAX_LOGCOUNTS],
				'xnorm': [HEX_VMIN_XNORM, HEX_VMAX_XNORM],
			},
		},
	}
	summary['global'] = _global_summary(df)

	df_ok = df[df['found_peak']].copy()
	df_ok = df_ok.dropna(subset=['mag', 'distance_hypo_km', 'dt_sec']).copy()
	df_ok = df_ok[np.isfinite(df_ok['mag'].to_numpy(dtype=float))].copy()
	df_ok = df_ok[np.isfinite(df_ok['distance_hypo_km'].to_numpy(dtype=float))].copy()
	df_ok = df_ok[np.isfinite(df_ok['dt_sec'].to_numpy(dtype=float))].copy()

	df_ok = df_ok[
		(df_ok['distance_hypo_km'] >= float(MIN_DIST_KM))
		& (df_ok['distance_hypo_km'] <= float(MAX_DIST_KM))
	].copy()

	if len(df_ok) == 0:
		raise ValueError(
			'no rows left after filtering to found_peak and distance range'
		)

	abs_dt = df_ok['abs_dt_sec'].to_numpy(dtype=float)
	abs_dt = abs_dt[np.isfinite(abs_dt)]
	y_clip = float(np.quantile(abs_dt, float(PLOT_TRIM_ABS_DT_Q)))
	if not np.isfinite(y_clip) or y_clip <= 0:
		y_clip = None
	summary['plot_trim'] = {
		'abs_dt_q': float(PLOT_TRIM_ABS_DT_Q),
		'abs_dt_clip_sec': y_clip,
	}

	outputs: dict[str, str] = {}
	outputs.update(_make_hist_and_cdf(df_ok, OUT_DIR))

	_make_mag_plot(
		df_ok,
		y_col='abs_dt_sec',
		out_png=OUT_DIR / 'mag_vs_abs_dt.png',
		out_bins_csv=OUT_DIR / 'mag_vs_abs_dt_binned.csv',
		title=f'mag vs |dt| (hex={HEX_MODE}, dist=[{MIN_DIST_KM},{MAX_DIST_KM}] km)',
		y_label='|dt_sec| (s)',
		y_clip=y_clip,
		clip_by_abs=False,
		ylim=YLIM_ABS_DT,
	)
	outputs['mag_vs_abs_dt'] = str(OUT_DIR / 'mag_vs_abs_dt.png')
	outputs['mag_vs_abs_dt_binned'] = str(OUT_DIR / 'mag_vs_abs_dt_binned.csv')

	_make_distance_plot(
		df_ok,
		y_col='abs_dt_sec',
		out_png=OUT_DIR / 'dist_vs_abs_dt.png',
		out_bins_csv=OUT_DIR / 'dist_vs_abs_dt_binned.csv',
		title=f'distance vs |dt| (hex={HEX_MODE}, x={DIST_XMODE}, dist=[{MIN_DIST_KM},{MAX_DIST_KM}] km)',
		y_label='|dt_sec| (s)',
		y_clip=y_clip,
		clip_by_abs=False,
		ylim=YLIM_ABS_DT,
	)
	outputs['dist_vs_abs_dt'] = str(OUT_DIR / 'dist_vs_abs_dt.png')
	outputs['dist_vs_abs_dt_binned'] = str(OUT_DIR / 'dist_vs_abs_dt_binned.csv')

	_make_mag_plot(
		df_ok,
		y_col='dt_sec',
		out_png=OUT_DIR / 'mag_vs_dt.png',
		out_bins_csv=OUT_DIR / 'mag_vs_dt_binned.csv',
		title=f'mag vs dt (signed) (hex={HEX_MODE}, dist=[{MIN_DIST_KM},{MAX_DIST_KM}] km)',
		y_label='dt_sec (s)',
		y_clip=y_clip,
		clip_by_abs=True,
		ylim=YLIM_DT,
	)
	outputs['mag_vs_dt'] = str(OUT_DIR / 'mag_vs_dt.png')
	outputs['mag_vs_dt_binned'] = str(OUT_DIR / 'mag_vs_dt_binned.csv')

	_make_distance_plot(
		df_ok,
		y_col='dt_sec',
		out_png=OUT_DIR / 'dist_vs_dt.png',
		out_bins_csv=OUT_DIR / 'dist_vs_dt_binned.csv',
		title=f'distance vs dt (signed) (hex={HEX_MODE}, x={DIST_XMODE}, dist=[{MIN_DIST_KM},{MAX_DIST_KM}] km)',
		y_label='dt_sec (s)',
		y_clip=y_clip,
		clip_by_abs=True,
		ylim=YLрఐM_DT if False else YLIM_DT,
	)
	outputs['dist_vs_dt'] = str(OUT_DIR / 'dist_vs_dt.png')
	outputs['dist_vs_dt_binned'] = str(OUT_DIR / 'dist_vs_dt_binned.csv')

	_make_goodrate_plot(
		df_ok,
		x_kind='mag',
		out_png=OUT_DIR / 'mag_vs_goodrate.png',
		title=f'mag vs good-rate (dist=[{MIN_DIST_KM},{MAX_DIST_KM}] km)',
	)
	outputs['mag_vs_goodrate'] = str(OUT_DIR / 'mag_vs_goodrate.png')

	_make_goodrate_plot(
		df_ok,
		x_kind='distance',
		out_png=OUT_DIR / 'dist_vs_goodrate.png',
		title=f'distance vs good-rate (x={DIST_XMODE}, dist=[{MIN_DIST_KM},{MAX_DIST_KM}] km)',
	)
	outputs['dist_vs_goodrate'] = str(OUT_DIR / 'dist_vs_goodrate.png')

	sta_sum = _group_summary(df_ok, key='station')
	sta_csv = OUT_DIR / 'station_summary.csv'
	sta_sum.to_csv(sta_csv, index=False)
	outputs['station_summary_csv'] = str(sta_csv)

	ev_sum = _group_summary(df_ok, key='event_id')
	ev_csv = OUT_DIR / 'event_summary.csv'
	ev_sum.to_csv(ev_csv, index=False)
	outputs['event_summary_csv'] = str(ev_csv)

	run_sum = _group_summary(df_ok, key='run_id')
	run_csv = OUT_DIR / 'run_summary.csv'
	run_sum.to_csv(run_csv, index=False)
	outputs['run_summary_csv'] = str(run_csv)

	_plot_top_barh(
		sta_sum,
		key='station',
		metric='p90_abs_dt_sec',
		out_png=OUT_DIR / 'worst_stations_by_p90_abs_dt.png',
		title=f'Worst stations by p90(|dt|) (top {TOPK_BAR})',
		n_top=TOPK_BAR,
		ascending=False,
	)
	outputs['worst_stations_by_p90_abs_dt'] = str(
		OUT_DIR / 'worst_stations_by_p90_abs_dt.png'
	)

	_plot_top_barh(
		sta_sum,
		key='station',
		metric='median_dt_sec',
		out_png=OUT_DIR / 'station_bias_by_median_dt.png',
		title=f'Station bias by median(dt) (top {TOPK_BAR} most positive)',
		n_top=TOPK_BAR,
		ascending=False,
	)
	outputs['station_bias_by_median_dt'] = str(
		OUT_DIR / 'station_bias_by_median_dt.png'
	)

	_plot_top_barh(
		ev_sum,
		key='event_id',
		metric='p90_abs_dt_sec',
		out_png=OUT_DIR / 'worst_events_by_p90_abs_dt.png',
		title=f'Worst events by p90(|dt|) (top {TOPK_BAR})',
		n_top=TOPK_BAR,
		ascending=False,
	)
	outputs['worst_events_by_p90_abs_dt'] = str(
		OUT_DIR / 'worst_events_by_p90_abs_dt.png'
	)

	_make_daily_drift_plot(df_ok, OUT_DIR / 'daily_median_dt_drift.png')
	outputs['daily_median_dt_drift'] = str(OUT_DIR / 'daily_median_dt_drift.png')

	summary['generated_at'] = datetime.utcnow().isoformat() + 'Z'
	summary['outputs'] = outputs
	(OUT_DIR / 'run_summary.json').write_text(
		json.dumps(summary, indent=2), encoding='utf-8'
	)

	print(f'[ok] wrote outputs under: {OUT_DIR}')
	print(f'[ok] rows_total={len(df)} rows_used_for_plots={len(df_ok)}')


if __name__ == '__main__':
	main()

# %%
