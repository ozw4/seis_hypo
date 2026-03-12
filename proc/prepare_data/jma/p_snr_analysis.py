# %%

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from viz.core.qc_plot import binned_stats, hexbin_with_color_mode, require_file

# =========================
# Hard-coded paths & params
# =========================
SNR_CSV = Path('/workspace/data/waveform/jma/snr_pick_table.csv')
OUT_DIR = Path('/workspace/data/waveform/jma/snr_analysis_out')

MIN_DIST_KM = 5.0
MAX_DIST_KM = 300.0

MAG_BINS = 30
LOGDIST_BINS = 35

EPS = 1e-12

# =========================
# Choose SNR column (fb)
# =========================
# "snr_db" | "energy" | "rms" | "stalta"
SNR_FB = 'rms'

# =========================
# Site correction (computed every run)
# =========================
APPLY_SITE_CORRECTION = False
SITE_SHRINK_K = 30

# =========================
# Hexbin visualization mode
# =========================
# 'xnorm'      : normalize counts within each x-bin (p(y | x-bin) style)
# 'log_counts' : raw counts but shown as log10(counts) via hexbin(bins='log')
# 'counts'     : raw counts (linear)
HEX_MODE = 'xnorm'

HEX_CMAP_BASE_NAME = 'inferno'
HEX_CMAP_XNORM_REGISTERED_NAME = 'Reds_under_gray_le_001'
HEX_UNDER_THRESHOLD_XNORM = 0.01
HEX_UNDER_COLOR_XNORM = '#d9d9d9'

HEX_GRIDSIZE_MAG = 70
HEX_GRIDSIZE_LOGD = 80
HEX_MINCNT_MAG = 1
HEX_MINCNT_LOGD = 1

# xnormモード時：x-binの総数が少ない列はノイジーなので隠す（0なら無効）
XBIN_MIN_COLSUM = 5.0

# =========================
# Color scale control (vmin/vmax)
# =========================
# Noneなら自動スケール
# HEX_MODE='counts' 用（counts）
HEX_VMIN_COUNTS = None
HEX_VMAX_COUNTS = None

# HEX_MODE='log_counts' 用（log10(counts) の表示値）
HEX_VMIN_LOGCOUNTS = None
HEX_VMAX_LOGCOUNTS = None

# HEX_MODE='xnorm' 用（fraction within x-bin）
# <= 0.01 を under 色で描くため、vmin は 0.01 よりわずかに大きくする
HEX_VMIN_XNORM = float(np.nextafter(HEX_UNDER_THRESHOLD_XNORM, np.inf))
HEX_VMAX_XNORM = 0.06


def _snr_col_from_fb(snr_fb: str) -> str:
	s = str(snr_fb).strip().lower()
	if s in {'snr_db', 'default', 'primary'}:
		return 'snr_db'
	if s in {'energy', 'snr_db_energy'}:
		return 'snr_db_energy'
	if s in {'rms', 'snr_db_rms'}:
		return 'snr_db_rms'
	if s in {'stalta', 'snr_db_stalta'}:
		return 'snr_db_stalta'
	raise ValueError(f'unknown SNR_FB: {snr_fb} (use snr_db/energy/rms/stalta)')


def _compute_site_terms_median_shrinkage(
	df: pd.DataFrame, *, y_col: str, shrink_k: float
) -> pd.DataFrame:
	y = df[y_col].astype(float)
	sta = df['station'].astype(str)

	global_med = float(np.nanmedian(y.to_numpy(dtype=float)))

	tmp = pd.DataFrame({'station': sta, 'y': y})
	g = tmp.groupby('station', dropna=False)['y']
	sta_med = g.median()
	sta_n = g.size().astype(float)

	site_term = sta_med - global_med
	shrink = sta_n / (sta_n + float(shrink_k))
	site_term_shrunk = site_term * shrink

	out = pd.DataFrame(
		{
			'station': sta_med.index.astype(str),
			'n': sta_n.to_numpy(dtype=float),
			'station_median': sta_med.to_numpy(dtype=float),
			'global_median': float(global_med),
			'site_term_db': site_term.to_numpy(dtype=float),
			'shrink': shrink.to_numpy(dtype=float),
			'site_term_shrunk_db': site_term_shrunk.to_numpy(dtype=float),
		}
	).reset_index(drop=True)

	return out


def _attach_site_corrected_y(
	df: pd.DataFrame, *, y_col: str, shrink_k: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
	terms = _compute_site_terms_median_shrinkage(df, y_col=y_col, shrink_k=shrink_k)

	df2 = df.copy()
	df2['station'] = df2['station'].astype(str)

	m = df2.merge(
		terms[['station', 'site_term_shrunk_db', 'shrink']], on='station', how='left'
	)
	df2['site_term_shrunk_db'] = m['site_term_shrunk_db'].astype(float)
	df2['site_shrink'] = m['shrink'].astype(float)

	df2['snr_raw_db'] = df2[y_col].astype(float)
	df2['snr_used_db'] = df2['snr_raw_db'] - df2['site_term_shrunk_db']

	return df2, terms


def _clim_for_mode(mode: str) -> tuple[float | None, float | None]:
	mode2 = str(mode).strip().lower()
	if mode2 == 'counts':
		return HEX_VMIN_COUNTS, HEX_VMAX_COUNTS
	if mode2 == 'log_counts':
		return HEX_VMIN_LOGCOUNTS, HEX_VMAX_LOGCOUNTS
	if mode2 == 'xnorm':
		return HEX_VMIN_XNORM, HEX_VMAX_XNORM
	raise ValueError(f'unknown HEX_MODE: {mode}')


def _ensure_registered_xnorm_cmap() -> str:
	if HEX_CMAP_XNORM_REGISTERED_NAME not in plt.colormaps():
		cmap = plt.get_cmap(HEX_CMAP_BASE_NAME).copy()
		cmap.set_under(HEX_UNDER_COLOR_XNORM)
		mpl.colormaps.register(cmap, name=HEX_CMAP_XNORM_REGISTERED_NAME)
	return HEX_CMAP_XNORM_REGISTERED_NAME


def _hex_cmap_name_for_mode(mode: str) -> str:
	mode2 = str(mode).strip().lower()
	if mode2 == 'xnorm':
		return _ensure_registered_xnorm_cmap()
	if mode2 in {'counts', 'log_counts'}:
		return HEX_CMAP_BASE_NAME
	raise ValueError(f'unknown HEX_MODE: {mode}')


def make_mag_plot(
	df: pd.DataFrame, *, y_col: str, out_png: Path, out_bins_csv: Path, title: str
) -> None:
	x = df['mag1'].to_numpy(dtype=float)
	y = df[y_col].to_numpy(dtype=float)

	xmin = float(np.nanmin(x))
	xmax = float(np.nanmax(x))
	if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
		raise ValueError('mag1 range invalid for binning')

	bins = np.linspace(xmin, xmax, MAG_BINS + 1)
	stats = binned_stats(x, y, bins)
	stats.to_csv(out_bins_csv, index=False)

	fig = plt.figure(figsize=(9, 6))
	ax = fig.add_subplot(111)
	vmin, vmax = _clim_for_mode(HEX_MODE)
	cmap_name = _hex_cmap_name_for_mode(HEX_MODE)
	hexbin_with_color_mode(
		fig,
		ax,
		x=x,
		y=y,
		gridsize=HEX_GRIDSIZE_MAG,
		mincnt=HEX_MINCNT_MAG,
		cmap=cmap_name,
		mode=HEX_MODE,
		xedges_for_xnorm=bins,
		xbin_min_colsum=float(XBIN_MIN_COLSUM),
		vmin=vmin,
		vmax=vmax,
	)

	ok = stats['n'].to_numpy() > 0
	ax.plot(
		stats.loc[ok, 'bin_center'].to_numpy(),
		stats.loc[ok, 'median'].to_numpy(),
		marker='o',
		linewidth=2.0,
		label='binned median',
	)
	ax.fill_between(
		stats.loc[ok, 'bin_center'].to_numpy(),
		stats.loc[ok, 'p16'].to_numpy(),
		stats.loc[ok, 'p84'].to_numpy(),
		alpha=0.4,
		label='p16–p84',
	)

	ax.set_xlabel('mag1')
	ax.set_ylabel(f'{y_col} (dB)')
	ax.set_title(title)
	ax.grid(True)
	ax.legend()

	fig.tight_layout()
	fig.savefig(out_png, dpi=200)
	plt.close(fig)


def make_distance_plot(
	df: pd.DataFrame, *, y_col: str, out_png: Path, out_bins_csv: Path, title: str
) -> None:
	dist = df['distance_hypo_km'].to_numpy(dtype=float)
	y = df[y_col].to_numpy(dtype=float)

	logd = np.log10(np.maximum(dist, EPS))
	dmin = float(np.nanmin(logd))
	dmax = float(np.nanmax(logd))
	if not np.isfinite(dmin) or not np.isfinite(dmax) or dmin == dmax:
		raise ValueError('distance range invalid for binning')

	bins = np.linspace(dmin, dmax, LOGDIST_BINS + 1)
	stats = binned_stats(logd, y, bins)
	stats.to_csv(out_bins_csv, index=False)

	fig = plt.figure(figsize=(9, 6))
	ax = fig.add_subplot(111)
	vmin, vmax = _clim_for_mode(HEX_MODE)
	cmap_name = _hex_cmap_name_for_mode(HEX_MODE)
	hexbin_with_color_mode(
		fig,
		ax,
		x=logd,
		y=y,
		gridsize=HEX_GRIDSIZE_LOGD,
		mincnt=HEX_MINCNT_LOGD,
		cmap=cmap_name,
		mode=HEX_MODE,
		xedges_for_xnorm=bins,
		xbin_min_colsum=float(XBIN_MIN_COLSUM),
		vmin=vmin,
		vmax=vmax,
	)

	ok = stats['n'].to_numpy() > 0
	ax.plot(
		stats.loc[ok, 'bin_center'].to_numpy(),
		stats.loc[ok, 'median'].to_numpy(),
		marker='o',
		linewidth=2.0,
		label='binned median',
	)
	ax.fill_between(
		stats.loc[ok, 'bin_center'].to_numpy(),
		stats.loc[ok, 'p16'].to_numpy(),
		stats.loc[ok, 'p84'].to_numpy(),
		alpha=0.4,
		label='p16–p84',
	)

	ax.set_xlabel('log10(distance_hypo_km)')
	ax.set_ylabel(f'{y_col} (dB)')
	ax.set_title(title)

	tick_km = np.array([5, 10, 20, 50, 100, 200, 500], dtype=float)
	tick_km = tick_km[(tick_km >= np.nanmin(dist)) & (tick_km <= np.nanmax(dist))]
	if len(tick_km) > 0:
		ax.set_xticks(np.log10(tick_km))
		ax.set_xticklabels([str(int(v)) for v in tick_km])

	ax.grid(True)
	ax.legend()

	fig.tight_layout()
	fig.savefig(out_png, dpi=200)
	plt.close(fig)


def run_base_regression(
	df: pd.DataFrame, *, y_col: str, out_json: Path, model_name: str
) -> None:
	df = df.copy()
	df['log10_dist'] = np.log10(
		np.maximum(df['distance_hypo_km'].to_numpy(dtype=float), EPS)
	)
	df['log10_depth'] = np.log10(df['depth_km'].to_numpy(dtype=float) + 1.0)

	X = sm.add_constant(df[['mag1', 'log10_dist', 'log10_depth']].astype(float))
	y = df[y_col].astype(float)

	fit = sm.OLS(y, X).fit(cov_type='HC3')

	out = {
		'model': model_name,
		'y_col': y_col,
		'nobs': int(fit.nobs),
		'rsquared': float(fit.rsquared),
		'params': {k: float(v) for k, v in fit.params.items()},
		'bse_hc3': {k: float(v) for k, v in fit.bse.items()},
		'pvalues': {k: float(v) for k, v in fit.pvalues.items()},
		'filters': {'min_dist_km': MIN_DIST_KM, 'max_dist_km': MAX_DIST_KM},
	}
	out_json.write_text(json.dumps(out, indent=2), encoding='utf-8')


def main() -> None:
	require_file(
		SNR_CSV,
		hint='Edit SNR_CSV at the top of this script to the correct path.',
	)
	OUT_DIR.mkdir(parents=True, exist_ok=True)

	snr_col = _snr_col_from_fb(SNR_FB)

	usecols = [
		'event_id',
		'snr_db',
		'snr_db_energy',
		'snr_db_rms',
		'snr_db_stalta',
		'mag1',
		'distance_hypo_km',
		'depth_km',
		'station',
		'network_code',
	]
	df = pd.read_csv(SNR_CSV, usecols=usecols, low_memory=False)

	if snr_col not in df.columns:
		raise ValueError(
			f'Requested SNR column not found: {snr_col}\n'
			f'Available snr columns: {sorted([c for c in df.columns if c.startswith("snr")])}'
		)

	df = df.dropna(
		subset=[snr_col, 'mag1', 'distance_hypo_km', 'depth_km', 'station']
	).copy()
	df = df[np.isfinite(df[snr_col].to_numpy(dtype=float))].copy()
	df = df[np.isfinite(df['mag1'].to_numpy(dtype=float))].copy()
	df = df[np.isfinite(df['distance_hypo_km'].to_numpy(dtype=float))].copy()
	df = df[np.isfinite(df['depth_km'].to_numpy(dtype=float))].copy()

	df = df[
		(df['distance_hypo_km'] >= MIN_DIST_KM)
		& (df['distance_hypo_km'] <= MAX_DIST_KM)
	].copy()

	tag = f'{snr_col}'
	if APPLY_SITE_CORRECTION:
		df2, terms = _attach_site_corrected_y(
			df, y_col=snr_col, shrink_k=float(SITE_SHRINK_K)
		)
		y_col = 'snr_used_db'
		tag = f'{snr_col}_sitecorr_k{int(round(float(SITE_SHRINK_K)))}'
		terms_csv = OUT_DIR / f'station_site_terms_{tag}.csv'
		terms.to_csv(terms_csv, index=False)
	else:
		df2 = df.copy()
		df2['snr_raw_db'] = df2[snr_col].astype(float)
		df2['snr_used_db'] = df2['snr_raw_db']
		y_col = 'snr_used_db'
		terms_csv = None

	model_name = f'{y_col} ~ const + mag1 + log10(distance_hypo_km) + log10(depth_km+1)'
	run_base_regression(
		df2,
		y_col=y_col,
		out_json=OUT_DIR / f'snr_base_regression_{tag}.json',
		model_name=model_name,
	)

	make_mag_plot(
		df2,
		y_col=y_col,
		out_png=OUT_DIR / f'mag1_vs_{y_col}_{tag}.png',
		out_bins_csv=OUT_DIR / f'mag1_vs_{y_col}_{tag}_binned.csv',
		title=f'mag1 vs {y_col} ({tag}, hex={HEX_MODE})',
	)
	make_distance_plot(
		df2,
		y_col=y_col,
		out_png=OUT_DIR / f'distance_vs_{y_col}_{tag}.png',
		out_bins_csv=OUT_DIR / f'distance_vs_{y_col}_{tag}_binned.csv',
		title=f'distance vs {y_col} ({tag}, hex={HEX_MODE})',
	)

	cmap_used = _hex_cmap_name_for_mode(HEX_MODE)

	summary = {
		'snr_fb': SNR_FB,
		'snr_col_raw': snr_col,
		'y_col_used': y_col,
		'apply_site_correction': bool(APPLY_SITE_CORRECTION),
		'site_shrink_k': float(SITE_SHRINK_K),
		'hex_mode': HEX_MODE,
		'hex_cmap_base': HEX_CMAP_BASE_NAME,
		'hex_cmap_used': cmap_used,
		'hex_under_xnorm': {
			'threshold_le': HEX_UNDER_THRESHOLD_XNORM,
			'color': HEX_UNDER_COLOR_XNORM,
		},
		'hex_clim': {
			'counts': [HEX_VMIN_COUNTS, HEX_VMAX_COUNTS],
			'log_counts': [HEX_VMIN_LOGCOUNTS, HEX_VMAX_LOGCOUNTS],
			'xnorm': [HEX_VMIN_XNORM, HEX_VMAX_XNORM],
		},
		'xbin_min_colsum': float(XBIN_MIN_COLSUM),
		'n_rows_used': len(df2),
		'unique_events': int(df2['event_id'].nunique())
		if 'event_id' in df2.columns
		else None,
		'unique_stations': int(df2['station'].nunique()),
		'unique_networks': int(df2['network_code'].nunique()),
		'filters': {'min_dist_km': MIN_DIST_KM, 'max_dist_km': MAX_DIST_KM},
		'outputs': [
			str(OUT_DIR / f'snr_base_regression_{tag}.json'),
			str(OUT_DIR / f'mag1_vs_{y_col}_{tag}.png'),
			str(OUT_DIR / f'distance_vs_{y_col}_{tag}.png'),
			str(OUT_DIR / f'mag1_vs_{y_col}_{tag}_binned.csv'),
			str(OUT_DIR / f'distance_vs_{y_col}_{tag}_binned.csv'),
		],
	}
	if APPLY_SITE_CORRECTION and terms_csv is not None:
		summary['outputs'].append(str(terms_csv))

	(OUT_DIR / f'run_summary_{tag}.json').write_text(
		json.dumps(summary, indent=2), encoding='utf-8'
	)

	print(f'[ok] SNR_FB={SNR_FB} -> snr_col={snr_col}')
	print(f'[ok] APPLY_SITE_CORRECTION={APPLY_SITE_CORRECTION} k={SITE_SHRINK_K}')
	print(f'[ok] HEX_MODE={HEX_MODE} cmap={cmap_used}')
	print(f'[ok] wrote outputs under: {OUT_DIR}')


if __name__ == '__main__':
	main()

# %%
