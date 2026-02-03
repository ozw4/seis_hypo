# src/viz/jma/arrivetime.py
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from viz.core.fig_io import save_current_figure


def _savefig(path: Path) -> None:
	save_current_figure(path, dpi=160, tight_layout=True, close=True)


def _plot_hist_logy(
	values: pd.Series, *, bins: int, title: str, xlabel: str, outpath: Path
) -> None:
	plt.figure(figsize=(8, 4.5))
	plt.hist(values.dropna(), bins=bins)
	plt.yscale('log')
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel('count')
	_savefig(outpath)


def _plot_hist2d(
	x: pd.Series,
	y: pd.Series,
	*,
	bins: int,
	title: str,
	xlabel: str,
	ylabel: str,
	outpath: Path,
) -> None:
	plt.figure(figsize=(7, 5.5))
	plt.hist2d(x.to_numpy(), y.to_numpy(), bins=bins)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	_savefig(outpath)


def plot_jma_arrivetime_qc(
	*,
	event_summary: pd.DataFrame,
	station_summary: pd.DataFrame,
	station_sp: pd.DataFrame,
	picks: pd.DataFrame,
	fig_dir: str | Path,
	clip_dist_km: float = 200.0,
) -> None:
	fig_dir = Path(fig_dir)
	fig_dir.mkdir(parents=True, exist_ok=True)

	_plot_hist_logy(
		event_summary['n_stations_meas'],
		bins=60,
		title='Stations per event (log-y)',
		xlabel='n_stations_meas',
		outpath=fig_dir / 'hist_n_stations.png',
	)
	_plot_hist_logy(
		event_summary['total_picks'],
		bins=70,
		title='Total picks per event (log-y)',
		xlabel='total_picks',
		outpath=fig_dir / 'hist_total_picks.png',
	)

	_plot_hist2d(
		event_summary['P'].fillna(0),
		event_summary['S'].fillna(0),
		bins=40,
		title='P picks vs S picks (2D hist)',
		xlabel='P picks',
		ylabel='S picks',
		outpath=fig_dir / 'hist2d_P_vs_S.png',
	)

	# travel-time histograms (event level)
	def _tt_clip(col: str, tmax: float, fname: str) -> None:
		v = event_summary[col].dropna()
		v = v[(v >= 0) & (v <= tmax)]
		_plot_hist_logy(
			v,
			bins=int(tmax),
			title=f'{col} clipped 0-{tmax}s (log-y)',
			xlabel=f'{col} [s]',
			outpath=fig_dir / fname,
		)

	for col, tmax, fname in [
		('min_P', 120.0, 'hist_event_min_P_0_120s.png'),
		('max_P', 200.0, 'hist_event_max_P_0_200s.png'),
		('min_S', 200.0, 'hist_event_min_S_0_200s.png'),
		('max_S', 300.0, 'hist_event_max_S_0_300s.png'),
	]:
		if col in event_summary.columns:
			_tt_clip(col, tmax, fname)

	# distance vs travel time (pick level)
	for phase, tmax, fname in [
		('P', 70.0, 'hist2d_dist_vs_tt_P.png'),
		('S', 70.0, 'hist2d_dist_vs_tt_S.png'),
	]:
		df = picks[picks['phase_type'] == phase].copy()
		df = df[
			(df['dist_km'] >= 0)
			& (df['dist_km'] <= clip_dist_km)
			& (df['tt_s'] >= 0)
			& (df['tt_s'] <= tmax)
		]
		if not df.empty:
			_plot_hist2d(
				df['dist_km'],
				df['tt_s'],
				bins=80,
				title=f'Distance vs travel time ({phase})',
				xlabel='distance [km]',
				ylabel='travel time [s]',
				outpath=fig_dir / fname,
			)

	# station-level S-P
	sp = station_sp.copy()
	sp = sp[
		(sp['S_minus_P_s'] >= 0)
		& (sp['S_minus_P_s'] <= 120)
		& (sp['dist_km'] <= clip_dist_km)
	]
	if not sp.empty:
		_plot_hist_logy(
			sp['S_minus_P_s'],
			bins=120,
			title='Station-level S-P (0-120s, log-y)',
			xlabel='S-P [s]',
			outpath=fig_dir / 'hist_station_SP_0_120s.png',
		)
		_plot_hist2d(
			sp['dist_km'],
			sp['S_minus_P_s'],
			bins=80,
			title='Distance vs S-P (2D hist)',
			xlabel='distance [km]',
			ylabel='S-P [s]',
			outpath=fig_dir / 'hist2d_dist_vs_SP.png',
		)

	# top stations
	top = station_summary.head(20).copy()
	plt.figure(figsize=(10, 5))
	plt.bar(top['station_code'], top['total'])
	plt.title('Top 20 stations by total picks')
	plt.xlabel('station_code')
	plt.ylabel('total picks')
	plt.xticks(rotation=60, ha='right')
	_savefig(fig_dir / 'bar_top20_stations_total.png')

	# magnitude views
	if 'mag1' in event_summary.columns:
		m = event_summary['mag1'].dropna()
		if not m.empty:
			plt.figure(figsize=(8, 4.5))
			plt.hist(m, bins=60)
			plt.title('Magnitude (mag1) histogram')
			plt.xlabel('mag1')
			plt.ylabel('count')
			_savefig(fig_dir / 'hist_mag1.png')

			tmp = event_summary.dropna(subset=['mag1', 'total_picks']).copy()
			mmin = np.floor(tmp['mag1'].min() * 2) / 2
			mmax = np.ceil(tmp['mag1'].max() * 2) / 2
			bins = np.arange(mmin, mmax + 0.5, 0.5)
			tmp['m_bin'] = pd.cut(tmp['mag1'], bins=bins, include_lowest=True)
			grp = tmp.groupby('m_bin', observed=True)['total_picks'].median()

			plt.figure(figsize=(10, 5))
			plt.bar([str(x) for x in grp.index], grp.values)
			plt.title('Median total_picks by mag1 bin (0.5 step)')
			plt.xlabel('mag1 bin')
			plt.ylabel('median total_picks')
			plt.xticks(rotation=60, ha='right')
			_savefig(fig_dir / 'bar_mag1_bin_median_total_picks.png')
