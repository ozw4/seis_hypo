# src/qc/event_quality_plot.py
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from viz.core.fig_io import save_current_figure


def _plot_histograms(
	df: pd.DataFrame,
	out_dir: Path,
	metrics: list[str],
	hist_ranges: dict[str, tuple[float, float]] | None = None,
) -> None:
	"""指定された指標カラムについてヒストグラムを描く（平均・中央値の線付き）。"""
	for col in metrics:
		if col not in df.columns:
			continue

		data = df[col].dropna()
		if data.empty:
			continue

		values = data.to_numpy(dtype=float)
		mean_val = float(values.mean())
		median_val = float(np.median(values))

		rng: tuple[float, float] | None = None
		if hist_ranges is not None and col in hist_ranges:
			rmin, rmax = hist_ranges[col]
			rng = (float(rmin), float(rmax))

		plt.figure()
		if rng is None:
			plt.hist(values, bins=30)
		else:
			plt.hist(values, bins=30, range=rng)

		plt.axvline(
			mean_val,
			linestyle='--',
			linewidth=1.5,
			label=f'mean={mean_val:.2f}',
		)
		plt.axvline(
			median_val,
			linestyle=':',
			linewidth=1.5,
			label=f'median={median_val:.2f}',
		)
		plt.legend()

		plt.xlabel(col)
		plt.ylabel('Count')
		plt.title(f'Histogram of {col}')
		plt.tight_layout()
		save_current_figure(out_dir / f'{col}_hist.png', dpi=200)


def _plot_vs_nwr(
	df: pd.DataFrame,
	out_dir: Path,
	metrics: list[str],
	nwr_col: str,
) -> None:
	"""各指標 vs NWR の散布図を描く。"""
	if nwr_col not in df.columns:
		return

	for col in metrics:
		if col not in df.columns:
			continue

		plt.figure()
		plt.scatter(df[nwr_col], df[col], s=8)
		plt.xlabel(f'{nwr_col} (weighted readings)')
		plt.ylabel(col)
		plt.title(f'{col} vs {nwr_col}')
		plt.tight_layout()
		save_current_figure(out_dir / f'{col}_vs_{nwr_col}.png', dpi=200)


def _plot_rms_vs_geometry(
	df: pd.DataFrame,
	out_dir: Path,
	rms_col: str,
	dmin_col: str,
	gap_col: str,
) -> None:
	"""RMS vs DMIN / RMS vs GAP の幾何指標プロット。"""
	if dmin_col in df.columns and rms_col in df.columns:
		plt.figure()
		plt.scatter(df[dmin_col], df[rms_col], s=8)
		plt.xlabel(f'{dmin_col} (km)')
		plt.ylabel(rms_col)
		plt.title(f'{rms_col} vs {dmin_col}')
		plt.tight_layout()
		save_current_figure(out_dir / f'{rms_col}_vs_{dmin_col}.png', dpi=200)

	if gap_col in df.columns and rms_col in df.columns:
		plt.figure()
		plt.scatter(df[gap_col], df[rms_col], s=8)
		plt.xlabel(f'{gap_col} (deg)')
		plt.ylabel(rms_col)
		plt.title(f'{rms_col} vs {gap_col}')
		plt.tight_layout()
		save_current_figure(out_dir / f'{rms_col}_vs_{gap_col}.png', dpi=200)


def _plot_metrics_vs_depth(
	df: pd.DataFrame,
	out_dir: Path,
	metrics: list[str],
	depth_col: str,
) -> None:
	"""RMS・ERH・ERZ などと深さの関係を散布図で描く。"""
	if depth_col not in df.columns:
		return

	for col in metrics:
		if col not in df.columns:
			continue

		plt.figure()
		plt.scatter(df[depth_col], df[col], s=8)
		plt.xlabel(f'{depth_col} (km)')
		plt.ylabel(col)
		plt.title(f'{col} vs {depth_col}')
		plt.tight_layout()
		save_current_figure(out_dir / f'{col}_vs_{depth_col}.png', dpi=200)


def _plot_spatial_rms(
	df: pd.DataFrame,
	out_dir: Path,
	rms_col: str,
	lat_col: str,
	lon_col: str,
) -> None:
	"""XY 上の空間分布として RMS をカラーマップ表示する。"""
	if (
		rms_col not in df.columns
		or lat_col not in df.columns
		or lon_col not in df.columns
	):
		return

	plt.figure()
	sc = plt.scatter(df[lon_col], df[lat_col], c=df[rms_col], s=8)
	plt.xlabel('Longitude')
	plt.ylabel('Latitude')
	plt.title(f'Spatial distribution of {rms_col}')
	plt.colorbar(sc, label=rms_col)
	plt.tight_layout()
	save_current_figure(out_dir / f'{rms_col}_spatial.png', dpi=200)


def _plot_coord_misfit(
	df: pd.DataFrame,
	out_dir: Path,
	*,
	lat_col_jma: str,
	lon_col_jma: str,
	depth_col_jma: str,
	lat_col_hyp: str,
	lon_col_hyp: str,
	depth_col_hyp: str,
	rms_col: str = 'RMS',
	mag_col_jma: str | None = None,
) -> None:
	"""JMA vs HYPOINV の座標ミスフィットを水平距離・深さ・RMS・マグで評価する。"""
	required_cols = [
		lat_col_jma,
		lon_col_jma,
		depth_col_jma,
		lat_col_hyp,
		lon_col_hyp,
		depth_col_hyp,
	]
	for c in required_cols:
		if c not in df.columns:
			return

	lat_jma = df[lat_col_jma].to_numpy(dtype=float)
	lon_jma = df[lon_col_jma].to_numpy(dtype=float)
	depth_jma = df[depth_col_jma].to_numpy(dtype=float)
	lat_hyp = df[lat_col_hyp].to_numpy(dtype=float)
	lon_hyp = df[lon_col_hyp].to_numpy(dtype=float)
	depth_hyp = df[depth_col_hyp].to_numpy(dtype=float)

	mask_horiz = (
		np.isfinite(lat_jma)
		& np.isfinite(lon_jma)
		& np.isfinite(lat_hyp)
		& np.isfinite(lon_hyp)
	)
	if not np.any(mask_horiz):
		return

	lat_jma_rad = np.deg2rad(lat_jma[mask_horiz])
	dlat_deg = lat_hyp[mask_horiz] - lat_jma[mask_horiz]
	dlon_deg = lon_hyp[mask_horiz] - lon_jma[mask_horiz]

	km_per_deg = 111.19
	dy_km = dlat_deg * km_per_deg
	dx_km = dlon_deg * km_per_deg * np.cos(lat_jma_rad)
	dr_horiz = np.sqrt(dx_km**2 + dy_km**2)

	mask_depth = np.isfinite(depth_jma) & np.isfinite(depth_hyp)
	dz_km = depth_hyp[mask_depth] - depth_jma[mask_depth]

	# 水平ミスフィットのヒストグラム
	plt.figure()
	plt.hist(dr_horiz, bins=30)
	plt.xlabel('Horizontal distance |JMA - HYPOINV| (km)')
	plt.ylabel('Count')
	plt.title('Histogram of horizontal hypocenter misfit')
	plt.tight_layout()
	save_current_figure(out_dir / 'coordmisfit_horizontal_hist.png', dpi=200)

	# 深さミスフィットのヒストグラム
	if np.any(mask_depth):
		plt.figure()
		plt.hist(dz_km, bins=30)
		plt.xlabel('Depth difference HYPOINV - JMA (km)')
		plt.ylabel('Count')
		plt.title('Histogram of depth misfit')
		plt.tight_layout()
		save_current_figure(out_dir / 'coordmisfit_depth_hist.png', dpi=200)

	# RMS vs 水平ミスフィット
	if rms_col in df.columns:
		rms = df[rms_col].to_numpy(dtype=float)
		mask_rms = np.isfinite(rms)
		mask = mask_horiz & mask_rms
		if np.any(mask):
			plt.figure()
			plt.scatter(dr_horiz[mask_horiz & mask_rms], rms[mask], s=8)
			plt.xlabel('Horizontal distance |JMA - HYPOINV| (km)')
			plt.ylabel(rms_col)
			plt.title(f'{rms_col} vs horizontal misfit')
			plt.tight_layout()
			save_current_figure(out_dir / 'coordmisfit_RMS_vs_horizontal.png', dpi=200)

	# 空間分布（JMA 位置をプロットしつつ色で水平ミスフィット）
	plt.figure()
	sc = plt.scatter(
		lon_jma[mask_horiz],
		lat_jma[mask_horiz],
		c=dr_horiz,
		s=8,
	)
	plt.xlabel('Longitude (JMA)')
	plt.ylabel('Latitude (JMA)')
	plt.title('Spatial distribution of JMA–HYPOINV horizontal misfit')
	plt.colorbar(sc, label='Horizontal misfit (km)')
	plt.tight_layout()
	save_current_figure(out_dir / 'coordmisfit_horizontal_spatial.png', dpi=200)

	# マグニチュード vs 水平ミスフィット（JMA マグを想定）
	if mag_col_jma is not None and mag_col_jma in df.columns:
		mag = df[mag_col_jma].to_numpy(dtype=float)
		mask_mag = np.isfinite(mag)
		mask2 = mask_horiz & mask_mag
		if np.any(mask2):
			plt.figure()
			plt.scatter(mag[mask2], dr_horiz[mask2], s=8)
			plt.xlabel(mag_col_jma)
			plt.ylabel('Horizontal distance |JMA - HYPOINV| (km)')
			plt.title(f'Horizontal misfit vs {mag_col_jma}')
			plt.tight_layout()
			save_current_figure(out_dir / 'coordmisfit_horizontal_vs_mag_jma.png', dpi=200)


def plot_event_quality(
	df: pd.DataFrame,
	out_dir: str | Path = '.',
	*,
	lat_col: str = 'lat_deg',
	lon_col: str = 'lon_deg',
	depth_col: str = 'depth_km',
	rms_col: str = 'RMS',
	erh_col: str = 'ERH',
	erz_col: str = 'ERZ',
	nwr_col: str = 'NWR',
	dmin_col: str = 'DMIN',
	gap_col: str = 'GAP',
	lat_col_jma: str = 'lat_deg_jma',
	lon_col_jma: str = 'lon_deg_jma',
	depth_col_jma: str = 'depth_km_jma',
	lat_col_hyp: str = 'lat_deg_hyp',
	lon_col_hyp: str = 'lon_deg_hyp',
	depth_col_hyp: str = 'depth_km_hyp',
	mag_col_jma: str = 'mag1_jma',
	hist_ranges: dict[str, tuple[float, float]] | None = None,
) -> None:
	"""震源品質と JMA–HYPOINV 座標誤差の標準的な可視化一式をまとめて作る。

	作る図:
	  - ヒストグラム: RMS / ERH / ERZ（必要なら hist_ranges で範囲指定）
	  - vs NWR: RMS / ERH / ERZ vs NWR
	  - 幾何: RMS vs DMIN, RMS vs GAP
	  - vs depth: RMS / ERH / ERZ vs depth_col
	  - 空間分布: (lon, lat) 上で RMS を色付け
	  - 座標誤差: JMA vs HYPOINV の水平距離・深さ差・RMS・マグなど
	"""
	out = Path(out_dir)
	out.mkdir(parents=True, exist_ok=True)

	metrics = [rms_col, erh_col, erz_col]

	_plot_histograms(df, out, metrics, hist_ranges=hist_ranges)
	_plot_vs_nwr(df, out, metrics, nwr_col=nwr_col)
	_plot_rms_vs_geometry(df, out, rms_col=rms_col, dmin_col=dmin_col, gap_col=gap_col)
	_plot_metrics_vs_depth(df, out, metrics, depth_col=depth_col)
	_plot_spatial_rms(df, out, rms_col=rms_col, lat_col=lat_col, lon_col=lon_col)

	if (
		lat_col_jma is not None
		and lon_col_jma is not None
		and depth_col_jma is not None
		and lat_col_hyp is not None
		and lon_col_hyp is not None
		and depth_col_hyp is not None
	):
		_plot_coord_misfit(
			df,
			out,
			lat_col_jma=lat_col_jma,
			lon_col_jma=lon_col_jma,
			depth_col_jma=depth_col_jma,
			lat_col_hyp=lat_col_hyp,
			lon_col_hyp=lon_col_hyp,
			depth_col_hyp=depth_col_hyp,
			rms_col=rms_col,
			mag_col_jma=mag_col_jma,
		)
	else:
		print('Skipping coordinate misfit plots: required columns not found.')
