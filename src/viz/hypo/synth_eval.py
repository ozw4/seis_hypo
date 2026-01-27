from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from viz.core.fig_io import save_current_figure, save_figure


def save_dxdy_scatter(
	dx_m: np.ndarray,
	dy_m: np.ndarray,
	out_png: Path,
	*,
	title: str = 'dx vs dy',
	xlabel: str = 'dx_m',
	ylabel: str = 'dy_m',
) -> Path:
	fig, ax = plt.subplots()
	ax.scatter(dx_m, dy_m, s=10)
	ax.axhline(0)
	ax.axvline(0)
	ax.set_title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)

	fig.tight_layout()
	return save_figure(fig, out_png, dpi=150)


def save_true_pred_xy_plot(
	true_xy: np.ndarray,
	pred_xy: np.ndarray,
	out_png: str,
	*,
	title: str | None = None,
	xlabel: str = 'X',
	ylabel: str = 'Y',
	marker_size: float = 20.0,
) -> None:
	true_xy = np.asarray(true_xy, dtype=float)
	pred_xy = np.asarray(pred_xy, dtype=float)

	if true_xy.shape != pred_xy.shape or true_xy.ndim != 2 or true_xy.shape[1] != 2:
		raise ValueError(
			f'true_xy/pred_xy は同じ形の (N,2) が必要です: true={true_xy.shape}, pred={pred_xy.shape}'
		)

	segs = np.stack([true_xy, pred_xy], axis=1)  # (N, 2, 2)

	fig, ax = plt.subplots(figsize=(7, 7))

	ax.scatter(true_xy[:, 0], true_xy[:, 1], s=marker_size, marker='o', label='True')
	ax.scatter(pred_xy[:, 0], pred_xy[:, 1], s=marker_size, marker='x', label='Pred')

	lc = LineCollection(segs, linestyles='dotted', linewidths=1.0, alpha=0.7)
	ax.add_collection(lc)

	all_xy = np.vstack([true_xy, pred_xy])
	x_min, x_max = float(all_xy[:, 0].min()), float(all_xy[:, 0].max())
	y_min, y_max = float(all_xy[:, 1].min()), float(all_xy[:, 1].max())

	x_pad = (x_max - x_min) * 0.05
	y_pad = (y_max - y_min) * 0.05
	if x_pad == 0.0:
		x_pad = 1.0
	if y_pad == 0.0:
		y_pad = 1.0

	ax.set_xlim(x_min - x_pad, x_max + x_pad)
	ax.set_ylim(y_min - y_pad, y_max + y_pad)
	ax.set_aspect('equal', adjustable='box')
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	if title:
		ax.set_title(title)
	ax.grid(True)
	ax.legend()

	fig.tight_layout()
	save_figure(fig, out_png, dpi=200)


def plot_xy_true_vs_hyp(df_eval: pd.DataFrame, out_png: Path) -> None:
	need = ['x_m_true', 'y_m_true', 'x_m_hyp', 'y_m_hyp']
	missing = [c for c in need if c not in df_eval.columns]
	if missing:
		raise KeyError(f'missing columns: {missing}. available={list(df_eval.columns)}')

	x0 = df_eval['x_m_true'].to_numpy(float)
	y0 = df_eval['y_m_true'].to_numpy(float)
	x1 = df_eval['x_m_hyp'].to_numpy(float)
	y1 = df_eval['y_m_hyp'].to_numpy(float)

	mask = np.isfinite(x0) & np.isfinite(y0) & np.isfinite(x1) & np.isfinite(y1)
	x0, y0, x1, y1 = x0[mask], y0[mask], x1[mask], y1[mask]

	fig, ax = plt.subplots()
	ax.scatter(x0, y0, label='True')
	ax.scatter(x1, y1, label='HypoInverse')

	for a, b, c, d in zip(x0, y0, x1, y1):
		ax.plot([a, c], [b, d], linestyle=':', linewidth=0.8)  # 点線リンク

	ax.set_xlabel('X (m)')
	ax.set_ylabel('Y (m)')
	ax.set_aspect('equal', adjustable='datalim')
	ax.legend()
	fig.tight_layout()

	save_figure(fig, out_png, dpi=200)


def save_hist(series: pd.Series, out_png: Path, title: str, xlabel: str) -> None:
	plt.figure()
	plt.hist(series.dropna().to_numpy(), bins=60)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel('count')
	plt.tight_layout()
	save_current_figure(out_png, dpi=150)


from viz.core.sections3 import (
	freeze_limits,
	make_3view_axes,
	plot_links_3view,
	ranges_from_xyz,
	sync_xyz_ranges,
)


def save_true_pred_xyz_3view(
	true_xyz_m: np.ndarray,
	pred_xyz_m: np.ndarray,
	out_png: Path,
	*,
	stations_xyz_m: np.ndarray | None = None,
	title: str | None = None,
	marker_size: float = 20.0,
	station_size: float = 16.0,
) -> None:
	true_xyz = np.asarray(true_xyz_m, float).reshape(-1, 3) / 1000.0
	pred_xyz = np.asarray(pred_xyz_m, float).reshape(-1, 3) / 1000.0
	if true_xyz.shape != pred_xyz.shape:
		raise ValueError(
			f'true/pred shape mismatch: {true_xyz.shape} vs {pred_xyz.shape}'
		)

	st_xyz = None
	if stations_xyz_m is not None:
		st_xyz = np.asarray(stations_xyz_m, float).reshape(-1, 3) / 1000.0

	fig, ax_xy, ax_xz, ax_yz, ax_empty = make_3view_axes(figsize=(10.0, 10.0))

	h_true = ax_xy.scatter(
		true_xyz[:, 0], true_xyz[:, 1], s=marker_size, marker='o', label='True'
	)
	h_pred = ax_xy.scatter(
		pred_xyz[:, 0], pred_xyz[:, 1], s=marker_size, marker='x', label='Pred'
	)

	ax_xz.scatter(true_xyz[:, 0], true_xyz[:, 2], s=marker_size, marker='o')
	ax_xz.scatter(pred_xyz[:, 0], pred_xyz[:, 2], s=marker_size, marker='x')

	ax_yz.scatter(true_xyz[:, 1], true_xyz[:, 2], s=marker_size, marker='o')
	ax_yz.scatter(pred_xyz[:, 1], pred_xyz[:, 2], s=marker_size, marker='x')

	h_sta = None
	if st_xyz is not None and st_xyz.size > 0:
		h_sta = ax_xy.scatter(
			st_xyz[:, 0], st_xyz[:, 1], s=station_size, marker='^', label='Stations'
		)
		ax_xz.scatter(st_xyz[:, 0], st_xyz[:, 2], s=station_size, marker='^')
		ax_yz.scatter(st_xyz[:, 1], st_xyz[:, 2], s=station_size, marker='^')

	pairs = [
		((a[0], a[1], a[2]), (b[0], b[1], b[2]))
		for a, b in zip(true_xyz, pred_xyz, strict=True)
	]

	with freeze_limits(ax_xy), freeze_limits(ax_xz), freeze_limits(ax_yz):
		plot_links_3view(
			ax_xy,
			ax_xz,
			ax_yz,
			pairs_xyz=pairs,
			color='black',
			linewidth=0.6,
			alpha=0.35,
			linestyle=':',
			yz_mode='y-z',
		)

	xr, yr, zr = ranges_from_xyz(
		[true_xyz, pred_xyz, st_xyz] if st_xyz is not None else [true_xyz, pred_xyz]
	)
	sync_xyz_ranges(
		ax_xy,
		ax_xz,
		ax_yz,
		x_range=xr,
		y_range=yr,
		z_range=zr,
		invert_z=True,
		yz_mode='y-z',
	)

	ax_xy.set_xlabel('X (km)')
	ax_xy.set_ylabel('Y (km)')
	ax_xy.set_aspect('equal', adjustable='box')
	ax_xy.grid(True)

	ax_xz.set_xlabel('X (km)')
	ax_xz.set_ylabel('Depth (km)')
	ax_xz.grid(True)

	ax_yz.set_xlabel('Y (km)')
	ax_yz.set_ylabel('Depth (km)')
	ax_yz.grid(True)

	if title:
		fig.suptitle(title)

	handles = [h_true, h_pred] + ([h_sta] if h_sta is not None else [])
	labels = [h.get_label() for h in handles]
	ax_empty.legend(handles, labels, loc='center')

	fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])
	save_figure(fig, out_png, dpi=200)
