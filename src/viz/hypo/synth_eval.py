from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from hypo.uncertainty_ellipsoid import ELLIPSE_COLS
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

	fig, ax = plt.subplots(figsize=(8, 8))

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
	scatter_points_3view,
	sync_xyz_ranges,
)


def _build_true_pred_xyz_3view_figure(
	true_xyz_m: np.ndarray,
	pred_xyz_m: np.ndarray,
	*,
	stations_xyz_m: np.ndarray | None,
	stations_is_das: np.ndarray | None,
	marker_size: float,
	station_size: float,
	geophone_size: float | None,
	das_size: float | None,
	geophone_color: str | None,
	das_color: str | None,
	geophone_marker: str,
	das_marker: str,
) -> tuple[
	plt.Figure,
	plt.Axes,
	plt.Axes,
	plt.Axes,
	plt.Axes,
	np.ndarray,
	np.ndarray,
	np.ndarray | None,
	np.ndarray | None,
	list[object],
]:
	true_xyz = np.asarray(true_xyz_m, float).reshape(-1, 3) / 1000.0
	pred_xyz = np.asarray(pred_xyz_m, float).reshape(-1, 3) / 1000.0
	if true_xyz.shape != pred_xyz.shape:
		raise ValueError(
			f'true/pred shape mismatch: {true_xyz.shape} vs {pred_xyz.shape}'
		)

	st_xyz = None
	if stations_xyz_m is not None:
		st_xyz = np.asarray(stations_xyz_m, float).reshape(-1, 3) / 1000.0

	mask_is_das = None
	if stations_is_das is not None:
		if st_xyz is None:
			raise ValueError('stations_is_das was provided but stations_xyz_m is None')

		mask = np.asarray(stations_is_das).reshape(-1)
		if mask.size != st_xyz.shape[0]:
			raise ValueError(
				'len(stations_is_das) must match stations_xyz_m rows: '
				f'{mask.size} vs {st_xyz.shape[0]}'
			)

		if mask.dtype == bool:
			mask_is_das = mask
		elif mask.dtype.kind in ('b', 'i', 'u'):
			mask_is_das = mask.astype(bool)
		else:
			raise TypeError(
				'stations_is_das must be a bool array or integer array (0/1). '
				f'got dtype={mask.dtype!s}'
			)

	fig, ax_xy, ax_xz, ax_yz, ax_empty = make_3view_axes(figsize=(12.0, 12.0))

	# Draw order (top -> bottom): Pred > True > Geophone > DAS
	# Links should stay behind all points.
	z_link = 5.0
	z_das = 10.0
	z_geo = 20.0
	z_sta = 20.0
	z_true = 30.0
	z_pred = 40.0

	h_true = ax_xy.scatter(
		true_xyz[:, 0],
		true_xyz[:, 1],
		s=marker_size,
		marker='o',
		label='True',
		alpha=0.8,
		zorder=z_true,
	)
	h_pred = ax_xy.scatter(
		pred_xyz[:, 0],
		pred_xyz[:, 1],
		s=marker_size,
		marker='x',
		label='Pred',
		alpha=0.8,
		zorder=z_pred,
	)

	ax_xz.scatter(
		true_xyz[:, 0],
		true_xyz[:, 2],
		s=marker_size,
		marker='o',
		alpha=0.8,
		zorder=z_true,
	)
	ax_xz.scatter(
		pred_xyz[:, 0],
		pred_xyz[:, 2],
		s=marker_size,
		marker='x',
		alpha=0.8,
		zorder=z_pred,
	)

	ax_yz.scatter(
		true_xyz[:, 2],
		true_xyz[:, 1],
		s=marker_size,
		marker='o',
		alpha=0.8,
		zorder=z_true,
	)
	ax_yz.scatter(
		pred_xyz[:, 2],
		pred_xyz[:, 1],
		s=marker_size,
		marker='x',
		alpha=0.8,
		zorder=z_pred,
	)

	h_sta = None
	h_geo = None
	h_das = None
	if st_xyz is not None and st_xyz.size > 0:
		if mask_is_das is None:
			kw: dict[str, object] = {
				's': float(station_size),
				'marker': '^',
				'zorder': z_sta,
			}
			# Keep matplotlib default color cycle (do not pass color=None)
			h_sta = scatter_points_3view(
				ax_xy,
				ax_xz,
				ax_yz,
				x=st_xyz[:, 0],
				y=st_xyz[:, 1],
				z=st_xyz[:, 2],
				yz_mode='z-y',
				label='Stations',
				alpha=0.8,
				**kw,
			)
		else:
			geo_size = (
				float(station_size) if geophone_size is None else float(geophone_size)
			)
			das_size_val = float(station_size) if das_size is None else float(das_size)

			geo_xyz = st_xyz[~mask_is_das]
			das_xyz = st_xyz[mask_is_das]

			if geo_xyz.shape[0] > 0:
				kw_geo: dict[str, object] = {
					's': geo_size,
					'marker': str(geophone_marker),
					'zorder': z_geo,
				}
				if geophone_color is not None:
					kw_geo['color'] = str(geophone_color)
				h_geo = scatter_points_3view(
					ax_xy,
					ax_xz,
					ax_yz,
					x=geo_xyz[:, 0],
					y=geo_xyz[:, 1],
					z=geo_xyz[:, 2],
					yz_mode='z-y',
					label='Geophone',
					alpha=0.8,
					**kw_geo,
				)

			if das_xyz.shape[0] > 0:
				kw_das: dict[str, object] = {
					's': das_size_val,
					'marker': str(das_marker),
					'zorder': z_das,
				}
				if das_color is not None:
					kw_das['color'] = str(das_color)
				h_das = scatter_points_3view(
					ax_xy,
					ax_xz,
					ax_yz,
					x=das_xyz[:, 0],
					y=das_xyz[:, 1],
					z=das_xyz[:, 2],
					yz_mode='z-y',
					label='DAS',
					alpha=0.8,
					**kw_das,
				)

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
			zorder=z_link,
			yz_mode='z-y',
		)

	handles: list[object] = [h_true, h_pred]
	if mask_is_das is None:
		if h_sta is not None:
			handles.append(h_sta)
	else:
		if h_geo is not None:
			handles.append(h_geo)
		if h_das is not None:
			handles.append(h_das)

	return (
		fig,
		ax_xy,
		ax_xz,
		ax_yz,
		ax_empty,
		true_xyz,
		pred_xyz,
		st_xyz,
		mask_is_das,
		handles,
	)


def _finalize_true_pred_xyz_3view(
	fig: plt.Figure,
	ax_xy: plt.Axes,
	ax_xz: plt.Axes,
	ax_yz: plt.Axes,
	ax_empty: plt.Axes,
	*,
	x_range: tuple[float, float],
	y_range: tuple[float, float],
	z_range: tuple[float, float],
	handles: list[object],
	title: str | None,
	out_png: Path,
) -> None:
	sync_xyz_ranges(
		ax_xy,
		ax_xz,
		ax_yz,
		x_range=x_range,
		y_range=y_range,
		z_range=z_range,
		invert_z=True,
		yz_mode='z-y',
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

	for ax in (ax_xy, ax_xz, ax_yz):
		ax.xaxis.labelpad = 2
		ax.yaxis.labelpad = 2
		ax.tick_params(axis='both', which='major', pad=2)
	if title:
		fig.suptitle(title, y=0.985)
	labels = [h.get_label() for h in handles]
	ax_empty.legend(handles, labels, loc='center')

	fig.subplots_adjust(
		left=0.06,
		right=0.99,
		bottom=0.07,
		top=0.95,
		wspace=0.32,
		hspace=0.32,
	)
	save_figure(fig, out_png, dpi=200)


def save_true_pred_xyz_3view(
	true_xyz_m: np.ndarray,
	pred_xyz_m: np.ndarray,
	out_png: Path,
	*,
	stations_xyz_m: np.ndarray | None = None,
	stations_is_das: np.ndarray | None = None,
	title: str | None = None,
	marker_size: float = 110.0,
	station_size: float = 100.0,
	geophone_size: float | None = None,
	das_size: float | None = 5,
	geophone_color: str | None = None,
	das_color: str | None = 'gray',
	geophone_marker: str = '^',
	das_marker: str = 's',
) -> None:
	(
		fig,
		ax_xy,
		ax_xz,
		ax_yz,
		ax_empty,
		true_xyz,
		pred_xyz,
		st_xyz,
		_,
		handles,
	) = _build_true_pred_xyz_3view_figure(
		true_xyz_m,
		pred_xyz_m,
		stations_xyz_m=stations_xyz_m,
		stations_is_das=stations_is_das,
		marker_size=marker_size,
		station_size=station_size,
		geophone_size=geophone_size,
		das_size=das_size,
		geophone_color=geophone_color,
		das_color=das_color,
		geophone_marker=geophone_marker,
		das_marker=das_marker,
	)

	xr, yr, zr = ranges_from_xyz(
		[true_xyz, pred_xyz, st_xyz] if st_xyz is not None else [true_xyz, pred_xyz]
	)
	_finalize_true_pred_xyz_3view(
		fig,
		ax_xy,
		ax_xz,
		ax_yz,
		ax_empty,
		x_range=(float(xr[0]), float(xr[1])),
		y_range=(float(yr[0]), float(yr[1])),
		z_range=(float(zr[0]), float(zr[1])),
		handles=handles,
		title=title,
		out_png=out_png,
	)


def _ellipse_axis_aligned_halfwidth(
	a_km: float, b_km: float, theta_rad: float
) -> tuple[float, float]:
	ct = float(np.cos(theta_rad))
	st = float(np.sin(theta_rad))
	hx = float(np.sqrt((a_km * ct) ** 2 + (b_km * st) ** 2))
	hy = float(np.sqrt((a_km * st) ** 2 + (b_km * ct) ** 2))
	if not np.isfinite(hx) or not np.isfinite(hy):
		raise ValueError(f'non-finite ellipse halfwidths: hx={hx}, hy={hy}')
	return hx, hy


def _ellipse_polyline(
	*,
	cx: float,
	cy: float,
	a_km: float,
	b_km: float,
	theta_rad: float,
	n_points: int,
) -> np.ndarray:
	if n_points < 20:
		raise ValueError('n_points must be >= 20')
	t = np.linspace(0.0, 2.0 * np.pi, int(n_points), endpoint=True)
	ct = float(np.cos(theta_rad))
	st = float(np.sin(theta_rad))
	x = cx + a_km * np.cos(t) * ct - b_km * np.sin(t) * st
	y = cy + a_km * np.cos(t) * st + b_km * np.sin(t) * ct
	xy = np.column_stack([x, y]).astype(float, copy=False)
	if not np.isfinite(xy).all():
		raise ValueError('ellipse polyline contains non-finite values')
	return xy


def _clip_ab(a_km: float, b_km: float, clip_km: float) -> tuple[float, float]:
	m = float(max(a_km, b_km))
	if m <= float(clip_km):
		return float(a_km), float(b_km)
	scale = float(clip_km) / m
	return float(a_km) * scale, float(b_km) * scale


def save_true_pred_xyz_3view_with_uncertainty(
	true_xyz_m: np.ndarray,
	pred_xyz_m: np.ndarray,
	df_eval: pd.DataFrame,
	out_png: Path,
	*,
	stations_xyz_m: np.ndarray | None = None,
	stations_is_das: np.ndarray | None = None,
	title: str | None = None,
	marker_size: float = 110.0,
	station_size: float = 100.0,
	geophone_size: float | None = None,
	das_size: float | None = 5,
	geophone_color: str | None = None,
	das_color: str | None = 'gray',
	geophone_marker: str = '^',
	das_marker: str = 's',
	sigma_scale_sec: float = 1.0,
	poor_thresh_km: float = 5.0,
	clip_km: float = 10.0,
	n_ellipse_points: int = 100,
	ellipse_lw: float = 0.8,
	ellipse_alpha: float = 0.85,
	ellipse_color_ok: str = 'tab:orange',
	ellipse_color_poor: str = 'tab:red',
) -> None:
	from hypo.uncertainty_ellipsoid import projected_ellipses_from_record

	if sigma_scale_sec <= 0.0 or not np.isfinite(float(sigma_scale_sec)):
		raise ValueError(f'invalid sigma_scale_sec: {sigma_scale_sec!r}')
	if poor_thresh_km <= 0.0 or not np.isfinite(float(poor_thresh_km)):
		raise ValueError(f'invalid poor_thresh_km: {poor_thresh_km!r}')
	if clip_km <= 0.0 or not np.isfinite(float(clip_km)):
		raise ValueError(f'invalid clip_km: {clip_km!r}')

	missing = [c for c in ELLIPSE_COLS if c not in df_eval.columns]
	if missing:
		raise KeyError(
			f'missing uncertainty columns for ellipse plotting: {missing}. '
			f'available={list(df_eval.columns)}'
		)

	(
		fig,
		ax_xy,
		ax_xz,
		ax_yz,
		ax_empty,
		true_xyz,
		pred_xyz,
		st_xyz,
		_,
		handles,
	) = _build_true_pred_xyz_3view_figure(
		true_xyz_m,
		pred_xyz_m,
		stations_xyz_m=stations_xyz_m,
		stations_is_das=stations_is_das,
		marker_size=marker_size,
		station_size=station_size,
		geophone_size=geophone_size,
		das_size=das_size,
		geophone_color=geophone_color,
		das_color=das_color,
		geophone_marker=geophone_marker,
		das_marker=das_marker,
	)

	if len(df_eval) != true_xyz.shape[0]:
		raise ValueError(
			'len(df_eval) must match number of events in true/pred arrays: '
			f'{len(df_eval)} vs {true_xyz.shape[0]}'
		)

	z_ell = 25.0
	segs_xy_ok: list[np.ndarray] = []
	segs_xy_poor: list[np.ndarray] = []
	segs_xz_ok: list[np.ndarray] = []
	segs_xz_poor: list[np.ndarray] = []
	segs_yz_ok: list[np.ndarray] = []
	segs_yz_poor: list[np.ndarray] = []

	pad_x = 0.0
	pad_y = 0.0
	pad_z = 0.0

	for i in range(len(df_eval)):
		rec = df_eval.iloc[i]
		ell = projected_ellipses_from_record(
			rec, sigma_scale_sec=float(sigma_scale_sec)
		)

		a_xy, b_xy = _clip_ab(
			float(ell['a_xy_km']), float(ell['b_xy_km']), float(clip_km)
		)
		a_xz, b_xz = _clip_ab(
			float(ell['a_xz_km']), float(ell['b_xz_km']), float(clip_km)
		)
		a_yz, b_yz = _clip_ab(
			float(ell['a_yz_km']), float(ell['b_yz_km']), float(clip_km)
		)

		th_xy = float(ell['theta_xy_rad'])
		th_xz = float(ell['theta_xz_rad'])
		th_yz = float(ell['theta_yz_rad'])

		hx_xy, hy_xy = _ellipse_axis_aligned_halfwidth(a_xy, b_xy, th_xy)
		hx_xz, hz_xz = _ellipse_axis_aligned_halfwidth(a_xz, b_xz, th_xz)
		hz_yz, hy_yz = _ellipse_axis_aligned_halfwidth(a_yz, b_yz, th_yz)

		pad_x = float(max(pad_x, hx_xy, hx_xz))
		pad_y = float(max(pad_y, hy_xy, hy_yz))
		pad_z = float(max(pad_z, hz_xz, hz_yz))

		cx_xy = float(pred_xyz[i, 0])
		cy_xy = float(pred_xyz[i, 1])
		cx_xz = float(pred_xyz[i, 0])
		cy_xz = float(pred_xyz[i, 2])
		cx_yz = float(pred_xyz[i, 1])
		cy_yz = float(pred_xyz[i, 2])

		poly_xy = _ellipse_polyline(
			cx=cx_xy,
			cy=cy_xy,
			a_km=a_xy,
			b_km=b_xy,
			theta_rad=th_xy,
			n_points=int(n_ellipse_points),
		)
		poly_xz = _ellipse_polyline(
			cx=cx_xz,
			cy=cy_xz,
			a_km=a_xz,
			b_km=b_xz,
			theta_rad=th_xz,
			n_points=int(n_ellipse_points),
		)
		poly_yz = _ellipse_polyline(
			cx=cx_yz,
			cy=cy_yz,
			a_km=a_yz,
			b_km=b_yz,
			theta_rad=th_yz,
			n_points=int(n_ellipse_points),
		)
		poly_yz = poly_yz[:, [1, 0]]

		is_poor = float(ell['ell_3d_max_km']) > float(poor_thresh_km)
		if is_poor:
			segs_xy_poor.append(poly_xy)
			segs_xz_poor.append(poly_xz)
			segs_yz_poor.append(poly_yz)
		else:
			segs_xy_ok.append(poly_xy)
			segs_xz_ok.append(poly_xz)
			segs_yz_ok.append(poly_yz)

	if segs_xy_ok:
		ax_xy.add_collection(
			LineCollection(
				segs_xy_ok,
				colors=str(ellipse_color_ok),
				linewidths=float(ellipse_lw),
				linestyles='dashed',
				alpha=float(ellipse_alpha),
				zorder=z_ell,
			)
		)
		ax_xz.add_collection(
			LineCollection(
				segs_xz_ok,
				colors=str(ellipse_color_ok),
				linewidths=float(ellipse_lw),
				linestyles='dashed',
				alpha=float(ellipse_alpha),
				zorder=z_ell,
			)
		)
		ax_yz.add_collection(
			LineCollection(
				segs_yz_ok,
				colors=str(ellipse_color_ok),
				linewidths=float(ellipse_lw),
				linestyles='dashed',
				alpha=float(ellipse_alpha),
				zorder=z_ell,
			)
		)

	if segs_xy_poor:
		ax_xy.add_collection(
			LineCollection(
				segs_xy_poor,
				colors=str(ellipse_color_poor),
				linewidths=float(ellipse_lw),
				alpha=float(ellipse_alpha),
				zorder=z_ell,
			)
		)
		ax_xz.add_collection(
			LineCollection(
				segs_xz_poor,
				colors=str(ellipse_color_poor),
				linewidths=float(ellipse_lw),
				alpha=float(ellipse_alpha),
				zorder=z_ell,
			)
		)
		ax_yz.add_collection(
			LineCollection(
				segs_yz_poor,
				colors=str(ellipse_color_poor),
				linewidths=float(ellipse_lw),
				alpha=float(ellipse_alpha),
				zorder=z_ell,
			)
		)

	xr, yr, zr = ranges_from_xyz(
		[true_xyz, pred_xyz, st_xyz] if st_xyz is not None else [true_xyz, pred_xyz]
	)
	xr2 = (float(xr[0]) - float(pad_x), float(xr[1]) + float(pad_x))
	yr2 = (float(yr[0]) - float(pad_y), float(yr[1]) + float(pad_y))
	zr2 = (float(zr[0]) - float(pad_z), float(zr[1]) + float(pad_z))

	if segs_xy_ok:
		handles.append(
			Line2D(
				[0.0], [0.0], color=str(ellipse_color_ok), lw=1.2, label='1σ ellipse'
			)
		)
	if segs_xy_poor:
		handles.append(
			Line2D(
				[0.0],
				[0.0],
				color=str(ellipse_color_poor),
				lw=1.2,
				label='1σ ellipse (poor)',
			)
		)

	_finalize_true_pred_xyz_3view(
		fig,
		ax_xy,
		ax_xz,
		ax_yz,
		ax_empty,
		x_range=xr2,
		y_range=yr2,
		z_range=zr2,
		handles=handles,
		title=title,
		out_png=out_png,
	)
