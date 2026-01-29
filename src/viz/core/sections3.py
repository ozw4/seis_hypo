from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


@contextmanager
def freeze_limits(ax: Axes):
	xlim = ax.get_xlim()
	ylim = ax.get_ylim()
	ax.set_autoscale_on(False)
	try:
		yield
	finally:
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)


def make_3view_axes(
	*,
	figsize: tuple[float, float] = (10.0, 10.0),
	width_ratios: tuple[float, float] = (3.0, 1.5),
	height_ratios: tuple[float, float] = (3.0, 1.5),
	wspace: float = 0.1,
	hspace: float = 0.1,
) -> tuple[Figure, Axes, Axes, Axes, Axes]:
	fig = plt.figure(figsize=figsize)
	gs = fig.add_gridspec(
		2,
		2,
		width_ratios=width_ratios,
		height_ratios=height_ratios,
		wspace=wspace,
		hspace=hspace,
	)
	ax_xy = fig.add_subplot(gs[0, 0])
	ax_yz = fig.add_subplot(gs[0, 1])
	ax_xz = fig.add_subplot(gs[1, 0])
	ax_empty = fig.add_subplot(gs[1, 1])
	ax_empty.axis('off')
	return fig, ax_xy, ax_xz, ax_yz, ax_empty


def _pad_range(
	vmin: float, vmax: float, *, pad_frac: float = 0.05
) -> tuple[float, float]:
	if not np.isfinite(vmin) or not np.isfinite(vmax):
		return (-1.0, 1.0)
	if vmin == vmax:
		return (vmin - 1.0, vmax + 1.0)
	pad = (vmax - vmin) * float(pad_frac)
	return (vmin - pad, vmax + pad)


def sync_xyz_ranges(
	ax_xy: Axes,
	ax_xz: Axes,
	ax_yz: Axes,
	*,
	x_range: tuple[float, float],
	y_range: tuple[float, float],
	z_range: tuple[float, float],
	invert_z: bool = True,
	yz_mode: str = 'y-z',  # 'y-z' or 'z-y'
) -> None:
	ax_xy.set_xlim(*x_range)
	ax_xy.set_ylim(*y_range)

	ax_xz.set_xlim(*x_range)
	ax_xz.set_ylim(*z_range)
	if invert_z:
		ax_xz.invert_yaxis()

	if yz_mode == 'y-z':
		ax_yz.set_xlim(*y_range)
		ax_yz.set_ylim(*z_range)
		if invert_z:
			ax_yz.invert_yaxis()
	elif yz_mode == 'z-y':
		ax_yz.set_xlim(*z_range)
		ax_yz.set_ylim(*y_range)
	else:
		raise ValueError(f'unknown yz_mode: {yz_mode}')


def plot_links_3view(
	ax_xy: Axes,
	ax_xz: Axes,
	ax_yz: Axes,
	*,
	pairs_xyz: Iterable[tuple[tuple[float, float, float], tuple[float, float, float]]],
	color: str = 'black',
	linewidth: float = 0.6,
	alpha: float = 0.35,
	label: str | None = None,
	linestyle: str = ':',
	zorder: float = 2.6,
	yz_mode: str = 'y-z',
) -> None:
	first = True
	for (x1, y1, z1), (x2, y2, z2) in pairs_xyz:
		lbl = label if (label is not None and first) else None
		first = False

		ax_xy.plot(
			[float(x1), float(x2)],
			[float(y1), float(y2)],
			color=color,
			linewidth=float(linewidth),
			alpha=float(alpha),
			linestyle=linestyle,
			label=lbl,
			zorder=zorder,
		)
		ax_xz.plot(
			[float(x1), float(x2)],
			[float(z1), float(z2)],
			color=color,
			linewidth=float(linewidth),
			alpha=float(alpha),
			linestyle=linestyle,
			zorder=zorder,
		)

		if yz_mode == 'y-z':
			ax_yz.plot(
				[float(y1), float(y2)],
				[float(z1), float(z2)],
				color=color,
				linewidth=float(linewidth),
				alpha=float(alpha),
				linestyle=linestyle,
				zorder=zorder,
			)
		elif yz_mode == 'z-y':
			ax_yz.plot(
				[float(z1), float(z2)],
				[float(y1), float(y2)],
				color=color,
				linewidth=float(linewidth),
				alpha=float(alpha),
				linestyle=linestyle,
				zorder=zorder,
			)
		else:
			raise ValueError(f'unknown yz_mode: {yz_mode}')


def ranges_from_xyz(
	xyz_list: list[np.ndarray],
	*,
	pad_frac: float = 0.05,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
	pts = [np.asarray(a, float) for a in xyz_list if a is not None]
	if not pts:
		return ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
	all_xyz = np.vstack([a.reshape(-1, 3) for a in pts])
	mask = np.isfinite(all_xyz).all(axis=1)
	all_xyz = all_xyz[mask]
	if all_xyz.size == 0:
		return ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))

	x0, x1 = float(all_xyz[:, 0].min()), float(all_xyz[:, 0].max())
	y0, y1 = float(all_xyz[:, 1].min()), float(all_xyz[:, 1].max())
	z0, z1 = float(all_xyz[:, 2].min()), float(all_xyz[:, 2].max())

	return (
		_pad_range(x0, x1, pad_frac=pad_frac),
		_pad_range(y0, y1, pad_frac=pad_frac),
		_pad_range(z0, z1, pad_frac=pad_frac),
	)


@contextmanager
def freeze_limits(ax: Axes):
	xlim = ax.get_xlim()
	ylim = ax.get_ylim()
	ax.set_autoscale_on(False)
	try:
		yield
	finally:
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)


def make_3view_axes(
	*,
	figsize: tuple[float, float] = (10.0, 10.0),
	width_ratios: tuple[float, float] = (3.0, 1.5),
	height_ratios: tuple[float, float] = (3.0, 1.5),
	wspace: float = 0.1,
	hspace: float = 0.1,
) -> tuple[Figure, Axes, Axes, Axes, Axes]:
	fig = plt.figure(figsize=figsize)
	gs = fig.add_gridspec(
		2,
		2,
		width_ratios=width_ratios,
		height_ratios=height_ratios,
		wspace=wspace,
		hspace=hspace,
	)
	ax_xy = fig.add_subplot(gs[0, 0])
	ax_yz = fig.add_subplot(gs[0, 1])
	ax_xz = fig.add_subplot(gs[1, 0])
	ax_empty = fig.add_subplot(gs[1, 1])
	ax_empty.axis('off')
	return fig, ax_xy, ax_xz, ax_yz, ax_empty


def sync_xyz_ranges(
	ax_xy: Axes,
	ax_xz: Axes,
	ax_yz: Axes,
	*,
	x_range: tuple[float, float],
	y_range: tuple[float, float],
	z_range: tuple[float, float],
	invert_z: bool = True,
	yz_mode: str = 'y-z',  # 'y-z' or 'z-y'
) -> None:
	ax_xy.set_xlim(*x_range)
	ax_xy.set_ylim(*y_range)

	ax_xz.set_xlim(*x_range)
	ax_xz.set_ylim(*z_range)
	if invert_z:
		ax_xz.invert_yaxis()

	if yz_mode == 'y-z':
		ax_yz.set_xlim(*y_range)
		ax_yz.set_ylim(*z_range)
		if invert_z:
			ax_yz.invert_yaxis()
	elif yz_mode == 'z-y':
		ax_yz.set_xlim(*z_range)
		ax_yz.set_ylim(*y_range)
	else:
		raise ValueError(f'unknown yz_mode: {yz_mode}')


def plot_links_3view(
	ax_xy: Axes,
	ax_xz: Axes,
	ax_yz: Axes,
	*,
	pairs_xyz: Iterable[tuple[tuple[float, float, float], tuple[float, float, float]]],
	color: str = 'black',
	linewidth: float = 0.6,
	alpha: float = 0.35,
	label: str | None = None,
	linestyle: str = ':',
	zorder: float = 2.6,
	yz_mode: str = 'y-z',
) -> None:
	first = True
	for (x1, y1, z1), (x2, y2, z2) in pairs_xyz:
		lbl = label if (label is not None and first) else None
		first = False

		ax_xy.plot(
			[float(x1), float(x2)],
			[float(y1), float(y2)],
			color=color,
			linewidth=float(linewidth),
			alpha=float(alpha),
			linestyle=linestyle,
			label=lbl,
			zorder=zorder,
		)

		ax_xz.plot(
			[float(x1), float(x2)],
			[float(z1), float(z2)],
			color=color,
			linewidth=float(linewidth),
			alpha=float(alpha),
			linestyle=linestyle,
			zorder=zorder,
		)

		if yz_mode == 'y-z':
			ax_yz.plot(
				[float(y1), float(y2)],
				[float(z1), float(z2)],
				color=color,
				linewidth=float(linewidth),
				alpha=float(alpha),
				linestyle=linestyle,
				zorder=zorder,
			)
		elif yz_mode == 'z-y':
			ax_yz.plot(
				[float(z1), float(z2)],
				[float(y1), float(y2)],
				color=color,
				linewidth=float(linewidth),
				alpha=float(alpha),
				linestyle=linestyle,
				zorder=zorder,
			)
		else:
			raise ValueError(f'unknown yz_mode: {yz_mode}')


def scatter_points_3view(
	ax_xy: Axes,
	ax_xz: Axes,
	ax_yz: Axes,
	*,
	x: np.ndarray,
	y: np.ndarray,
	z: np.ndarray,
	yz_mode: str = 'y-z',
	label: str | None = None,
	**scatter_kwargs,
):
	"""同一点群(x,y,z)を XY/XZ/YZ に散布図として描く。

	- label は XY のみに付ける（凡例重複を防ぐ）
	- yz_mode:
	- 'y-z' => YZ は (y, z)
	- 'z-y' => YZ は (z, y)  ※ events_map の Depth vs Lat に対応
	"""
	xv = np.asarray(x, float).ravel()
	yv = np.asarray(y, float).ravel()
	zv = np.asarray(z, float).ravel()
	if not (xv.size == yv.size == zv.size):
		raise ValueError('x,y,z length mismatch')

	# color が指定されていない場合、ax_xy 側で自動色を決めさせ、
	# 採用色(RGBA)を xz/yz にも反映して 3view の見た目を揃える。
	use_auto_color = ('color' not in scatter_kwargs) and ('c' not in scatter_kwargs)

	h_xy = ax_xy.scatter(xv, yv, label=label, **scatter_kwargs)

	kw_other = dict(scatter_kwargs)
	if use_auto_color:
		rgba = None
		fc = h_xy.get_facecolors()
		if fc is not None and len(fc) > 0:
			rgba = fc[0]
		else:
			ec = h_xy.get_edgecolors()
			if ec is not None and len(ec) > 0:
				rgba = ec[0]
		if rgba is not None:
			kw_other['color'] = tuple(float(v) for v in rgba)

	ax_xz.scatter(xv, zv, label=None, **kw_other)

	if yz_mode == 'y-z':
		ax_yz.scatter(yv, zv, label=None, **kw_other)
	elif yz_mode == 'z-y':
		ax_yz.scatter(zv, yv, label=None, **kw_other)
	else:
		raise ValueError(f'unknown yz_mode: {yz_mode}')

	return h_xy
