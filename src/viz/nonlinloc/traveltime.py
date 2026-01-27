from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _annotate_source_location_xy(
	ax: Any,
	x_km: float,
	y_km: float,
	*,
	label: str | None,
) -> None:
	"""TT マップ上にステーション位置を重ね描きする。"""
	ax.scatter([x_km], [y_km], marker='*', s=160)
	if label is None:
		return
	ax.text(
		x_km,
		y_km,
		label,
		ha='left',
		va='bottom',
		fontsize=9,
	)


def _plot_tt_slice(
	data_2d: np.ndarray,
	*,
	extent: list[float],
	xlabel: str,
	ylabel: str,
	title: str,
	colorbar_label: str,
	figsize: tuple[float, float],
	out_png: str | Path,
	annotate: Callable[[Any], Any] | None = None,
) -> Path:
	"""走時スライスの共通描画ユーティリティ。"""
	out_png = Path(out_png)
	out_png.parent.mkdir(parents=True, exist_ok=True)

	fig, ax = plt.subplots(figsize=figsize)
	im = ax.imshow(
		data_2d,
		origin='lower',
		extent=extent,
		aspect='auto',
	)
	if annotate is not None:
		annotate(ax)
	fig.colorbar(im, ax=ax, label=colorbar_label)

	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_title(title)

	fig.tight_layout()
	fig.savefig(out_png, dpi=200)
	plt.close(fig)
	return out_png


def plot_tt_horizontal_slice(
	slice_2d: np.ndarray,
	x_axis_km: np.ndarray,
	y_axis_km: np.ndarray,
	*,
	z_km: float,
	title: str,
	out_png: str | Path,
	source_xy_km: tuple[float, float] | None = None,
	source_label: str | None = None,
) -> Path:
	"""水平スライスの走時マップ。"""
	annotate: Callable[[Any], Any] | None
	if source_xy_km is None:
		annotate = None
	else:
		x_km, y_km = source_xy_km
		annotate = lambda ax: _annotate_source_location_xy(
			ax,
			x_km,
			y_km,
			label=source_label,
		)

	return _plot_tt_slice(
		slice_2d,
		extent=[
			float(x_axis_km[0]),
			float(x_axis_km[-1]),
			float(y_axis_km[0]),
			float(y_axis_km[-1]),
		],
		xlabel='x East (km)',
		ylabel='y North (km)',
		title=f'{title} | z={z_km:.2f} km',
		colorbar_label='Travel time (s)',
		figsize=(7, 6),
		out_png=out_png,
		annotate=annotate,
	)


def plot_tt_vertical_xz(
	section_xz: np.ndarray,
	x_axis_km: np.ndarray,
	z_axis_km: np.ndarray,
	*,
	title: str,
	out_png: str | Path,
) -> Path:
	"""X-Z断面（固定 y インデックス）。"""
	return _plot_tt_slice(
		section_xz,
		extent=[
			float(x_axis_km[0]),
			float(x_axis_km[-1]),
			float(z_axis_km[0]),
			float(z_axis_km[-1]),
		],
		xlabel='x East (km)',
		ylabel='z (km)',
		title=title,
		colorbar_label='Travel time (s)',
		figsize=(7, 5.5),
		out_png=out_png,
	)


def plot_tt_vertical_yz(
	section_yz: np.ndarray,
	y_axis_km: np.ndarray,
	z_axis_km: np.ndarray,
	*,
	title: str,
	out_png: str | Path,
) -> Path:
	"""Y-Z断面（固定 x インデックス）。"""
	return _plot_tt_slice(
		section_yz,
		extent=[
			float(y_axis_km[0]),
			float(y_axis_km[-1]),
			float(z_axis_km[0]),
			float(z_axis_km[-1]),
		],
		xlabel='y North (km)',
		ylabel='z (km)',
		title=title,
		colorbar_label='Travel time (s)',
		figsize=(7, 5.5),
		out_png=out_png,
	)


def plot_tt_ps_difference_slice(
	diff_2d: np.ndarray,
	x_axis_km: np.ndarray,
	y_axis_km: np.ndarray,
	*,
	z_km: float,
	title: str,
	out_png: str | Path,
) -> Path:
	"""S - P の水平スライス差分。"""
	return _plot_tt_slice(
		diff_2d,
		extent=[
			float(x_axis_km[0]),
			float(x_axis_km[-1]),
			float(y_axis_km[0]),
			float(y_axis_km[-1]),
		],
		xlabel='x East (km)',
		ylabel='y North (km)',
		title=f'{title} | z={z_km:.2f} km',
		colorbar_label='S - P (s)',
		figsize=(7, 6),
		out_png=out_png,
	)
