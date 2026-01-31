from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg', force=True)

import numpy as np

from hypo.synth_eval.heatmap_types import GridAxes
from viz.hypo.synth_eval import (
	save_heatmap_2d,
	save_heatmap_xy_slices,
	save_heatmap_xz_center_y,
	save_heatmap_yz_center_x,
)


def _make_axes() -> GridAxes:
	return GridAxes(
		x_m=np.array([0, 1000], dtype=float),
		y_m=np.array([0, 2000, 3000], dtype=float),
		z_m=np.array([0, 1000], dtype=float),
	)


def _make_grid() -> np.ndarray:
	grid = np.arange(12, dtype=float).reshape((2, 3, 2))
	return grid


def test_save_heatmap_2d_creates_png(tmp_path: Path) -> None:
	out_png = tmp_path / 'heatmap.png'
	data = np.arange(6, dtype=float).reshape((2, 3))
	save_heatmap_2d(
		data,
		out_png,
		title='t',
		xlabel='X (km)',
		ylabel='Y (km)',
		extent=(0.0, 1.0, 0.0, 1.0),
		vmin=0.0,
		vmax=10.0,
	)
	assert out_png.is_file()
	assert out_png.stat().st_size > 0


def test_save_heatmap_xy_slices_outputs_all_z(tmp_path: Path) -> None:
	axes = _make_axes()
	grid = _make_grid()
	out_dir = tmp_path / 'xy'
	paths = save_heatmap_xy_slices(
		grid,
		axes,
		out_dir,
		metric='err3d_m',
		vmin=0.0,
		vmax=10.0,
	)
	assert len(paths) == len(axes.z_m)
	names = {p.name for p in paths}
	assert 'xy_z0m.png' in names
	assert 'xy_z1000m.png' in names
	for p in paths:
		assert p.is_file()
		assert p.stat().st_size > 0


def test_save_heatmap_xz_center_y_outputs_png(tmp_path: Path) -> None:
	axes = _make_axes()
	grid = _make_grid()
	out_dir = tmp_path / 'xz'
	out_png = save_heatmap_xz_center_y(
		grid,
		axes,
		out_dir,
		metric='err3d_m',
		vmin=0.0,
		vmax=10.0,
		center_y_index=axes.center_y_index(),
	)
	assert out_png.is_file()
	assert out_png.stat().st_size > 0
	assert out_png.name == 'xz_y2000m.png'


def test_save_heatmap_yz_center_x_outputs_png(tmp_path: Path) -> None:
	axes = _make_axes()
	grid = _make_grid()
	out_dir = tmp_path / 'yz'
	out_png = save_heatmap_yz_center_x(
		grid,
		axes,
		out_dir,
		metric='err3d_m',
		vmin=0.0,
		vmax=10.0,
		center_x_index=axes.center_x_index(),
	)
	assert out_png.is_file()
	assert out_png.stat().st_size > 0
	assert out_png.name == 'yz_x1000m.png'
