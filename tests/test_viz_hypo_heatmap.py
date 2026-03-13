from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg', force=True)

import numpy as np
import pytest

import viz.hypo.synth_eval as hviz
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
		y_m=np.array([0, 1000, 2000], dtype=float),
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


def test_save_heatmap_2d_sets_title_and_colorbar_label(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	out_png = tmp_path / 'heatmap_labeled.png'
	data = np.arange(6, dtype=float).reshape((2, 3))
	captured: dict[str, object] = {}

	def _save_figure(fig, path: Path, *, dpi: int = 150) -> Path:
		_ = dpi
		captured['fig'] = fig
		path.write_text('stub', encoding='utf-8')
		return path

	monkeypatch.setattr(hviz, 'save_figure', _save_figure)

	hviz.save_heatmap_2d(
		data,
		out_png,
		title='Azimuthal GAP XY z=0 m',
		xlabel='X (km)',
		ylabel='Y (km)',
		extent=(0.0, 1.0, 0.0, 1.0),
		vmin=0.0,
		vmax=360.0,
		colorbar_label='GAP (deg)',
	)

	fig = captured['fig']
	assert fig.axes[0].get_title() == 'Azimuthal GAP XY z=0 m'
	assert fig.axes[1].get_ylabel() == 'GAP (deg)'


def test_save_heatmap_xy_slices_uses_display_name_and_colorbar_label(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	axes = _make_axes()
	grid = _make_grid()
	out_dir = tmp_path / 'xy_meta'
	captured: list[tuple[str, str | None]] = []

	def _save_heatmap_2d(
		data_2d: np.ndarray,
		out_png: Path,
		*,
		title: str,
		xlabel: str,
		ylabel: str,
		extent: tuple[float, float, float, float],
		vmin: float,
		vmax: float,
		colorbar_label: str | None = None,
		origin: str = 'lower',
		invert_y: bool = False,
	) -> Path:
		_ = (data_2d, xlabel, ylabel, extent, vmin, vmax, origin, invert_y)
		captured.append((title, colorbar_label))
		out_png.parent.mkdir(parents=True, exist_ok=True)
		out_png.write_text('stub', encoding='utf-8')
		return out_png

	monkeypatch.setattr(hviz, 'save_heatmap_2d', _save_heatmap_2d)

	paths = hviz.save_heatmap_xy_slices(
		grid,
		axes,
		out_dir,
		metric='GAP',
		display_name='Azimuthal GAP',
		colorbar_label='GAP (deg)',
		vmin=0.0,
		vmax=360.0,
	)

	assert len(paths) == len(axes.z_m)
	assert captured == [
		('Azimuthal GAP XY z=0 m', 'GAP (deg)'),
		('Azimuthal GAP XY z=1000 m', 'GAP (deg)'),
	]


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
	assert out_png.name == 'xz_y1000m.png'


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
