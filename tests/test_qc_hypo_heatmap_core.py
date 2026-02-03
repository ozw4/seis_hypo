from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qc.hypo.heatmap import (
	build_metric_grid_zyx,
	build_metric_grids_zyx,
	compute_vmin_vmax,
	load_grid_axes_from_index_csv,
	map_true_xyz_to_zyx_indices,
	write_axes_json,
	write_metric_grid_npy,
)


def _write_index_csv(tmp_path: Path) -> Path:
	df = pd.DataFrame(
		{
			'x_m': [10, 0, 10],
			'y_m': [5, 5, 0],
			'z_m': [100, 0, 100],
		}
	)
	path = tmp_path / 'index.csv'
	df.to_csv(path, index=False)
	return path


def test_load_grid_axes_unique_sorted(tmp_path: Path) -> None:
	index_csv = _write_index_csv(tmp_path)
	axes = load_grid_axes_from_index_csv(index_csv)

	assert np.array_equal(axes.x_m, np.array([0, 10]))
	assert np.array_equal(axes.y_m, np.array([0, 5]))
	assert np.array_equal(axes.z_m, np.array([0, 100]))
	assert axes.shape_zyx() == (2, 2, 2)


def test_map_true_xyz_to_zyx_indices_missing_coord_raises(tmp_path: Path) -> None:
	index_csv = _write_index_csv(tmp_path)
	axes = load_grid_axes_from_index_csv(index_csv)

	df_eval = pd.DataFrame(
		{
			'x_m_true': [0, 999],
			'y_m_true': [0, 0],
			'z_m_true': [0, 0],
		}
	)

	with pytest.raises(ValueError):
		map_true_xyz_to_zyx_indices(df_eval, axes)


def test_build_metric_grid_zyx_nanmedian() -> None:
	indices = np.array(
		[
			[0, 0, 0],
			[0, 0, 0],
			[0, 0, 0],
			[1, 1, 1],
		],
		dtype=int,
	)
	values = np.array([1.0, 100.0, np.nan, 5.0], dtype=float)
	grid = build_metric_grid_zyx(indices, values, (2, 2, 2))

	assert grid.shape == (2, 2, 2)
	assert np.isclose(grid[0, 0, 0], 50.5)
	assert np.isclose(grid[1, 1, 1], 5.0)
	assert np.isnan(grid[0, 1, 1])


def test_build_metric_grids_zyx_missing_metric_raises(tmp_path: Path) -> None:
	index_csv = _write_index_csv(tmp_path)
	axes = load_grid_axes_from_index_csv(index_csv)

	df_eval = pd.DataFrame(
		{
			'x_m_true': [0],
			'y_m_true': [0],
			'z_m_true': [0],
			'err3d_m': [1.0],
		}
	)

	with pytest.raises(ValueError):
		build_metric_grids_zyx(df_eval, axes, ['missing'])


def test_compute_vmin_vmax_err3d_and_dz() -> None:
	grid = np.array([[[0.0, 1.0], [2.0, 3.0]]], dtype=float)
	percentile = 99.0
	vmin, vmax = compute_vmin_vmax(
		'err3d_m', grid, percentile=percentile, dz_symmetric=True
	)
	assert vmin == 0.0
	assert np.isclose(vmax, np.nanpercentile(grid, percentile))

	grid_dz = np.array([[[-2.0, 0.0, 2.0, 4.0]]], dtype=float)
	vmin_dz, vmax_dz = compute_vmin_vmax(
		'dz_m',
		grid_dz,
		percentile=percentile,
		dz_symmetric=True,
	)
	expected = np.nanpercentile(np.abs(grid_dz), percentile)
	assert np.isclose(vmax_dz, expected)
	assert np.isclose(vmin_dz, -expected)


def test_write_axes_json_and_metric_grid_npy(tmp_path: Path) -> None:
	index_csv = _write_index_csv(tmp_path)
	axes = load_grid_axes_from_index_csv(index_csv)

	out_json = tmp_path / 'axes.json'
	out_npy = tmp_path / 'grid.npy'

	grid = np.arange(8, dtype=float).reshape((2, 2, 2))

	write_axes_json(out_json, axes)
	write_metric_grid_npy(out_npy, grid)

	assert out_json.is_file()
	assert out_npy.is_file()

	loaded = json.loads(out_json.read_text(encoding='utf-8'))
	for key in ['x_m', 'y_m', 'z_m', 'shape', 'order', 'center_x_m', 'center_y_m']:
		assert key in loaded

	grid_loaded = np.load(out_npy)
	assert np.array_equal(grid_loaded, grid)
