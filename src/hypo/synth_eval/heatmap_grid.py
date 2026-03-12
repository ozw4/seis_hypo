from __future__ import annotations

import numpy as np

from hypo.synth_eval.heatmap_types import GridAxes


def axis_extent_km_from_centers(axis_m: np.ndarray) -> tuple[float, float]:
	axis = np.asarray(axis_m, dtype=float).reshape(-1)
	if axis.size == 0:
		raise ValueError('axis is empty')
	if axis.size < 2:
		raise ValueError('axis must have at least 2 points')
	d = np.diff(axis)
	if not np.all(d > 0):
		raise ValueError('axis must be strictly increasing')
	if not np.allclose(d, d[0]):
		raise ValueError('axis must be evenly spaced')
	dx = float(d[0])
	xmin = float(np.min(axis)) - dx * 0.5
	xmax = float(np.max(axis)) + dx * 0.5
	return (xmin / 1000.0, xmax / 1000.0)


def validate_heatmap_grid_shape(grid_zyx: np.ndarray, axes: GridAxes) -> None:
	if grid_zyx.ndim != 3:
		raise ValueError(f'grid_zyx must be 3D, got shape={grid_zyx.shape}')
	shape = axes.shape_zyx()
	if grid_zyx.shape != shape:
		raise ValueError(f'grid_zyx shape mismatch: {grid_zyx.shape} vs {shape}')
