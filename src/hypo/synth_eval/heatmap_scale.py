from __future__ import annotations

import numpy as np


def compute_vmin_vmax(
	metric: str,
	grid_zyx: np.ndarray,
	*,
	percentile: float,
	dz_symmetric: bool,
) -> tuple[float, float]:
	"""Compute vmin/vmax (pXX; dz_m uses symmetric scale on |dz|)."""
	if not (0.0 < float(percentile) <= 100.0):
		raise ValueError('percentile must satisfy 0.0 < p <= 100.0')

	vals = np.asarray(grid_zyx, dtype=float)

	if metric == 'dz_m':
		if not dz_symmetric:
			raise ValueError('dz_symmetric must be True for dz_m')
		vmax = float(np.nanpercentile(np.abs(vals), percentile))
		return (-vmax, vmax)

	vmax = float(np.nanpercentile(vals, percentile))
	return (0.0, vmax)
