# src/common/array_util.py
from __future__ import annotations

import numpy as np


def as_1d_float(x: object, *, name: str = 'x') -> np.ndarray:
	a = np.asarray(x, dtype=float)
	if a.ndim != 1:
		raise ValueError(f'{name} must be 1D, got shape={a.shape}')
	if a.size == 0:
		raise ValueError(f'{name} must be non-empty')
	return a


__all__ = ['as_1d_float']
