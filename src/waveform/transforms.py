# src/waveform/transforms.py
from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.signal as sp_signal


def _as_1d_float(x: np.ndarray, *, name: str = 'x') -> np.ndarray:
	a = np.asarray(x, dtype=float)
	if a.ndim != 1:
		raise ValueError(f'{name} must be 1D, got shape={a.shape}')
	if a.size == 0:
		raise ValueError(f'{name} must be non-empty')
	return a


def abs_1d(x: np.ndarray) -> np.ndarray:
	"""|x|（1D）"""
	a = _as_1d_float(x)
	return np.abs(a)


def square_1d(x: np.ndarray) -> np.ndarray:
	"""x^2（1D）"""
	a = _as_1d_float(x)
	return a * a


def energy_1d(
	x: np.ndarray, *, kind: Literal['square', 'abs'] = 'square'
) -> np.ndarray:
	"""エネルギー系変換（1D）。kind='square' は x^2、kind='abs' は |x|。"""
	a = _as_1d_float(x)
	if kind == 'square':
		return a * a
	if kind == 'abs':
		return np.abs(a)
	raise ValueError(f'Unsupported kind={kind!r}')


def analytic_signal_1d(x: np.ndarray) -> np.ndarray:
	"""解析信号 z = x + iH{x}（1D）"""
	a = _as_1d_float(x)
	return sp_signal.hilbert(a)


def envelope_1d(x: np.ndarray) -> np.ndarray:
	"""包絡（Hilbert）: |x + iH{x}|（1D）"""
	z = analytic_signal_1d(x)
	return np.abs(z)


__all__ = [
	'abs_1d',
	'analytic_signal_1d',
	'energy_1d',
	'envelope_1d',
	'square_1d',
]
