# src/waveform/transforms.py
from __future__ import annotations

import numpy as np
import scipy.signal as sp_signal

from common.array_util import as_1d_float


def abs_1d(x: np.ndarray) -> np.ndarray:
	"""|x|（1D）"""
	a = as_1d_float(x)
	return np.abs(a)


def analytic_signal_1d(x: np.ndarray) -> np.ndarray:
	"""解析信号 z = x + iH{x}（1D）"""
	a = as_1d_float(x)
	return sp_signal.hilbert(a)


def envelope_1d(x: np.ndarray) -> np.ndarray:
	"""包絡（Hilbert）: |x + iH{x}|（1D）"""
	z = analytic_signal_1d(x)
	return np.abs(z)


__all__ = [
	'abs_1d',
	'analytic_signal_1d',
	'envelope_1d',
]
