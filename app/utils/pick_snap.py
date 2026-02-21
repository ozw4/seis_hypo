"""Pick snapping utilities ported from the legacy waveform viewer logic."""

from __future__ import annotations

import numpy as np

_EPS = 1.0e-12
_MODE_SET = frozenset({'none', 'peak', 'trough', 'rise'})
_REFINE_SET = frozenset({'none', 'parabolic', 'zc'})


def _as_1d_trace(trace: np.ndarray) -> np.ndarray:
	arr = np.asarray(trace, dtype=float)
	if arr.ndim != 1:
		raise ValueError('trace must be a 1D array')
	return arr


def _js_round(x: float) -> int:
	if not np.isfinite(x):
		raise ValueError('round input must be finite')
	if x >= 0.0:
		return int(np.floor(x + 0.5))
	return int(np.ceil(x - 0.5))


def _clamp_float(x: float, lo: float, hi: float) -> float:
	if x < lo:
		return lo
	if x > hi:
		return hi
	return x


def _snap_peak_index(arr: np.ndarray, i0: int, lo: int, hi: int) -> int:
	best_idx = -1
	best_dist = float('inf')
	for i in range(lo, hi + 1):
		if arr[i] >= arr[i - 1] and arr[i] >= arr[i + 1]:
			dist = abs(i - i0)
			if dist < best_dist:
				best_dist = float(dist)
				best_idx = i
	if best_idx >= 0:
		return best_idx

	fallback_idx = lo
	for i in range(lo + 1, hi + 1):
		if arr[i] > arr[fallback_idx]:
			fallback_idx = i
	return fallback_idx


def _snap_trough_index(arr: np.ndarray, i0: int, lo: int, hi: int) -> int:
	best_idx = -1
	best_dist = float('inf')
	for i in range(lo, hi + 1):
		if arr[i] <= arr[i - 1] and arr[i] <= arr[i + 1]:
			dist = abs(i - i0)
			if dist < best_dist:
				best_dist = float(dist)
				best_idx = i
	if best_idx >= 0:
		return best_idx

	fallback_idx = lo
	for i in range(lo + 1, hi + 1):
		if arr[i] < arr[fallback_idx]:
			fallback_idx = i
	return fallback_idx


def _snap_rise_index(arr: np.ndarray, i0: int, lo: int, hi: int) -> int:
	best_idx = -1
	best_dist = float('inf')
	for i in range(lo, hi):
		if arr[i] <= 0.0 and arr[i + 1] > 0.0:
			cand = i if abs(arr[i]) < abs(arr[i + 1]) else i + 1
			dist = abs(cand - i0)
			if dist < best_dist:
				best_dist = float(dist)
				best_idx = cand
	if best_idx >= 0:
		return best_idx

	best_slope = -float('inf')
	best_grad_idx = -1
	for i in range(lo, hi + 1):
		slope = arr[i + 1] - arr[i - 1]
		if slope > 0.0 and slope > best_slope:
			best_slope = float(slope)
			best_grad_idx = i
	if best_grad_idx >= 0:
		return best_grad_idx

	return i0


def _snap_core_index(arr: np.ndarray, i0: int, lo: int, hi: int, mode: str) -> float:
	if mode == 'peak':
		return float(_snap_peak_index(arr, i0, lo, hi))
	if mode == 'trough':
		return float(_snap_trough_index(arr, i0, lo, hi))
	if mode == 'rise':
		return float(_snap_rise_index(arr, i0, lo, hi))
	return _clamp_float(float(i0), 0.0, float(arr.size - 1))


def _apply_refine(arr: np.ndarray, idx: float, mode: str, refine: str) -> float:
	if mode in {'peak', 'trough'} and refine == 'parabolic':
		return parabolic_refine(arr, int(idx))
	if mode == 'rise' and refine == 'zc':
		return zero_cross_refine(arr, int(idx))
	return idx


def parabolic_refine(arr: np.ndarray, i: int) -> float:
	"""Refine a peak/trough index by a 3-point parabolic fit."""
	trace = _as_1d_trace(arr)
	n = trace.size
	ii = int(i)
	if n < 3:
		return float(ii)

	ii = min(max(ii, 1), n - 2)
	ym1 = float(trace[ii - 1])
	y0 = float(trace[ii])
	yp1 = float(trace[ii + 1])

	denom = ym1 - (2.0 * y0) + yp1
	if not np.isfinite(denom) or abs(denom) < _EPS:
		return float(ii)

	delta = 0.5 * (ym1 - yp1) / denom
	if not np.isfinite(delta) or abs(delta) > 0.6:
		return float(ii)

	xhat = float(ii) + float(delta)
	if not np.isfinite(xhat):
		return float(ii)
	return _clamp_float(xhat, 0.0, float(n - 1))


def zero_cross_refine(arr: np.ndarray, i: int) -> float:
	"""Refine an index to a nearby up-going zero-cross by linear interpolation."""
	trace = _as_1d_trace(arr)
	n = trace.size
	if n < 2:
		return float(i)

	i0 = min(max(int(i), 0), n - 2)
	i1 = i0 + 1
	left0 = i0 - 1
	right1 = i1 + 1

	x0 = -1
	x1 = -1
	if trace[i0] <= 0.0 and trace[i1] > 0.0:
		x0, x1 = i0, i1
	elif left0 >= 0 and trace[left0] <= 0.0 and trace[i0] > 0.0:
		x0, x1 = left0, i0
	elif right1 < n and trace[i1] <= 0.0 and trace[right1] > 0.0:
		x0, x1 = i1, right1
	else:
		return float(i)

	dy = float(trace[x1] - trace[x0])
	if not np.isfinite(dy) or abs(dy) < _EPS:
		return float(i)

	xhat = float(x0) + ((0.0 - float(trace[x0])) / dy)
	if not np.isfinite(xhat):
		return float(i)
	return _clamp_float(xhat, 0.0, float(n - 1))


def snap_pick_index(
	trace: np.ndarray,
	idx0: float,
	*,
	mode: str,
	refine: str,
	window_samples: int,
) -> float:
	"""Snap a floating sample index to a feature in a local window."""
	arr = _as_1d_trace(trace)
	if mode not in _MODE_SET:
		raise ValueError(f'unsupported mode: {mode}')
	if refine not in _REFINE_SET:
		raise ValueError(f'unsupported refine: {refine}')

	idx0f = float(idx0)
	if not np.isfinite(idx0f):
		raise ValueError('idx0 must be finite')

	n = arr.size
	if n < 3:
		return idx0f

	i0 = _js_round(idx0f)
	rad = max(1, int(window_samples))
	lo = max(1, i0 - rad)
	hi = min(n - 2, i0 + rad)
	if lo > hi:
		return idx0f

	idx = _snap_core_index(arr, i0, lo, hi, mode)
	idx = _apply_refine(arr, idx, mode, refine)

	return _clamp_float(idx, 0.0, float(n - 1))


def snap_pick_time_s(  # noqa: PLR0913
	trace: np.ndarray,
	time_s: float,
	*,
	dt: float,
	mode: str,
	refine: str,
	window_ms: float,
) -> float:
	"""Snap a pick time (seconds) to a local feature and return snapped seconds."""
	dtf = float(dt)
	if not np.isfinite(dtf) or dtf <= 0.0:
		raise ValueError('dt must be positive')

	time_sf = float(time_s)
	if not np.isfinite(time_sf):
		raise ValueError('time_s must be finite')

	window_msf = float(window_ms)
	if not np.isfinite(window_msf):
		raise ValueError('window_ms must be finite')

	idx0 = float(_js_round(time_sf / dtf))
	window_samples = _js_round((window_msf / 1000.0) / dtf)
	idx = snap_pick_index(
		trace,
		idx0,
		mode=mode,
		refine=refine,
		window_samples=window_samples,
	)
	return float(idx) * dtf
