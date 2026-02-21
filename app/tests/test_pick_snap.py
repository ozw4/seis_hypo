"""Unit tests for pick snap utilities."""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.utils.pick_snap import (
	parabolic_refine,
	snap_pick_index,
	snap_pick_time_s,
	zero_cross_refine,
)


def _assert_close(actual: float, expected: float, *, atol: float = 1.0e-8) -> None:
	if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=atol):
		raise AssertionError(f'actual={actual}, expected={expected}')


def _assert_equal(actual: float, expected: float) -> None:
	if actual != expected:
		raise AssertionError(f'actual={actual}, expected={expected}')


def test_parabolic_refine_returns_expected_fractional_index() -> None:
	"""Parabolic refinement returns stable symmetric/asymmetric sub-sample indices."""
	arr_sym = np.array([0.0, 1.0, 4.0, 1.0, 0.0], dtype=float)
	_assert_close(parabolic_refine(arr_sym, 2), 2.0)

	arr_asym = np.array([0.0, 1.0, 4.0, 3.0, 0.0], dtype=float)
	_assert_close(parabolic_refine(arr_asym, 2), 2.25)


def test_zero_cross_refine_returns_linear_interpolated_index() -> None:
	"""Zero-cross refinement returns the linear interpolation location."""
	arr = np.array([-1.0, 1.0], dtype=float)
	_assert_close(zero_cross_refine(arr, 0), 0.5)


def test_snap_pick_index_peak_prefers_nearest_local_maximum() -> None:
	"""Peak mode snaps to the nearest local maximum in window."""
	trace = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0], dtype=float)
	out = snap_pick_index(
		trace,
		4.6,
		mode='peak',
		refine='none',
		window_samples=3,
	)
	_assert_equal(out, 5.0)


def test_snap_pick_index_peak_falls_back_to_window_maximum() -> None:
	"""Peak mode falls back to the maximum in window when no local maxima exist."""
	trace = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
	out = snap_pick_index(
		trace,
		2.0,
		mode='peak',
		refine='none',
		window_samples=3,
	)
	_assert_equal(out, 3.0)


def test_snap_pick_index_trough_prefers_nearest_local_minimum() -> None:
	"""Trough mode snaps to the nearest local minimum in window."""
	trace = np.array([3.0, 1.0, 2.0, 0.0, 2.0, 1.0, 3.0], dtype=float)
	out = snap_pick_index(
		trace,
		1.2,
		mode='trough',
		refine='none',
		window_samples=3,
	)
	_assert_equal(out, 1.0)


def test_snap_pick_index_trough_falls_back_to_window_minimum() -> None:
	"""Trough mode falls back to the minimum in window when no local minima exist."""
	trace = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
	out = snap_pick_index(
		trace,
		2.0,
		mode='trough',
		refine='none',
		window_samples=3,
	)
	_assert_equal(out, 1.0)


def test_snap_pick_index_rise_prefers_nearest_upgoing_zero_cross() -> None:
	"""Rise mode snaps to the nearest up-going zero-cross candidate."""
	trace = np.array([-2.0, -1.0, 1.0, 2.0, -1.0, 1.0], dtype=float)
	out = snap_pick_index(
		trace,
		2.2,
		mode='rise',
		refine='none',
		window_samples=3,
	)
	_assert_equal(out, 2.0)


def test_snap_pick_index_rise_falls_back_to_max_positive_gradient() -> None:
	"""Rise mode falls back to the index with maximum positive central gradient."""
	trace = np.array([5.0, 4.0, 3.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
	out = snap_pick_index(
		trace,
		2.0,
		mode='rise',
		refine='none',
		window_samples=3,
	)
	_assert_equal(out, 4.0)


def test_snap_pick_time_s_uses_dt_and_window_ms() -> None:
	"""Time-based wrapper converts dt/window_ms and returns snapped time."""
	trace = np.array([0.0, 1.0, 0.0, 2.0, 0.0], dtype=float)
	out = snap_pick_time_s(
		trace,
		0.26,
		dt=0.1,
		mode='peak',
		refine='none',
		window_ms=120.0,
	)
	_assert_close(out, 0.3)


def test_invalid_dt_raises_value_error() -> None:
	"""Non-positive dt is rejected."""
	trace = np.array([0.0, 1.0, 0.0], dtype=float)
	with pytest.raises(ValueError, match='dt must be positive'):
		snap_pick_time_s(
			trace,
			0.1,
			dt=0.0,
			mode='peak',
			refine='none',
			window_ms=100.0,
		)


def test_invalid_mode_raises_value_error() -> None:
	"""Unknown mode is rejected."""
	trace = np.array([0.0, 1.0, 0.0], dtype=float)
	with pytest.raises(ValueError, match='unsupported mode'):
		snap_pick_index(
			trace,
			1.0,
			mode='unknown',
			refine='none',
			window_samples=1,
		)
