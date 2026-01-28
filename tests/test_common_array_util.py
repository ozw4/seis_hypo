"""Tests for common.array_util."""

from __future__ import annotations

import numpy as np
import pytest

from common.array_util import as_1d_float


def test_as_1d_float_ok() -> None:
	out = as_1d_float([1, 2, 3])
	assert out.dtype == float
	assert out.ndim == 1
	assert np.allclose(out, np.array([1.0, 2.0, 3.0]))


def test_as_1d_float_rejects_empty() -> None:
	with pytest.raises(ValueError):
		as_1d_float([])


def test_as_1d_float_rejects_ndim_ne1() -> None:
	with pytest.raises(ValueError):
		as_1d_float([[1, 2], [3, 4]])
