"""Tests for common.stride."""

from __future__ import annotations

import pytest

from common.stride import normalize_channel_stride


def test_normalize_channel_stride_none() -> None:
	assert normalize_channel_stride(None) is None


def test_normalize_channel_stride_zero_or_negative() -> None:
	with pytest.raises(ValueError):
		normalize_channel_stride(0)
	with pytest.raises(ValueError):
		normalize_channel_stride(-1)


def test_normalize_channel_stride_one_or_less_is_none() -> None:
	assert normalize_channel_stride(1) is None
	assert normalize_channel_stride(1.0) is None


def test_normalize_channel_stride_gt1_returns_int() -> None:
	assert normalize_channel_stride(2) == 2
	assert normalize_channel_stride(3.2) == 3
