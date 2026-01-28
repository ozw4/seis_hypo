"""Tests for common.time_util."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from common.time_util import (
	ceil_minutes,
	floor_minute,
	get_event_origin_utc,
	iso_to_ns,
	minute_range,
	month_label,
	parse_cfg_time_utc,
	to_utc,
)


def test_month_label() -> None:
	assert month_label(dt.date(2020, 1, 2)) == '2020-01'


def test_iso_to_ns_matches_numpy() -> None:
	iso = '2020-01-01T00:00:00+00:00'
	ns = iso_to_ns(iso)
	expected = int(
		np.datetime64(pd.Timestamp(iso).to_datetime64())
		.astype('datetime64[ns]')
		.astype('int64')
	)
	assert ns == expected


def test_floor_minute() -> None:
	t = dt.datetime(2020, 1, 1, 12, 34, 56, 789)
	out = floor_minute(t)
	assert out == dt.datetime(2020, 1, 1, 12, 34)


def test_ceil_minutes() -> None:
	assert ceil_minutes(1.0) == 1
	assert ceil_minutes(60.0) == 1
	assert ceil_minutes(60.1) == 2
	with pytest.raises(ValueError):
		ceil_minutes(0.0)


def test_minute_range_inclusive() -> None:
	start = dt.datetime(2020, 1, 1, 0, 0, 30)
	end = dt.datetime(2020, 1, 1, 0, 2, 1)
	out = list(minute_range(start, end))
	assert out == [
		dt.datetime(2020, 1, 1, 0, 0),
		dt.datetime(2020, 1, 1, 0, 1),
		dt.datetime(2020, 1, 1, 0, 2),
	]


def test_to_utc_naive_treated_as_jst_by_default() -> None:
	ts = pd.Timestamp('2020-01-01T00:00:00')
	out = to_utc(ts)
	assert str(out) == '2019-12-31 15:00:00+00:00'


def test_get_event_origin_utc_prefers_origin_time_jst(tmp_path: Path) -> None:
	ev = {'origin_time_jst': '2020-01-01T00:00:00'}
	p = tmp_path / 'event.json'
	out = get_event_origin_utc(ev, p)
	assert str(out) == '2019-12-31 15:00:00+00:00'


def test_get_event_origin_utc_origin_time_naive_treated_as_utc(tmp_path: Path) -> None:
	ev = {'origin_time': '2020-01-01T00:00:00'}
	p = tmp_path / 'event.json'
	out = get_event_origin_utc(ev, p)
	assert str(out) == '2020-01-01 00:00:00+00:00'


def test_parse_cfg_time_utc() -> None:
	assert parse_cfg_time_utc(None) is None
	assert str(parse_cfg_time_utc('2020-01-01T00:00:00')) == '2019-12-31 15:00:00+00:00'
	assert (
		str(parse_cfg_time_utc('2020-01-01T00:00:00+00:00'))
		== '2020-01-01 00:00:00+00:00'
	)
