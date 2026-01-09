from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from datetime import timezone

import pandas as pd


def floor_minute(t: dt.datetime) -> dt.datetime:
	"""秒以下を切り捨てて分単位にそろえる。"""
	return t.replace(second=0, microsecond=0)


def minute_range(start: dt.datetime, end: dt.datetime) -> Iterator[dt.datetime]:
	"""start〜end を覆う分頭の時刻を列挙（両端含む）。"""
	cur = floor_minute(start)
	last = floor_minute(end)
	while cur <= last:
		yield cur
		cur += dt.timedelta(minutes=1)


def to_utc(ts: pd.Timestamp, *, naive_tz: str = 'Asia/Tokyo') -> pd.Timestamp:
	"""Timestamp を UTC に正規化する。

	- tz-aware: UTC へ変換
	- tz-naive: naive_tz を付与してから UTC へ変換
	"""
	ts = pd.Timestamp(ts)
	if pd.isna(ts):
		raise ValueError('timestamp is NaT')
	if ts.tzinfo is None:
		ts = ts.tz_localize(naive_tz)
	return ts.tz_convert('UTC')


def utc_ms_to_iso(ms: int) -> str:
	return dt.datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc).isoformat()


def as_utc_aware(d: dt.datetime) -> dt.datetime:
	if d.tzinfo is None:
		return d.replace(tzinfo=timezone.utc)
	return d.astimezone(timezone.utc)


def origin_to_utc(origin: object) -> pd.Timestamp:
	"""Parse origin time treating naive as JST, return UTC-aware Timestamp."""
	ts = pd.to_datetime(origin)
	if ts.tzinfo is None:
		ts = ts.tz_localize('Asia/Tokyo')
	return ts.tz_convert('UTC')


_JST_UTC_OFFSET_HOURS = 9
