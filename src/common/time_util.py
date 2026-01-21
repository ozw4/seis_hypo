from __future__ import annotations

import datetime as dt
import math
from collections.abc import Iterator
from datetime import timezone
from pathlib import Path

import pandas as pd


def floor_minute(t: dt.datetime) -> dt.datetime:
	"""秒以下を切り捨てて分単位にそろえる。"""
	return t.replace(second=0, microsecond=0)


def ceil_minutes(delta_seconds: float) -> int:
	if delta_seconds <= 0:
		raise ValueError(f'invalid delta_seconds={delta_seconds}')
	return int(math.ceil(delta_seconds / 60.0))


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


def get_event_origin_utc(ev: dict, event_json_path: Path) -> pd.Timestamp:
	"""event.json の origin_time を UTC-aware Timestamp に正規化する。

	優先順位:
	- origin_time_jst があればそれを優先
	- なければ origin_time を使う

	naive の扱い:
	- origin_time_jst: JST として扱う
	- origin_time: UTC として扱う
	"""
	origin_jst = ev.get('origin_time_jst')
	origin_other = ev.get('origin_time')
	origin_raw = origin_jst if origin_jst is not None else origin_other
	if origin_raw is None:
		raise ValueError(f'missing origin_time(_jst) in {event_json_path}')

	origin = pd.to_datetime(origin_raw)
	if pd.isna(origin):
		raise ValueError(f'failed to parse origin_time in {event_json_path}')

	naive_tz = 'Asia/Tokyo' if origin_jst is not None else 'UTC'
	return to_utc(origin, naive_tz=naive_tz)


_JST_UTC_OFFSET_HOURS = 9


def parse_cfg_time_utc(raw: str | None) -> pd.Timestamp | None:
	"""Config 時刻文字列を UTC-aware Timestamp に正規化する。

	- raw が None の場合は None
	- timezone 省略時は JST として扱う
	"""
	if raw is None:
		return None
	ts = pd.to_datetime(raw)
	if pd.isna(ts):
		raise ValueError(f'failed to parse time: {raw}')
	# Config times are treated as JST if timezone is omitted.
	return to_utc(ts, naive_tz='Asia/Tokyo')
