import datetime as dt
from collections.abc import Iterator


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


_JST_UTC_OFFSET_HOURS = 9
