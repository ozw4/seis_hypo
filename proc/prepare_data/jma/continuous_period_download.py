from __future__ import annotations

import datetime as dt
from pathlib import Path

from HinetPy import Client

from common.time_util import floor_minute
from jma.download import download_win_for_stations


def read_station_list_txt(path: str | Path) -> list[str]:
	p = Path(path)
	lines = p.read_text(encoding='utf-8').splitlines()
	stations = [
		ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith('#')
	]
	if not stations:
		raise ValueError(f'empty station list: {p}')
	return stations


def iter_span_starts(
	start_jst: dt.datetime, end_jst: dt.datetime, *, span_min: int
) -> list[dt.datetime]:
	if start_jst.tzinfo is not None or end_jst.tzinfo is not None:
		raise ValueError('use tz-naive datetimes (treated as JST)')

	if span_min <= 0:
		raise ValueError(f'invalid span_min={span_min}')

	t0 = floor_minute(start_jst)
	t1 = floor_minute(end_jst)
	if t0 > t1:
		raise ValueError('start_jst > end_jst')

	cur = t0
	out: list[dt.datetime] = []
	step = dt.timedelta(minutes=int(span_min))
	while cur <= t1:
		out.append(cur)
		cur = cur + step
	return out


def download_continuous_period(
	client: Client,
	*,
	network_code: str,
	stations: list[str],
	start_jst: dt.datetime,
	end_jst: dt.datetime,
	out_dir: str | Path,
	span_min: int,
	threads: int,
	use_select: bool,
	cleanup: bool = True,
	skip_if_exists: bool = True,
) -> None:
	"""start_jst〜end_jst（tz-naive, JST扱い）を span_min 分刻みで WIN32 を保存する。

	use_select:
	- True : select_stations を使う（0101など）
	- False: select_stations を使わない（0203/0207/0301はこの運用）
	"""
	code = str(network_code).strip()
	if not code:
		raise ValueError('network_code is empty')
	if not stations:
		raise ValueError('stations is empty')

	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	when_list = iter_span_starts(start_jst, end_jst, span_min=int(span_min))

	for when in when_list:
		download_win_for_stations(
			client,
			stations=stations,
			when=when,
			network_code=code,
			span_min=int(span_min),
			outdir=out_dir,
			threads=int(threads),
			cleanup=bool(cleanup),
			clear_selection=False,
			skip_if_exists=bool(skip_if_exists),
			use_select=bool(use_select),
		)

	# select を使った場合は解除（レポの download_win_for_event_multi_network と同じ流儀）
	if use_select:
		client.select_stations(code)
