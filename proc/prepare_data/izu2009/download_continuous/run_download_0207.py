# %%
# proc/prepare_data/jma/izu2009/run_download_0207.py
from __future__ import annotations

import datetime as dt
from pathlib import Path

from common.time_util import floor_minute
from jma.download import (
	_name_stem,
	_supports_station_selection,
	create_hinet_client,
	download_win_for_stations,
)

if __name__ == '__main__':
	# ========= ユーザー設定（直書き） =========
	network_code = '0207'

	# これは「意図の記録」と「stationsが空で落ちるのを避ける」ために渡す。
	# 0207はselect非対応なので、このリストでサーバ側が絞られることは期待しない（=全局DL）。
	stations_txt = Path('../profile/stations47/stations_0207.txt')

	# 出力先（0207だけ別ディレクトリに）
	out_dir = Path('/workspace/data/izu2009/continuous/0207')

	# 対象期間（JST扱い、tz-naive）
	# endは「排他的」にしておくと分かりやすい（最後は end - span_min まで）
	start_jst = dt.datetime(2009, 12, 17, 0, 0)
	end_exclusive_jst = dt.datetime(2009, 12, 21, 0, 0)

	# HinetPyの制限を考えると、0207はまず5分推奨
	span_min = 10
	threads = 8  # まずは安定寄り（必要なら8へ）
	cleanup = True
	skip_if_exists = True
	# =======================================

	if not stations_txt.is_file():
		raise FileNotFoundError(stations_txt)

	stations = [
		ln.strip()
		for ln in stations_txt.read_text(encoding='utf-8').splitlines()
		if ln.strip() and not ln.strip().startswith('#')
	]
	if not stations:
		raise ValueError(f'empty station list: {stations_txt}')

	select_supported = _supports_station_selection(network_code)
	if select_supported:
		# 0207では通常ここに来ない想定
		stations_for_name = stations
	else:
		# select非対応ネットは ALL を明示して安定名にする（repoの他スクリプトと同じ流儀）
		stations_for_name = ['ALL']

	out_dir.mkdir(parents=True, exist_ok=True)

	client = create_hinet_client()

	t0 = floor_minute(start_jst)
	t_end = floor_minute(end_exclusive_jst)

	step = dt.timedelta(minutes=int(span_min))
	cur = t0
	while cur < t_end:
		stem = _name_stem(network_code, cur, stations_for_name, span_min)
		data_name = f'{stem}.cnt'
		ctable_name = f'{stem}.ch'

		download_win_for_stations(
			client,
			stations=stations,
			when=cur,
			network_code=network_code,
			span_min=span_min,
			outdir=out_dir,
			threads=threads,
			cleanup=cleanup,
			clear_selection=False,
			skip_if_exists=skip_if_exists,
			use_select=select_supported,  # 0207はFalse想定
			data_name=data_name,
			ctable_name=ctable_name,
		)

		cur = cur + step
