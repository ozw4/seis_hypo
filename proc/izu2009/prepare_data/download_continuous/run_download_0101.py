# %%
from __future__ import annotations

import datetime as dt
from pathlib import Path

from jma.continuous_period_download import (
	download_continuous_period,
	read_station_list_txt,
)
from jma.download import create_hinet_client

if __name__ == '__main__':
	# 入力（自分の環境に合わせて直書きで変更）
	station_list_path = Path(
		'/workspace/proc/prepare_data/izu2009/profile/stations47/stations_0101.txt'
	)
	out_dir = Path('/workspace/data/izu2009/continuous/0101')

	start_jst = dt.datetime(2009, 12, 17, 0, 0)
	end_jst = dt.datetime(2009, 12, 20, 23, 59)

	span_min = 10
	threads = 8

	stations = read_station_list_txt(station_list_path)

	client = create_hinet_client()

	download_continuous_period(
		client,
		network_code='0101',
		stations=stations,
		start_jst=start_jst,
		end_jst=end_jst,
		out_dir=out_dir,
		span_min=span_min,
		threads=threads,
		use_select=True,
		cleanup=True,
		skip_if_exists=True,
	)
