# %%
from __future__ import annotations

from pathlib import Path

from jma.prepare.station_subset_ch import (
	read_station_list_txt,
	write_station_subset_ch_dir,
)

if __name__ == '__main__':
	# ====== 直書き設定 ======
	base_cont_dir = Path('../../../data/izu2009/continuous')  # DL済み .ch がある場所
	stations_dir = Path('./profile/stations47')  # stations_0101.txt 等がある場所
	out_base_dir = Path('./download_continuous/continuous_ch47')  # 出力先（新規）

	networks = ['0101', '0203', '0207', '0301']
	# =======================

	for net in networks:
		in_dir = base_cont_dir / net
		out_dir = out_base_dir / net
		sta_txt = stations_dir / f'stations_{net}.txt'

		keep_stations = read_station_list_txt(sta_txt)

		write_station_subset_ch_dir(
			in_dir=in_dir,
			out_dir=out_dir,
			keep_stations=keep_stations,
			pattern='*.ch',
			skip_if_exists=True,
		)
