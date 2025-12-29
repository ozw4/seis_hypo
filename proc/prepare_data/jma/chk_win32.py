# %%
from __future__ import annotations

import os
from pathlib import Path

from jma.win32_reader import get_evt_info

# =========================
# 設定（ここを直書きでOK）
# =========================


# 出力先（この中に zip が落ちる）
DIR = Path('/workspace/data/waveform/jma/event').resolve()


def main() -> None:
	# get_event_waveform は「カレントディレクトリ」に落ちる想定なので移動
	os.chdir(DIR)

	files = {p.name for p in DIR.glob('*')}

	for zip_name in files:
		zpath = DIR / zip_name
		event_dir = DIR / zpath.stem
		event_dir.mkdir(exist_ok=True)

		evt_files = sorted(event_dir.glob('*.evt'))
		ch_files = sorted(event_dir.glob('*.ch'))
		if not evt_files or not ch_files:
			continue

		for evt_path, ch_path in zip(evt_files, ch_files):
			info = get_evt_info(evt_path, scan_rate_blocks=1000)

			print(f'start_time          : {info.start_time}')
			print(f'end_time_exclusive  : {info.end_time_exclusive}')
			print(f'n_second_blocks     : {info.n_second_blocks}')
			print(f'span_seconds        : {info.span_seconds}')
			print(f'missing_seconds_est : {info.missing_seconds_est}')
			print(f'timestamp_anomalies : {info.timestamp_anomalies}')
			print(f'sampling_rates_hz   : {info.sampling_rates_hz}')
			print(f'base_sampling_rate  : {info.base_sampling_rate_hz}')
			print(f'Processing event file: {evt_path.name}')
			# win32_data = read_win32(evt_path, channel_table=ch_path)
			# print(win32_data.shape)


if __name__ == '__main__':
	main()
