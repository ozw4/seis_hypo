# %%
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from jma.win32_reader import get_evt_info
from viz.gather import plot_gather
from waveform.filters import bandpass_iir_filtfilt

# =========================
# 設定（ここを直書きでOK）
# =========================


# 出力先（この中に zip が落ちる）
DIR = Path('/workspace/data/waveform/jma/event').resolve()

from jma.win32_reader import read_win32


def main() -> None:
	# get_event_waveform は「カレントディレクトリ」に落ちる想定なので移動
	os.chdir(DIR)
	files = {p.name for p in DIR.glob('*')}

	for zip_name in files:
		zpath = DIR / zip_name
		event_dir = DIR / zpath.stem
		event_dir.mkdir(exist_ok=True)

		evt_files = sorted(event_dir.glob('*.evt'))
		ch_files = sorted(event_dir.glob('*_active.ch'))
		if not evt_files or not ch_files:
			continue
		count = 0
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
			win32_data = read_win32(
				evt_path, channel_table=ch_path, duration_SECOND=info.span_seconds
			)
			win32_data = np.nan_to_num(win32_data, nan=0.0, posinf=0.0, neginf=0.0)
			active_ch = np.any(win32_data != 0, axis=1)
			print(active_ch.shape[0], 'total channels, of which')
			print(active_ch.sum(), 'active channels')
			active_data = win32_data[active_ch]
			active_data = bandpass_iir_filtfilt(
				active_data,
				fs=info.base_sampling_rate_hz,
				fstop_lo=1.0,
				fpass_lo=2.0,
				fpass_hi=25.0,
				fstop_hi=30.0,
				gpass=1.0,
				gstop=40.0,
			)
			plot_gather(active_data, amp=1.0, detrend='linear', taper_frac=0.05)
			print(win32_data.shape)
			count += 1
		if count > 20:
			break


if __name__ == '__main__':
	main()
