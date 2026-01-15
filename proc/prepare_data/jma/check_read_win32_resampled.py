# proc/prepare_data/jma/check_read_win32_resampled.py
# %%
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from jma.station_reader import read_hinet_channel_table
from jma.win32_reader import (
	get_evt_info,
	read_win32,
	read_win32_resampled,
	scan_channel_sampling_rate_map_win32,
)

# =========================
# 設定（ここを直書き）
# =========================
EVT_PATH = Path(
	'/workspace/data/waveform/jma/event/D20230211000408_20/D20230211000408_20.evt'
).resolve()

# active_ch が無いなら通常 .ch を指定
CH_PATH = Path(
	'/workspace/data/waveform/jma/event/D20230211000408_20/D20230211000408_20.ch'
).resolve()

TARGET_FS_HZ = 100

# None にすると get_evt_info().span_seconds を使う
DURATION_SECOND: int | None = None

# 旧 read_win32 を実行して落ちるか確認（base=info.base_sampling_rate_hz）
RUN_OLD_READ = False

# 表示する欠損 ch_int の最大数
PRINT_MISSING_MAX = 30


def main() -> None:
	if not EVT_PATH.is_file():
		raise FileNotFoundError(EVT_PATH)
	if not CH_PATH.is_file():
		raise FileNotFoundError(CH_PATH)

	info = get_evt_info(EVT_PATH, scan_rate_blocks=1000)
	sec = int(info.span_seconds) if DURATION_SECOND is None else int(DURATION_SECOND)

	print('==== EVT INFO ====')
	print(f'evt_path           : {EVT_PATH}')
	print(f'start_time         : {info.start_time}')
	print(f'end_time_exclusive : {info.end_time_exclusive}')
	print(f'span_seconds       : {info.span_seconds}')
	print(f'sampling_rates_hz  : {info.sampling_rates_hz}')
	print(f'base_sampling_rate : {info.base_sampling_rate_hz}')
	print(f'duration_SECOND    : {sec}')
	print(f'target_fs_hz       : {TARGET_FS_HZ}')
	print()

	station_df_all = read_hinet_channel_table(CH_PATH)
	if 'ch_int' not in station_df_all.columns:
		raise ValueError(f'channel table missing ch_int column: {CH_PATH}')

	fs_by_ch = scan_channel_sampling_rate_map_win32(EVT_PATH)
	present_ch = set(int(k) for k in fs_by_ch.keys())

	ch_ints_all = station_df_all['ch_int'].astype(int).tolist()
	missing_ch = sorted([ch for ch in ch_ints_all if ch not in present_ch])

	print('==== Channel presence check ====')
	print(f'.ch channels            : {len(ch_ints_all)}')
	print(f'WIN32-present channels  : {len(present_ch)}')
	print(f'missing in WIN32        : {len(missing_ch)}')
	if missing_ch:
		show = missing_ch[:PRINT_MISSING_MAX]
		show_s = ' '.join([f'{c}(0x{c:04X})' for c in show])
		print(f'missing examples (first {len(show)}): {show_s}')
	print()

	# WIN32に実在するチャンネルだけに絞る（active_chが無いケースの基本ムーブ）
	station_df = station_df_all[
		station_df_all['ch_int'].astype(int).isin(present_ch)
	].copy()
	if station_df.empty:
		raise ValueError('no channels in .ch are present in WIN32 (cannot proceed)')

	# fs分布（この時点でKeyErrorは起きない）
	ch_ints = station_df['ch_int'].astype(int).tolist()
	row_fs = [int(fs_by_ch[int(ch)]) for ch in ch_ints]
	cnt = Counter(row_fs)

	print('==== FS distribution over selected channels (WIN32-present) ====')
	for fs in sorted(cnt.keys()):
		print(f'fs={fs}: n_ch={cnt[fs]}')
	print()

	print('==== READ (resampled) ====')
	y = read_win32_resampled(
		EVT_PATH,
		station_df,
		target_sampling_rate_HZ=int(TARGET_FS_HZ),
		duration_SECOND=sec,
	)
	print(f'OK: y.shape={y.shape} dtype={y.dtype}')
	if np.isnan(y).any():
		raise ValueError('NaN detected in output array')

	out_path = EVT_PATH.with_name(f'{EVT_PATH.stem}_resampled_{TARGET_FS_HZ}Hz.npy')
	np.save(out_path, y)
	print(f'[saved] {out_path}')
	print()

	if RUN_OLD_READ:
		print('==== READ (old read_win32) ====')
		y0 = read_win32(
			EVT_PATH,
			station_df,
			base_sampling_rate_HZ=int(info.base_sampling_rate_hz),
			duration_SECOND=sec,
		)
		print(f'OLD OK (unexpected): y0.shape={y0.shape} dtype={y0.dtype}')


if __name__ == '__main__':
	main()
