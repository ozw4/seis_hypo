from __future__ import annotations

import datetime as dt
import mmap
import warnings
from collections.abc import Sequence
from pathlib import Path

import numba
import numpy as np
import pandas as pd
from obspy import UTCDateTime

from common.core import slice_with_pad
from common.time_util import floor_minute
from jma.station_reader import read_hinet_channel_table


@numba.jit(nopython=True, cache=True)
def _number_BCD(mm, length):
	out = 0
	for i in range(length):
		out += (mm[i] // 16 * 10 + (mm[i] % 16)) * 100 ** (length - 1 - i)
	return out


@numba.jit(nopython=True, cache=True)
def _sampling_rate(mm):
	return ((mm[0] & 0x0F) << 4) | mm[1]


@numba.jit(nopython=True, cache=True)
def _channel_no(mm):
	return (mm[0] << 8) | mm[1]


@numba.jit(nopython=True, cache=True)
def _sample0(mm):
	out = (mm[0] << 24) | (mm[1] << 16) | (mm[2] << 8) | mm[3]
	out = -((1 << (32 - 1)) & out) | out
	return out


@numba.jit(nopython=True, cache=True)
def _4bytes(mm, count):
	out = np.zeros(count, dtype=numba.int32)
	for i in range(count):
		aa = (
			(mm[i * 4] << 24)
			| (mm[i * 4 + 1] << 16)
			| (mm[i * 4 + 2] << 8)
			| mm[i * 4 + 3]
		)
		aa = -((1 << (32 - 1)) & aa) | aa
		out[i] = aa
	return out


@numba.jit(nopython=True, cache=True)
def _3bytes(mm, count):
	out = np.zeros(count, dtype=numba.int32)
	for i in range(count):
		aa = (mm[i * 3] << 16) | (mm[i * 3 + 1] << 8) | mm[i * 3 + 2]
		aa = -((1 << (24 - 1)) & aa) | aa
		out[i] = (0x00 << 24) | aa
	return out


@numba.jit(nopython=True, cache=True)
def _2bytes(mm, count):
	out = np.zeros(count, dtype=numba.int32)
	for i in range(count):
		aa = (mm[i * 2] << 8) | mm[i * 2 + 1]
		aa = -((1 << (16 - 1)) & aa) | aa
		out[i] = (0x00 << 24) | aa
	return out


@numba.jit(nopython=True, cache=True)
def _1byte(mm, count):
	out = np.zeros(count, dtype=np.int32)
	for i in range(count):
		aa = -((1 << (8 - 1)) & mm[i]) | mm[i]
		out[i] = (0x00 << 24) | aa
	return out[:count]


@numba.jit(nopython=True, cache=True)
def _05byte(mm, count):
	if count % 2 == 0:
		cnt = count
	else:
		cnt = count + 1
	out = np.zeros(cnt, dtype=np.int32)

	for i in range(cnt // 2):
		aa = (mm[i] & 0xF0) >> 4
		aa = -((1 << (4 - 1)) & aa) | aa
		out[2 * i] = (0x00 << 24) | aa

	for i in range(count // 2):
		aa = mm[i] & 0x0F
		aa = -((1 << (4 - 1)) & aa) | aa
		out[2 * i + 1] = (0x00 << 24) | aa

	return out


@numba.jit(nopython=True, cache=True)
def _process_secondblock(mm, secondblock_BYTES, channel_array, base_sampling_rate_HZ):
	offset = 0
	nch = len(channel_array)
	out = np.zeros((nch, base_sampling_rate_HZ), dtype=np.int32)

	while offset < secondblock_BYTES:
		sid = _number_BCD(mm[offset + 0 : offset + 1], 1)
		mid = _number_BCD(mm[offset + 1 : offset + 2], 1)

		channel_no = _channel_no(mm[offset + 2 : offset + 4])

		sample_size_code = (mm[offset + 4] & 0xF0) >> 4
		sampling_rate_HZ = _sampling_rate(mm[offset + 4 : offset + 6])

		if sampling_rate_HZ != base_sampling_rate_HZ:
			print('sampling_rate of this block is not base_sampling_rate_HZ')

		diff_sample_number = sampling_rate_HZ - 1

		a = offset + 10
		if sample_size_code == 0:
			if diff_sample_number % 2 == 0:
				diff_samples_size_BYTE = diff_sample_number // 2
			else:
				diff_samples_size_BYTE = diff_sample_number // 2 + 1

		elif sample_size_code == 1:
			diff_samples_size_BYTE = diff_sample_number * 1

		elif sample_size_code == 2:
			diff_samples_size_BYTE = diff_sample_number * 2

		elif sample_size_code == 3:
			diff_samples_size_BYTE = diff_sample_number * 3

		elif sample_size_code == 4:
			diff_samples_size_BYTE = diff_sample_number * 4

		b = a + diff_samples_size_BYTE

		for i in range(nch):
			if channel_no != channel_array[i]:
				continue

			sample0 = _sample0(mm[offset + 6 : offset + 10])
			samples = np.zeros(diff_sample_number + 1, dtype=np.int32)
			samples = samples + sample0

			if sample_size_code == 0:
				diff_samples = _05byte(mm[a:b], diff_sample_number)

			elif sample_size_code == 1:
				diff_samples = _1byte(mm[a:b], diff_sample_number)

			elif sample_size_code == 2:
				diff_samples = _2bytes(mm[a:b], diff_sample_number)

			elif sample_size_code == 3:
				diff_samples = _3bytes(mm[a:b], diff_sample_number)

			elif sample_size_code == 4:
				diff_samples = _4bytes(mm[a:b], diff_sample_number)

			samples[1:] = samples[1:] + np.cumsum(diff_samples[:diff_sample_number])
			out[i] = samples

		offset = b

	return out


@numba.jit(nopython=True, cache=True)
def _datetime(mm):
	year = str(mm[0] // 16) + str(mm[0] % 16) + str(mm[1] // 16) + str(mm[1] % 16)
	month = str(mm[2] // 16) + str(mm[2] % 16)
	day = str(mm[3] // 16) + str(mm[3] % 16)
	hour = str(mm[4] // 16) + str(mm[4] % 16)
	minute = str(mm[5] // 16) + str(mm[5] % 16)
	second = str(mm[6] // 16) + str(mm[6] % 16)
	deci = str(mm[7] % 16)
	out = (
		year
		+ '-'
		+ month
		+ '-'
		+ day
		+ 'T'
		+ hour
		+ ':'
		+ minute
		+ ':'
		+ second
		+ '.'
		+ deci
		+ '00'
	)
	return out


@numba.jit(nopython=True, cache=True)
def _secondblock_BYTES(mm):
	return (mm[0] << 24) | (mm[1] << 16) | (mm[2] << 8) | mm[3]


@numba.jit(nopython=True, cache=True)
def _process_file(
	mm,
	file_size,
	sampling_rate_HZ,
	channel_no_int_array,
	conversion_coefficients_array,
	number_output_channels,
	number_outout_timesamples,
):
	output = np.zeros(
		(number_output_channels, number_outout_timesamples), dtype=np.float32
	)
	time_index = 0
	offset = 4
	reshaped_conversion_array = conversion_coefficients_array.reshape(-1, 1)
	ndarrayed_channel_no_int_array = channel_no_int_array

	while offset < file_size:
		secondblock_BYTES = _secondblock_BYTES(mm[offset + 12 : offset + 16])

		if secondblock_BYTES == 0:
			return time_index, output

		offset += 16

		secondblock = _process_secondblock(
			mm[offset : offset + secondblock_BYTES],
			secondblock_BYTES,
			ndarrayed_channel_no_int_array,
			sampling_rate_HZ,
		)

		output[:, time_index : time_index + sampling_rate_HZ] = (
			reshaped_conversion_array * secondblock
		)
		offset += secondblock_BYTES
		time_index += sampling_rate_HZ

	return time_index, output


def _process_file_with_timestamp(
	mm,
	file_size,
	sampling_rate_HZ,
	channel_no_int_array,
	conversion_coefficients_array,
	number_output_channels,
	number_outout_timesamples,
):
	output = np.zeros(
		(number_output_channels, number_outout_timesamples), dtype=np.float32
	)
	time_index = 0
	offset = 4
	reshaped_conversion_array = np.array(conversion_coefficients_array).reshape(-1, 1)
	ndarrayed_channel_no_int_array = np.array(channel_no_int_array)

	timestamp = _datetime(mm[offset + 0 : offset + 8])
	if timestamp[:10] == '0000-00-00':
		time0 = None
	else:
		time0 = np.datetime64(timestamp, 'ms')
	time_index_offset = 0

	while offset < file_size:
		timestamp = _datetime(mm[offset + 0 : offset + 8])
		if timestamp[:10] != '0000-00-00':
			time1 = np.datetime64(timestamp, 'ms')
			if time0 is None:
				time0 = time1
				time_index_offset = time1.astype(int) // 1000 * sampling_rate_HZ
			else:
				time_index = (
					time_index_offset
					+ (time1 - time0).astype(int) // 1000 * sampling_rate_HZ
				)

		secondblock_BYTES = _secondblock_BYTES(mm[offset + 12 : offset + 16])
		if secondblock_BYTES == 0:
			print(
				f'terminated {offset}/{file_size} because block size is not found {mm[offset + 12 : offset + 16]}'
			)
			return output
		offset += 16

		secondblock = _process_secondblock(
			mm[offset : offset + secondblock_BYTES],
			secondblock_BYTES,
			ndarrayed_channel_no_int_array,
			sampling_rate_HZ,
		)

		output[:, time_index : time_index + sampling_rate_HZ] = (
			reshaped_conversion_array * secondblock
		)
		offset += secondblock_BYTES
	return output


def parse_win32_starttime(file_path: str | Path) -> UTCDateTime:
	"""WIN32 先頭ブロックのタイムスタンプから開始時刻を取得.

	win32_reader.py 内の _datetime を利用している前提。
	"""
	file_path = Path(file_path)
	if not file_path.is_file():
		msg = f'WIN32 file not found: {file_path}'
		raise FileNotFoundError(msg)

	with Path.open(file_path, 'rb') as f:
		mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

	offset = 4
	ts = _datetime(mm[offset + 0 : offset + 8])

	if ts[:10] == '0000-00-00':
		msg = f'WIN32 timestamp is not available at file head: {file_path.name}'
		raise ValueError(msg)

	return UTCDateTime(ts)


def select_hinet_channels(
	channel_table: pd.DataFrame | str | Path,
	*,
	channels_hex: list[str] | None = None,
	station: str | None = None,
	components: list[str] | None = None,
) -> pd.DataFrame:
	"""Hi-net channel table を読み込み、フィルタ済み DataFrame を返す。

	この返り値の行順が read_win32 の出力 ndarray の行順になる前提で使う。
	"""
	if isinstance(channel_table, (str, Path)):
		df = read_hinet_channel_table(channel_table)
	else:
		df = channel_table.copy()

	if channels_hex is not None:
		hex_set = {s.upper() for s in channels_hex}
		df = df[df['ch_hex'].isin(hex_set)]

	if station is not None:
		df = df[df['station'] == station]

	if components is not None:
		df = df[df['component'].isin(components)]

	if df.empty:
		raise ValueError(
			'no channels selected (check filters: channels_hex/station/components)'
		)

	return df.reset_index(drop=True)


def read_win32(
	file_path: str | Path,
	channel_table: pd.DataFrame | str,
	*,
	base_sampling_rate_HZ: int = 100,
	duration_SECOND: int = 15 * 60,
	channels_hex: list[str] | None = None,  # 例: ["0003","0004","0005"]
	station: str | None = None,  # 例: "N.AGWH"
	components: list[str] | None = None,  # 例: ["U","N","E"]
) -> np.ndarray:
	"""Hi-net チャネル表 DataFrame を元に、WIN32 を読み出して物理量に換算して返す.

	- channel_table: read_hinet_channel_table() の戻り値(列: ch_hex/ch_int/conv_coeff 等)
	- channels_hex / station / components で抽出条件を指定(いずれか/併用可)
	- 返り値: shape=(n_ch, duration_SECOND * base_sampling_rate_HZ), dtype=float32
	"""
	file_path = Path(file_path)
	if not file_path.is_file():
		msg = f'WIN32 file not found: {file_path}'
		raise FileNotFoundError(msg)

	df = select_hinet_channels(
		channel_table,
		channels_hex=channels_hex,
		station=station,
		components=components,
	)

	# 必要列を取り出し
	channel_no_int_array = df['ch_int'].to_numpy(dtype=np.int32)
	conversion_coefficients_array = df['conv_coeff'].to_numpy(dtype=np.float32)

	number_output_channels = len(df)
	number_output_timesamples = int(duration_SECOND * base_sampling_rate_HZ)

	file_size = file_path.stat().st_size

	with file_path.open('rb') as f:
		mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

		# ブロック数が揃っているか確認しつつ処理
		n_read, output = _process_file(
			mm,
			file_size,
			base_sampling_rate_HZ,
			channel_no_int_array,
			conversion_coefficients_array,
			number_output_channels,
			number_output_timesamples,
		)

		# 欠落秒があればタイムスタンプ経路で再配置
		if n_read != number_output_timesamples:
			warnings.warn(
				f'found missing second blocks in {file_path.name}; '
				'fallback to timestamp-based placement',
				RuntimeWarning,
			)
			output = _process_file_with_timestamp(
				mm,
				file_size,
				base_sampling_rate_HZ,
				channel_no_int_array.tolist(),
				conversion_coefficients_array.tolist(),
				number_output_channels,
				number_output_timesamples,
			)

	return output


def compute_event_time_window(
	origin_time: dt.datetime,
	*,
	pre_sec: int,
	post_sec: int,
) -> tuple[dt.datetime, dt.datetime]:
	t_start = origin_time - dt.timedelta(seconds=pre_sec)
	t_end = origin_time + dt.timedelta(seconds=post_sec)
	if t_end <= t_start:
		msg = 't_end must be later than t_start'
		raise ValueError(msg)
	return t_start, t_end


def read_event_win32_window_as_array(
	event_row: pd.Series,
	ch_path: Path,
	cnt_path_list: Sequence[Path],
	*,
	base_sampling_rate_HZ: int = 100,
	pre_sec: int,
	post_sec: int,
) -> Tuple[np.ndarray, pd.DataFrame, dt.datetime, dt.datetime]:
	"""イベント1件について、pre/post 秒の窓に対応する ndarray を返す。

	戻り値:
	- arr_event: shape=(n_ch, n_samples_event)
	- station_df: read_hinet_channel_table(ch_path) の結果
	- t_start: 窓の開始時刻（JST）
	- t_end:   窓の終了時刻（JST）
	"""
	if not cnt_path_list:
		msg = 'cnt_path_list is empty'
		raise ValueError(msg)

	origin_time = pd.to_datetime(event_row['origin_time'])
	t_start, t_end = compute_event_time_window(
		origin_time,
		pre_sec=pre_sec,
		post_sec=post_sec,
	)

	station_df = read_hinet_channel_table(ch_path)

	# 1分タイルを time 軸に連結
	arr_list: list[np.ndarray] = []
	for cnt_path in sorted(cnt_path_list):
		arr_min = read_win32(
			cnt_path,
			station_df,
			base_sampling_rate_HZ=base_sampling_rate_HZ,
			duration_SECOND=60,
		)  # (n_ch, 60*fs)
		arr_list.append(arr_min)

	if len(arr_list) == 1:
		arr_concat = arr_list[0]
	else:
		arr_concat = np.concatenate(arr_list, axis=1)  # (n_ch, nt_total)

	# 連結列の time=0 を、最初の minute の分頭とみなす
	first_minute = floor_minute(t_start)
	nt_total = arr_concat.shape[1]
	total_sec = nt_total / float(base_sampling_rate_HZ)

	offset_start_sec = (t_start - first_minute).total_seconds()
	offset_end_sec = (t_end - first_minute).total_seconds()

	start_idx = int(round(offset_start_sec * base_sampling_rate_HZ))
	end_idx = int(round(offset_end_sec * base_sampling_rate_HZ))

	start_idx = max(start_idx, 0)
	end_idx = min(end_idx, int(round(total_sec * base_sampling_rate_HZ)))

	arr_event = slice_with_pad(arr_concat, start_idx, end_idx)

	return arr_event, station_df, t_start, t_end
