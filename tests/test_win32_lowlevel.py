import numpy as np

import jma.win32_reader as wr


def _py(fn):
	return getattr(fn, 'py_func', fn)


def test_sampling_rate_and_channel_no():
	sampling_rate = _py(wr._sampling_rate)
	channel_no = _py(wr._channel_no)

	mm = bytes([0xA5, 0x3C])
	assert sampling_rate(mm) == (((0xA5 & 0x0F) << 4) | 0x3C)

	mm2 = bytes([0x12, 0x34])
	assert channel_no(mm2) == 0x1234


def test_sample0_sign_extend_32bit():
	sample0 = _py(wr._sample0)

	assert sample0(bytes([0x00, 0x00, 0x00, 0x01])) == 1
	assert sample0(bytes([0x7F, 0xFF, 0xFF, 0xFF])) == 2147483647
	assert sample0(bytes([0x80, 0x00, 0x00, 0x00])) == -2147483648
	assert sample0(bytes([0xFF, 0xFF, 0xFF, 0xFF])) == -1


def test_1byte_sign_extend():
	f = _py(wr._1byte)
	mm = bytes([0x00, 0x7F, 0x80, 0xFF])
	out = f(mm, 4)
	assert out.dtype == np.int32
	assert out.tolist() == [0, 127, -128, -1]


def test_2bytes_sign_extend():
	f = _py(wr._2bytes)
	mm = bytes([0x00, 0x01, 0x7F, 0xFF, 0x80, 0x00, 0xFF, 0xFF])
	out = f(mm, 4)
	assert out.tolist() == [1, 32767, -32768, -1]


def test_3bytes_sign_extend():
	f = _py(wr._3bytes)
	mm = bytes(
		[
			0x00,
			0x00,
			0x01,
			0x7F,
			0xFF,
			0xFF,
			0x80,
			0x00,
			0x00,
			0xFF,
			0xFF,
			0xFF,
		]
	)
	out = f(mm, 4)
	assert out.tolist() == [1, 8388607, -8388608, -1]


def test_4bytes_sign_extend():
	f = _py(wr._4bytes)
	mm = bytes(
		[
			0x00,
			0x00,
			0x00,
			0x01,
			0x7F,
			0xFF,
			0xFF,
			0xFF,
			0x80,
			0x00,
			0x00,
			0x00,
			0xFF,
			0xFF,
			0xFF,
			0xFF,
		]
	)
	out = f(mm, 4)
	assert out.tolist() == [1, 2147483647, -2147483648, -1]


def test_05byte_nibbles_and_odd_count():
	f = _py(wr._05byte)
	# 上位ニブル: 0xF -> -1, 下位ニブル: 0x0 -> 0 のペア
	mm = bytes([0xF0, 0x10])
	out = f(mm, 3)  # 奇数count
	assert out[:3].tolist() == [-1, 0, 1]


def test_datetime_bcd_to_str():
	f = _py(wr._datetime)
	# 2010-01-02T03:04:05.600
	ts8 = bytes([0x20, 0x10, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06])
	out = f(ts8)
	assert out == '2010-01-02T03:04:05.600'
