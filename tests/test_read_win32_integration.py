import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import jma.win32_reader as wr


def _bcd_ts8(y, mo, da, hh, mi, ss, deci):
	y_th = (y // 1000) % 10
	y_hu = (y // 100) % 10
	y_te = (y // 10) % 10
	y_on = y % 10
	b0 = (y_th << 4) | y_hu
	b1 = (y_te << 4) | y_on
	b2 = ((mo // 10) << 4) | (mo % 10)
	b3 = ((da // 10) << 4) | (da % 10)
	b4 = ((hh // 10) << 4) | (hh % 10)
	b5 = ((mi // 10) << 4) | (mi % 10)
	b6 = ((ss // 10) << 4) | (ss % 10)
	b7 = deci & 0x0F
	return bytes([b0, b1, b2, b3, b4, b5, b6, b7])


def _header(ts8, block_size):
	return ts8 + bytes([0, 0, 0, 0]) + int(block_size).to_bytes(4, 'big')


def _subblock_1ch(channel_no, fs, sample0, diffs_1byte):
	sample_size_code = 1
	b4 = ((sample_size_code & 0x0F) << 4) | ((fs >> 4) & 0x0F)
	b5 = fs & 0xFF
	hdr10 = bytes(
		[
			0x00,
			0x00,
			(channel_no >> 8) & 0xFF,
			channel_no & 0xFF,
			b4,
			b5,
			(sample0 >> 24) & 0xFF,
			(sample0 >> 16) & 0xFF,
			(sample0 >> 8) & 0xFF,
			sample0 & 0xFF,
		]
	)
	return hdr10 + bytes(diffs_1byte)


def test_read_win32_applies_coeff_and_missing_block_fallback(tmp_path: Path):
	fp = tmp_path / 'x.cnt'
	ch = 0x0003
	fs = 4
	duration = 3  # 3秒ぶん期待（12サンプル）

	# 秒0 と 秒2 だけ入れて欠落を作る
	payload0 = _subblock_1ch(ch, fs, sample0=100, diffs_1byte=[1, 1, 1])
	payload2 = _subblock_1ch(ch, fs, sample0=200, diffs_1byte=[1, 1, 1])
	bs = len(payload0)

	data = bytearray()
	data += bytes([0, 0, 0, 0])
	data += _header(_bcd_ts8(2010, 1, 1, 0, 0, 0, 0), bs)
	data += payload0
	data += _header(_bcd_ts8(2010, 1, 1, 0, 0, 2, 0), bs)
	data += payload2
	data += _header(_bcd_ts8(0, 0, 0, 0, 0, 0, 0), 0)
	fp.write_bytes(data)

	df = pd.DataFrame(
		{
			'ch_hex': ['0003'],
			'ch_int': [ch],
			'conv_coeff': [2.0],
			'station': ['N.TEST'],
			'component': ['U'],
		}
	)

	with warnings.catch_warnings(record=True) as w:
		warnings.simplefilter('always')
		out = wr.read_win32(fp, df, base_sampling_rate_HZ=fs, duration_SECOND=duration)

	assert out.shape == (1, duration * fs)
	assert out.dtype == np.float32
	assert any(isinstance(x.message, RuntimeWarning) for x in w)

	# 秒0: sample0=100, diffs=[1,1,1] -> [100,101,102,103] に係数2倍
	assert out[0, 0:4].tolist() == [200.0, 202.0, 204.0, 206.0]

	# 秒1（欠落）はゼロ埋め想定
	assert out[0, 4:8].tolist() == [0.0, 0.0, 0.0, 0.0]

	# 秒2: sample0=200 -> [200,201,202,203] に係数2倍
	assert out[0, 8:12].tolist() == [400.0, 402.0, 404.0, 406.0]
