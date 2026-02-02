from pathlib import Path

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


def _secondblock_header(ts8, block_size):
	return ts8 + bytes([0x00, 0x00, 0x00, 0x00]) + int(block_size).to_bytes(4, 'big')


def _pack_subblock(channel_no, fs, sample_size_code):
	diff = fs - 1
	if sample_size_code == 1:
		diff_bytes = bytes([0x00] * diff)
	elif sample_size_code == 2:
		diff_bytes = bytes([0x00] * (diff * 2))
	else:
		diff_bytes = bytes([0x00] * diff)
	b4 = ((sample_size_code & 0x0F) << 4) | ((fs >> 4) & 0x0F)
	b5 = fs & 0xFF
	hdr10 = bytes(
		[0x00, 0x00, (channel_no >> 8) & 0xFF, channel_no & 0xFF, b4, b5, 0, 0, 0, 0]
	)
	return hdr10 + diff_bytes


def test_get_evt_info_missing_and_rates(tmp_path: Path):
	fp = tmp_path / 'x.evt'
	fs = 10
	payload = _pack_subblock(0x0003, fs, 1)
	block_size = len(payload)

	# 4B reserved + (16B header + payload) * 2
	data = bytearray()
	data += bytes([0, 0, 0, 0])

	ts0 = _bcd_ts8(2010, 1, 1, 0, 0, 0, 0)
	ts2 = _bcd_ts8(2010, 1, 1, 0, 0, 2, 0)
	data += _secondblock_header(ts0, block_size)
	data += payload
	data += _secondblock_header(ts2, block_size)
	data += payload
	data += _secondblock_header(_bcd_ts8(0, 0, 0, 0, 0, 0, 0), 0)

	fp.write_bytes(data)

	info = wr.get_evt_info(fp, scan_rate_blocks=2)
	assert info.span_seconds == 3
	assert info.missing_seconds_est == 1
	assert info.sampling_rates_hz == (fs,)
	assert info.base_sampling_rate_hz == fs
