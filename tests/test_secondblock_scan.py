from collections import defaultdict

import jma.win32_reader as wr


def _pack_subblock(channel_no, fs, sample_size_code, sample0, diff_bytes):
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
	return hdr10 + diff_bytes


def test_diff_samples_size_bytes_all_codes():
	assert wr._diff_samples_size_bytes(0, 4) == 2  # diff=3 -> ceil(3/2)=2
	assert wr._diff_samples_size_bytes(1, 4) == 3
	assert wr._diff_samples_size_bytes(2, 4) == 6
	assert wr._diff_samples_size_bytes(3, 4) == 9
	assert wr._diff_samples_size_bytes(4, 4) == 12

	try:
		wr._diff_samples_size_bytes(9, 4)
	except ValueError:
		pass
	else:
		raise AssertionError('ValueError expected')


def test_scan_sampling_rates_in_secondblock():
	payload = b''
	payload += _pack_subblock(0x0003, 100, 1, 0, b'\x00' * 99)
	payload += _pack_subblock(0x0004, 50, 2, 0, b'\x00' * (49 * 2))
	rates = wr._scan_sampling_rates_in_secondblock(payload)
	assert rates == {100, 50}


def test_scan_secondblock_channel_rates_and_filter():
	rates_by_ch = defaultdict(set)

	payload = b''
	payload += _pack_subblock(0x0003, 100, 1, 0, b'\x00' * 99)
	payload += _pack_subblock(0x0004, 50, 2, 0, b'\x00' * (49 * 2))

	wr._scan_secondblock_channel_rates(payload, rates_by_ch, channel_filter={0x0003})

	assert rates_by_ch[0x0003] == {100}
	assert 0x0004 not in rates_by_ch


def test_scan_secondblock_channel_rates_truncated_header():
	rates_by_ch = defaultdict(set)

	channel_no = 0x0001
	fs = 2
	sample_size_code = 1
	b4 = ((sample_size_code & 0x0F) << 4) | ((fs >> 4) & 0x0F)
	b5 = fs & 0xFF
	hdr10 = bytes(
		[0x00, 0x00, (channel_no >> 8) & 0xFF, channel_no & 0xFF, b4, b5, 0, 0, 0, 0]
	)

	payload = hdr10 + b'\x00' + b'\x00'  # 正常11 + ゴミ1

	try:
		wr._scan_secondblock_channel_rates(payload, rates_by_ch)
	except ValueError as e:
		assert 'truncated subblock header' in str(e)
	else:
		raise AssertionError('ValueError expected')
