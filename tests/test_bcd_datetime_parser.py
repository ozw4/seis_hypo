from datetime import datetime

import jma.win32_reader as wr


def test_parse_bcd_datetime_8_valid():
	ts8 = bytes([0x20, 0x10, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06])
	dt = wr._parse_bcd_datetime_8(ts8)
	assert dt == datetime(2010, 1, 2, 3, 4, 5, 600000)


def test_parse_bcd_datetime_8_none_for_zero():
	ts8 = bytes([0x00, 0x00, 0x00, 0x00, 0x12, 0x34, 0x56, 0x07])
	assert wr._parse_bcd_datetime_8(ts8) is None


def test_parse_bcd_datetime_8_len_error():
	try:
		wr._parse_bcd_datetime_8(b'\x00')
	except ValueError as e:
		assert '8 bytes' in str(e)
	else:
		raise AssertionError('ValueError expected')
