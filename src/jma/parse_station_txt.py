#!/usr/bin/env python3
from pandas import DataFrame, concat

ENCODING = 'euc_jp'  # 文字化けする場合は "utf-8" に変更してください


def is_date_token(token: str) -> bool:
	return token.isdigit() and len(token) == 8


def format_date(token: str) -> str:
	if token == '':
		return ''
	return token[0:4] + '-' + token[4:6] + '-' + token[6:8]


def first_non_ascii_idx(s: str) -> int:
	for i, ch in enumerate(s):
		if ord(ch) >= 128:
			return i
	return -1


def parse_station_line(line: str, comment: str):
	s = line.rstrip('\n')
	if s.strip() == '':
		return None

	# 56文字未満は観測点行として成立しない（安全側に落とす）
	if len(s) < 56:
		return None

	# 固定位置（0-based, end exclusive）
	# code:   [0:6]
	# number: [6:12]
	# latdeg: [12:15]
	# latmin: [15:22]
	# londeg: [22:26]
	# lonmin: [26:33]
	# height: [33:39]  ※負の標高で "-" がここに入る
	# From:   [39:47]
	# (space) [47:48]
	# To:     [48:56]
	# tail:   [56:]
	code = s[0:6].strip()
	number_str = s[6:12].strip()

	lat_deg_str = s[12:15].strip()
	lat_min_str = s[15:22].strip()
	lon_deg_str = s[22:26].strip()
	lon_min_str = s[26:33].strip()

	height_str = s[33:39].strip()

	from_raw = s[39:47].strip()
	to_raw = s[48:56].strip()

	tail = s[56:].rstrip()
	non_ascii = first_non_ascii_idx(tail)
	if non_ascii == -1:
		seismographs = tail.strip()
		area = ''
	else:
		seismographs = tail[:non_ascii].strip()
		area = tail[non_ascii:].strip()

	station_number = int(number_str)
	lat_deg = float(lat_deg_str)
	lat_min = float(lat_min_str)
	lon_deg = float(lon_deg_str)
	lon_min = float(lon_min_str)
	height = int(height_str)

	latitude = lat_deg + lat_min / 60.0
	longitude = lon_deg + lon_min / 60.0

	from_date = format_date(from_raw) if is_date_token(from_raw) else ''
	to_date = format_date(to_raw) if is_date_token(to_raw) else ''

	return {
		'station_code': code,
		'station_number': station_number,
		'Latitude_deg': latitude,
		'Longitude_deg': longitude,
		'Height': height,
		'From': from_date,
		'To': to_date,
		'Seismographs': seismographs,
		'Area': area,
		'Comment': comment,
	}


def build_station_csv_from_jma_txt(input_path: str, output_path: str) -> None:
	fieldnames: list[str] = [
		'station_code',
		'station_number',
		'Latitude_deg',
		'Longitude_deg',
		'Height',
		'From',
		'To',
		'Seismographs',
		'Area',
		'Comment',
	]

	with open(input_path, encoding=ENCODING) as f:
		lines = f.readlines()

	comment_indices: list[int] = []
	for i, line in enumerate(lines):
		if 'JMA  Station' in line:
			comment_indices.append(i - 1)

	df = DataFrame([], columns=fieldnames)

	for i, comment_idx in enumerate(comment_indices):
		comment = lines[comment_idx].strip()
		start_idx = comment_idx + 4

		if i + 1 < len(comment_indices):
			end_idx = comment_indices[i + 1] - 1
		else:
			end_idx = len(lines)

		rows = []
		for j in range(start_idx, end_idx):
			raw = lines[j]
			if raw.strip() == '':
				continue
			row = parse_station_line(raw, comment)
			if row is None:
				continue
			rows.append(row)

		if not rows:
			continue

		df_part = DataFrame(rows, columns=fieldnames)
		df = concat([df, df_part], ignore_index=True)

	df.to_csv(output_path, index=False, encoding='utf-8')
