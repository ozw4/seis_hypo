# %%
#!/usr/bin/env python3
from pandas import DataFrame, concat

ENCODING = 'euc_jp'  # 文字化けする場合は "utf-8" に変更してください


def is_date_token(token: str) -> bool:
	return token.isdigit() and len(token) == 8


def is_ascii_token(token: str) -> bool:
	for ch in token:
		if ord(ch) >= 128:
			return False
	return True


def format_date(token: str) -> str:
	if token == '':
		return ''
	return token[0:4] + '-' + token[4:6] + '-' + token[6:8]


def parse_station_line(line: str, comment: str):
	stripped = line.strip()
	if stripped == '':
		return None

	tokens = line.rstrip('\n').split()
	if len(tokens) < 7:
		return None

	code = tokens[0]
	number_str = tokens[1]
	lat_deg_str = tokens[2]
	lat_min_str = tokens[3]
	lon_deg_str = tokens[4]
	lon_min_str = tokens[5]
	height_str = tokens[6]

	idx = 7
	from_raw = ''
	to_raw = ''

	if idx < len(tokens) and is_date_token(tokens[idx]):
		from_raw = tokens[idx]
		idx += 1
	if idx < len(tokens) and is_date_token(tokens[idx]):
		to_raw = tokens[idx]
		idx += 1

	seismo_parts = []
	area_parts = []
	saw_area = False

	for token in tokens[idx:]:
		if not saw_area and is_ascii_token(token):
			seismo_parts.append(token)
		else:
			saw_area = True
			area_parts.append(token)

	if not area_parts and seismo_parts:
		area_parts.append(seismo_parts.pop())

	seismographs = ' '.join(seismo_parts)
	area = ' '.join(area_parts)

	lat_deg = float(lat_deg_str)
	lat_min = float(lat_min_str)
	lon_deg = float(lon_deg_str)
	lon_min = float(lon_min_str)
	latitude = lat_deg + lat_min / 60.0
	longitude = lon_deg + lon_min / 60.0
	height = int(height_str)
	station_number = int(number_str)

	from_date = format_date(from_raw)
	to_date = format_date(to_raw)

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
			# "JMA  Station" の一つ前の行がコメント行
			comment_indices.append(i - 1)

	df = DataFrame([], columns=fieldnames)

	for i, comment_idx in enumerate(comment_indices):
		comment = lines[comment_idx].strip()

		# データ開始行
		start_idx = comment_idx + 4

		# 次のコメント行の一つ前までをこのブロックとみなす
		if i + 1 < len(comment_indices):
			end_idx = comment_indices[i + 1] - 1
		else:
			# 最後のブロックはファイル末尾まで
			end_idx = len(lines)

		rows = []
		for j in range(start_idx, end_idx):
			line = lines[j].strip()
			if not line:
				continue
			row = parse_station_line(line, comment)
			rows.append(row)

		if not rows:
			continue

		df_part = DataFrame(rows, columns=fieldnames)
		df = concat([df, df_part], ignore_index=True)

	df.to_csv(output_path, index=False, encoding='utf-8')
