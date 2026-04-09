import csv
from pathlib import Path

from tqdm import tqdm

RECORD_LENGTH = 96

FIELD_WIDTHS = [
	1,  # 1 レコード種別ヘッダ A1
	4,  # 2 西暦 I4
	2,  # 3 月 I2
	2,  # 4 日 I2
	2,  # 5 時 I2
	2,  # 6 分 I2
	4,  # 7 秒 F4.2（小数点以下2桁）
	4,  # 8 標準誤差（秒） F4.2
	3,  # 9 緯度（度） I3
	4,  # 10 緯度（分） F4.2
	4,  # 11 緯度標準誤差（分） F4.2
	4,  # 12 経度（度） I4
	4,  # 13 経度（分） F4.2
	4,  # 14 経度標準誤差（分） F4.2
	5,  # 15 深さ（km） F5.2
	3,  # 16 深さ標準誤差（km） F3.2
	2,  # 17 マグニチュード1 F2.1
	1,  # 18 マグニチュード1種別 A1
	2,  # 19 マグニチュード2 F2.1
	1,  # 20 マグニチュード2種別 A1
	1,  # 21 使用走時表 A1
	1,  # 22 震源評価 A1
	1,  # 23 震源補助情報 A1
	1,  # 24 最大震度 A1
	1,  # 25 被害規模 A1
	1,  # 26 津波規模 A1
	1,  # 27 大地域区分番号 I1
	3,  # 28 小地域区分番号 I3
	24,  # 29 震央地名 A24
	3,  # 30 観測点数 I3
	1,  # 31 震源決定フラグ A1
]

EPICENTER_HEADERS = {'J'}

EVENT_FIELD_NAMES = [
	'event_id',
	'record_type',
	'origin_time',
	'origin_time_std_s',
	'latitude_deg',
	'latitude_std_deg',
	'longitude_deg',
	'longitude_std_deg',
	'depth_km',
	'depth_std_km',
	'mag1',
	'mag1_type',
	'mag2',
	'mag2_type',
	'tt_table',
	'hypocenter_eval',
	'hypocenter_info',
	'max_intensity',
	'damage_scale',
	'tsunami_scale',
	'region_large_code',
	'region_small_code',
	'epicenter_name',
	'station_count',
	'hypocenter_flag',
]


def split_fixed(line: str):
	fields = []
	pos = 0
	for w in FIELD_WIDTHS:
		fields.append(line[pos : pos + w])
		pos += w
	return fields


def parse_int_field(s: str):
	s = s.strip()
	return int(s) if s else None


def parse_fxx_2_field(s: str):
	# F4.2, F5.2, F3.2 のように小数点以下2桁を整数で持っているフィールド
	# 例: "5115" -> 51.15
	s = s.strip()
	return int(s) / 100.0 if s else None


def parse_mag_field(s: str):
	"""F2.1 マグニチュード用フィールドを float に変換する。

	正値 : '43' -> 4.3

	負値は気象庁ルールでエンコードされている:
	-0.1 ～ -0.9 : '-1' ～ '-9'
	-1.0 ～ -1.9 : 'A0' ～ 'A9'
	-2.0 : 'B0'
	-3.0 : 'C0'

	1 桁の整数のみの場合は、その整数をそのまま M.x0 として扱う（例: '5' -> 5.0）。
	空白は None を返す。
	"""
	s = s.strip()
	if not s:
		return None

	# 一文字だけの整数マグニチュード（例: '5' -> 5.0）
	if len(s) == 1 and s.isdigit():
		return float(s)

	if len(s) != 2:
		raise ValueError(f'unexpected F2.1 field length: {s!r}')

	a, b = s[0], s[1]

	# -0.1 ～ -0.9 : '-1' ～ '-9'
	if a == '-' and b.isdigit():
		return -0.1 * int(b)

	# -1.0 ～ -1.9 : 'A0' ～ 'A9'
	if a == 'A' and b.isdigit():
		return -1.0 - 0.1 * int(b)

	# -2.0 : 'B0'
	if a == 'B' and b == '0':
		return -2.0

	# -3.0 : 'C0'
	if a == 'C' and b == '0':
		return -3.0

	# それ以外は 0.0 ～ 9.9 の通常の F2.1（例: '43' -> 4.3）
	if s.isdigit():
		return int(s) / 10.0

	raise ValueError(f'invalid F2.1 magnitude code: {s!r}')


def build_origin_time(year, month, day, hour, minute, second):
	# "YYYY-MM-DDTHH:MM:SS.ss" 形式の文字列を作る
	if None in (year, month, day, hour, minute, second):
		return ''

	sec_str = f'{second:.2f}'
	if second < 10.0 and not sec_str.startswith('0'):
		sec_str = '0' + sec_str

	return f'{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{sec_str}'


def parse_epicenter_line(line: str, verbose: bool = False):
	if len(line) != RECORD_LENGTH:
		raise ValueError(
			f'unexpected line length {len(line)} (expected {RECORD_LENGTH})'
		)

	if verbose:
		print(line)

	header = line[0]
	if header not in EPICENTER_HEADERS:
		return None

	f = split_fixed(line)

	year = parse_int_field(f[1])
	month = parse_int_field(f[2])
	day = parse_int_field(f[3])
	hour = parse_int_field(f[4])
	minute = parse_int_field(f[5])
	second = parse_fxx_2_field(f[6])
	origin_std = parse_fxx_2_field(f[7])

	lat_deg_i = parse_int_field(f[8])
	lat_min = parse_fxx_2_field(f[9])
	lat_std_min = parse_fxx_2_field(f[10])

	lon_deg_i = parse_int_field(f[11])
	lon_min = parse_fxx_2_field(f[12])
	lon_std_min = parse_fxx_2_field(f[13])

	depth = parse_fxx_2_field(f[14])
	depth_std = parse_fxx_2_field(f[15])

	mag1 = parse_mag_field(f[16])
	mag1_type = f[17].strip()
	mag2 = parse_mag_field(f[18])
	mag2_type = f[19].strip()

	tt_table = f[20].strip()
	hypo_eval = f[21].strip()
	hypo_info = f[22].strip()
	max_int = f[23].strip()
	damage_scale = f[24].strip()
	tsunami = f[25].strip()
	region_large = f[26].strip()
	region_small = f[27].strip()
	epicenter = f[28].rstrip()
	station_cnt = parse_int_field(f[29])
	hypo_flag = f[30].strip()

	origin_time = build_origin_time(year, month, day, hour, minute, second)

	latitude_deg = None
	if lat_deg_i is not None:
		latitude_deg = float(lat_deg_i)
		if lat_min is not None:
			latitude_deg += lat_min / 60.0

	longitude_deg = None
	if lon_deg_i is not None:
		longitude_deg = float(lon_deg_i)
		if lon_min is not None:
			longitude_deg += lon_min / 60.0

	latitude_std_deg = lat_std_min / 60.0 if lat_std_min is not None else None
	longitude_std_deg = lon_std_min / 60.0 if lon_std_min is not None else None

	return {
		'record_type': header,
		'origin_time': origin_time,
		'origin_time_std_s': origin_std,
		'latitude_deg': latitude_deg,
		'latitude_std_deg': latitude_std_deg,
		'longitude_deg': longitude_deg,
		'longitude_std_deg': longitude_std_deg,
		'depth_km': depth,
		'depth_std_km': depth_std,
		'mag1': mag1,
		'mag1_type': mag1_type,
		'mag2': mag2,
		'mag2_type': mag2_type,
		'tt_table': tt_table,
		'hypocenter_eval': hypo_eval,
		'hypocenter_info': hypo_info,
		'max_intensity': max_int,
		'damage_scale': damage_scale,
		'tsunami_scale': tsunami,
		'region_large_code': region_large,
		'region_small_code': region_small,
		'epicenter_name': epicenter,
		'station_count': station_cnt,
		'hypocenter_flag': hypo_flag,
	}


def format_float(val, digits: int):
	if val is None:
		return ''
	return f'{val:.{digits}f}'


def convert_epicenter_to_csv(input_paths, output_path: str):
	"""複数の日別 input.txt を受け取り、
	全ファイルの震源レコードをまとめて1つの CSV に出力する。

	event_id は出力 CSV の行番号として、1 からの連番で付与する。
	"""
	dst = Path(output_path)
	with dst.open('w', encoding='utf-8', newline='') as fout:
		writer = csv.writer(fout)
		writer.writerow(EVENT_FIELD_NAMES)

		event_id = 1
		for input_path in tqdm(input_paths):
			try:
				src = Path(input_path)
				with src.open('r', encoding='utf-8', errors='strict') as fin:
					for raw_line in fin:
						line = raw_line.rstrip('\r\n')
						if not line:
							continue

						rec = parse_epicenter_line(line)
						if rec is None:
							continue

						row = [
							str(event_id),
							rec['record_type'],
							rec['origin_time'],
							format_float(rec['origin_time_std_s'], 2),
							format_float(rec['latitude_deg'], 6),
							format_float(rec['latitude_std_deg'], 6),
							format_float(rec['longitude_deg'], 6),
							format_float(rec['longitude_std_deg'], 6),
							format_float(rec['depth_km'], 2),
							format_float(rec['depth_std_km'], 2),
							format_float(rec['mag1'], 1),
							rec['mag1_type'],
							format_float(rec['mag2'], 1),
							rec['mag2_type'],
							rec['tt_table'],
							rec['hypocenter_eval'],
							rec['hypocenter_info'],
							rec['max_intensity'],
							rec['damage_scale'],
							rec['tsunami_scale'],
							rec['region_large_code'],
							rec['region_small_code'],
							rec['epicenter_name'],
							''
							if rec['station_count'] is None
							else str(rec['station_count']),
							rec['hypocenter_flag'],
						]
						writer.writerow(row)
						event_id += 1

			except Exception as e:
				print(f'Error processing file {input_path}: {e}')
				try:
					parse_epicenter_line(line, verbose=True)
				except Exception:
					continue


if __name__ == '__main__':
	input_dir = Path('../')
	input_files = sorted(input_dir.glob('*/arrivetime_*.txt'))
	output_csv = '../arrivetime_epicenters.csv'
	convert_epicenter_to_csv(input_files, output_csv)
