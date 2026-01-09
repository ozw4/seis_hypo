# %%
import csv
from datetime import datetime, timedelta
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
	5,  # 15 深さ（km） F5.2 相当だが、実データは " 50  " など 3桁整数 + 2桁小数とみなす
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

# 検測レコード（先頭 '_' や 'W' など）のフィールド定義
FIELD_WIDTHS_MEAS = [
	1,  # レコード種別ヘッダ
	6,  # 観測点コード
	4,  # 観測点番号
	1,  # 空白
	1,  # 地震計種別
	2,  # 発現時（日）
	4,  # 相名
	2,  # 発現時（時）
	2,  # 発現時（分）
	4,  # 発現時（秒）
	4,  # 相名
	2,  # 発現時（分）
	4,  # 発現時（秒）
	5,  # 最大振幅 (N-S)
	3,  # 最大振幅の周期
	3,  # 最大振幅の出現時刻
	5,  # 最大振幅 (E-W)
	3,  # 最大振幅の周期
	3,  # 最大振幅の出現時刻
	5,  # 最大振幅 (U-D)
	3,  # 最大振幅の周期
	3,  # 最大振幅の出現時刻
	1,  # 最大振幅の単位
	1,  # 初動方向 (N-S)
	3,  # 初動振幅 (N-S)
	1,  # 初動方向 (E-W)
	3,  # 初動振幅 (E-W)
	1,  # 初動方向 (U-D)
	3,  # 初動振幅 (U-D)
	1,  # 初動振幅の単位
	3,  # 継続時間
	2,  # 年
	2,  # 月
	1,  # 検測フラグ
	1,  # 検測フラグ
	1,  # 検測フラグ
	1,  # 検測フラグ
	1,  # 震源計算ウェイト
]

MEAS_FIELD_NAMES = [
	'event_id',
	'record_type_header',
	'station_code',
	'station_number',
	'blank',
	'sensor_type',
	'phase_name_1',
	'phase1_time',
	'phase_name_2',
	'phase2_time',
	'max_amplitude_ns',
	'max_period_ns',
	'max_time_ns',
	'max_amplitude_ew',
	'max_period_ew',
	'max_time_ew',
	'max_amplitude_ud',
	'max_period_ud',
	'max_time_ud',
	'max_amplitude_unit',
	'first_motion_dir_ns',
	'first_motion_amp_ns',
	'first_motion_dir_ew',
	'first_motion_amp_ew',
	'first_motion_dir_ud',
	'first_motion_amp_ud',
	'first_motion_unit',
	'duration',
	'pick_flag_1',
	'pick_flag_2',
	'pick_flag_3',
	'pick_flag_4',
	'hypocenter_weight',
]


def split_fixed_width(line: str, widths: list[int]) -> list[str]:
	fields: list[str] = []
	pos = 0
	for w in widths:
		fields.append(line[pos : pos + w])
		pos += w
	return fields


def parse_measure_line(line: str):
	"""検測レコード1行をフィールド配列にして返す。J/j/E 行は None。"""
	# 行が短い場合は右側をスペースでパディングして解釈を試みる
	if len(line) < RECORD_LENGTH:
		line = line.ljust(RECORD_LENGTH)
	elif len(line) > RECORD_LENGTH:
		raise ValueError(
			f'unexpected line length {len(line)} (expected {RECORD_LENGTH})'
		)

	header = line[0]
	# J, j, E は震源関連レコードなので検測としては無視
	if header != '_':
		return None

	return split_fixed_width(line, FIELD_WIDTHS_MEAS)


def parse_int_field(s: str):
	s = s.strip()
	return int(s) if s else None


def parse_fxx_2_field(s: str):
	# F4.2, F5.2, F3.2 のように小数点以下2桁を整数で持っているフィールド
	# 例: "5115" -> 51.15, " 390" -> 3.90
	s = s.strip()
	if not s:
		return None

	# 正常系：完全に数字だけ -> 2桁小数として /100.0
	if s.isdigit():
		return int(s) / 100.0

	# 「整数 + 空白 + 1桁小数」救済系
	# 例: "39 0" -> ["39", "0"] -> 39.0, "4 6" -> ["4", "6"] -> 4.6
	parts = s.split()
	if (
		len(parts) == 2
		and parts[0].isdigit()
		and parts[1].isdigit()
		and len(parts[1]) == 1
	):
		int_part = int(parts[0])
		frac = int(parts[1])
		return int_part + frac / 10.0

	# ここに来るのは本当に未知のフォーマット
	raise ValueError(f'invalid Fxx.2 field: {s!r}')


def parse_depth_field(s: str):
	"""深さ F5.2 フィールド用。
	5 文字を前半 3 桁（整数部）、後半 2 桁（小数部）として解釈する。
	例:
	" 50  " -> " 50" + "  " -> 50.0
	"03000" -> "030" + "00" -> 30.00
	"00525" -> "005" + "25" -> 5.25
	空白だけの場合は None。
	"""
	if s is None:
		return None

	# 深さフィールドは固定 5 文字前提
	if len(s) < 5:
		s = s.ljust(5)
	elif len(s) > 5:
		s = s[:5]

	int_part_raw = s[0:3]
	frac_part_raw = s[3:5]

	int_part_str = int_part_raw.strip()
	frac_part_str = frac_part_raw.strip()

	if not int_part_str:
		return None

	# 整数部は符号付き整数として解釈
	int_part = int(int_part_str)

	if frac_part_str:
		if not frac_part_str.isdigit():
			raise ValueError(f'invalid depth fractional digits: {s!r}')
		frac_part = int(frac_part_str)
	else:
		frac_part = 0

	if int_part < 0:
		return int_part - frac_part / 100.0
	return int_part + frac_part / 100.0


def parse_mag_field(s: str):
	"""F2.1 マグニチュード用フィールドを float に変換する。
	正値  : '43' -> 4.3
	負値は気象庁ルールでエンコードされている:
	-0.1 ～ -0.9 : '-1' ～ '-9'
	-1.0 ～ -1.9 : 'A0' ～ 'A9'
	-2.0         : 'B0'
	-3.0         : 'C0'
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


def build_phase_time(year_str, month_str, day_str, hour_str, minute_str, second_str):
	"""検測レコードの年月日時分秒から datetime を作る。
	年月日時分秒はすべて文字列（2桁年 + F4.2秒）。
	どれか欠損なら None を返す。
	"""
	year_str = year_str.strip()
	month_str = month_str.strip()
	day_str = day_str.strip()
	hour_str = hour_str.strip()
	minute_str = minute_str.strip()
	second_str = second_str.strip()

	if not (
		year_str and month_str and day_str and hour_str and minute_str and second_str
	):
		return None

	y2 = int(year_str)
	if y2 < 100:
		year = 1900 + y2 if y2 >= 70 else 2000 + y2
	else:
		year = y2

	month = int(month_str)
	day = int(day_str)
	hour = int(hour_str)
	minute = int(minute_str)

	sec = parse_fxx_2_field(second_str)
	if sec is None:
		return None

	base = datetime(year, month, day)
	total_seconds = hour * 3600 + minute * 60 + sec

	# total_seconds が 60 秒を超えても自動で日付・時刻を繰り上げる
	return base + timedelta(seconds=total_seconds)


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

	f = split_fixed_width(line, FIELD_WIDTHS)

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

	# 深さフィールドは F5.2 相当だが、実データは " 50  " などを
	# 「整数3桁 + 小数2桁」として扱うのが安定なので専用パーサを使う
	depth = parse_depth_field(f[14])
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
			src = Path(input_path)

			with src.open('r', encoding='utf-8', errors='strict') as fin:
				for raw_line in fin:
					line = raw_line.rstrip('\r\n')
					if not line:
						continue

					try:
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
					except Exception as e:
						# ★ このイベントだけをスキップ（警告出力）
						print(f'[WARN] skip event in {input_path}: {e} | line={line!r}')
						continue

					writer.writerow(row)
					event_id += 1


def convert_measure_to_csv(input_paths, output_path: str):
	"""複数の日別 input.txt から検測レコードを抜き出し、
	イベント CSV と同じ event_id を付けて1つの CSV にまとめる。
	検測の時刻は phase1_time / phase2_time の2カラムにまとめる。
	"""
	dst = Path(output_path)

	with dst.open('w', encoding='utf-8', newline='') as fout:
		writer = csv.writer(fout)
		writer.writerow(MEAS_FIELD_NAMES)

		event_id = 1

		for input_path in tqdm(input_paths):
			src = Path(input_path)
			current_event_id = None

			with src.open('r', encoding='utf-8', errors='strict') as fin:
				for raw_line in fin:
					line = raw_line.rstrip('\r\n')
					if not line:
						continue

					header = line[0]

					if header == 'J':
						# 震源レコードをパースして event_id を進める
						try:
							rec_ep = parse_epicenter_line(line)
							if rec_ep is None:
								current_event_id = None
							else:
								current_event_id = event_id
								event_id += 1
						except Exception as e:
							print(
								f'[WARN] skip event for measurements in {input_path}: {e} | line={line!r}'
							)
							current_event_id = None

					elif header == 'E':
						current_event_id = None

					else:
						# 検測レコード候補
						if current_event_id is None:
							continue

						try:
							fields = parse_measure_line(line)
							if fields is None:
								continue
						except Exception as e:
							print(
								f'[WARN] skip measurement in {input_path}: {e} | line={line!r}'
							)
							continue

						# インデックス対応：
						#  0: record_type_header
						#  1: station_code
						#  2: station_number
						#  3: blank
						#  4: sensor_type
						#  5: onset_day
						#  6: phase_name_1
						#  7: onset_hour_1
						#  8: onset_minute_1
						#  9: onset_second_1
						# 10: phase_name_2
						# 11: onset_minute_2
						# 12: onset_second_2
						# 13–21: 各種振幅・時刻
						# 22–29: 初動・単位
						# 30: duration
						# 31: year
						# 32: month
						# 33–36: pick_flag_1〜4
						# 37: hypocenter_weight

						record_type_header = fields[0].strip()
						station_code = fields[1].strip()
						station_number = fields[2].strip()
						blank = fields[3].strip()
						sensor_type = fields[4].strip()

						onset_day = fields[5].strip()
						phase_name_1 = fields[6].strip()
						onset_hour_1 = fields[7].strip()
						onset_minute_1 = fields[8].strip()
						onset_second_1 = fields[9].strip()

						phase_name_2 = fields[10].strip()
						onset_minute_2 = fields[11].strip()
						onset_second_2 = fields[12].strip()

						max_amplitude_ns = fields[13].strip()
						max_period_ns = fields[14].strip()
						max_time_ns = fields[15].strip()
						max_amplitude_ew = fields[16].strip()
						max_period_ew = fields[17].strip()
						max_time_ew = fields[18].strip()
						max_amplitude_ud = fields[19].strip()
						max_period_ud = fields[20].strip()
						max_time_ud = fields[21].strip()
						max_amplitude_unit = fields[22].strip()

						first_motion_dir_ns = fields[23].strip()
						first_motion_amp_ns = fields[24].strip()
						first_motion_dir_ew = fields[25].strip()
						first_motion_amp_ew = fields[26].strip()
						first_motion_dir_ud = fields[27].strip()
						first_motion_amp_ud = fields[28].strip()
						first_motion_unit = fields[29].strip()

						duration = fields[30].strip()
						year_str = fields[31].strip()
						month_str = fields[32].strip()
						pick_flag_1 = fields[33].strip()
						pick_flag_2 = fields[34].strip()
						pick_flag_3 = fields[35].strip()
						pick_flag_4 = fields[36].strip()
						hypocenter_weight = fields[37].strip()
						try:
							# P の datetime
							dt_p = build_phase_time(
								year_str,
								month_str,
								onset_day,
								onset_hour_1,
								onset_minute_1,
								onset_second_1,
							)

							# S の datetime（hour フィールドは共通）
							dt_s = build_phase_time(
								year_str,
								month_str,
								onset_day,
								onset_hour_1,
								onset_minute_2,
								onset_second_2,
							)

							# 文字列出力用に初期化
							phase1_time = ''
							phase2_time = ''

							if dt_p is not None:
								phase1_time = dt_p.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-4]

							if dt_s is not None:
								# P/S 両方あるときだけ「S は P より後」補正をかける
								if dt_p is not None and dt_s < dt_p:
									dt_s = dt_s + timedelta(hours=1)
								phase2_time = dt_s.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-4]

						except Exception as e:
							print(
								f'[WARN] skip measurement phase2_time build in {input_path}: {e} | line={line!r}'
							)
							continue

						row = [
							str(current_event_id),
							record_type_header,
							station_code,
							station_number,
							blank,
							sensor_type,
							phase_name_1,
							phase1_time,
							phase_name_2,
							phase2_time,
							max_amplitude_ns,
							max_period_ns,
							max_time_ns,
							max_amplitude_ew,
							max_period_ew,
							max_time_ew,
							max_amplitude_ud,
							max_period_ud,
							max_time_ud,
							max_amplitude_unit,
							first_motion_dir_ns,
							first_motion_amp_ns,
							first_motion_dir_ew,
							first_motion_amp_ew,
							first_motion_dir_ud,
							first_motion_amp_ud,
							first_motion_unit,
							duration,
							pick_flag_1,
							pick_flag_2,
							pick_flag_3,
							pick_flag_4,
							hypocenter_weight,
						]

						writer.writerow(row)


# %%
