# ruff: noqa: ANN001, ANN201, C901, D100, D103, DTZ001, E501, INP001, PLR0911, PLR0913, TC003
from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator
from datetime import datetime, timedelta
from pathlib import Path

RECORD_LENGTH = 96

FIELD_WIDTHS = [
	1,
	4,
	2,
	2,
	2,
	2,
	4,
	4,
	3,
	4,
	4,
	4,
	4,
	4,
	5,
	3,
	2,
	1,
	2,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	3,
	24,
	3,
	1,
]

EPICENTER_HEADERS = {'J'}
MEASUREMENT_HEADERS = {'_', 'M'}

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

FIELD_WIDTHS_MEAS = [
	1,
	6,
	4,
	1,
	1,
	2,
	4,
	2,
	2,
	4,
	4,
	2,
	4,
	5,
	3,
	3,
	5,
	3,
	3,
	5,
	3,
	3,
	1,
	1,
	3,
	1,
	3,
	1,
	3,
	1,
	3,
	2,
	2,
	1,
	1,
	1,
	1,
	1,
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
	for width in widths:
		fields.append(line[pos : pos + width])
		pos += width
	return fields


def parse_int_field(value: str):
	value = value.strip()
	return int(value) if value else None


def parse_fxx_2_field(value: str):
	value = value.strip()
	if not value:
		return None

	if value.isdigit():
		return int(value) / 100.0

	parts = value.split()
	if (
		len(parts) == 2
		and parts[0].isdigit()
		and parts[1].isdigit()
		and len(parts[1]) == 1
	):
		return int(parts[0]) + int(parts[1]) / 10.0

	raise ValueError(f'invalid Fxx.2 field: {value!r}')


def parse_depth_field(value: str):
	if value is None:
		return None
	if len(value) < 5:
		value = value.ljust(5)
	elif len(value) > 5:
		value = value[:5]

	int_part_str = value[0:3].strip()
	frac_part_str = value[3:5].strip()
	if not int_part_str:
		return None

	int_part = int(int_part_str)
	frac_part = 0
	if frac_part_str:
		if not frac_part_str.isdigit():
			raise ValueError(f'invalid depth fractional digits: {value!r}')
		frac_part = int(frac_part_str)

	if int_part < 0:
		return int_part - frac_part / 100.0
	return int_part + frac_part / 100.0


def parse_mag_field(value: str):
	value = value.strip()
	if not value:
		return None
	if len(value) == 1 and value.isdigit():
		return float(value)
	if len(value) != 2:
		raise ValueError(f'unexpected F2.1 field length: {value!r}')

	head = value[0]
	tail = value[1]
	if head == '-' and tail.isdigit():
		return -0.1 * int(tail)
	if head == 'A' and tail.isdigit():
		return -1.0 - 0.1 * int(tail)
	if head == 'B' and tail == '0':
		return -2.0
	if head == 'C' and tail == '0':
		return -3.0
	if value.isdigit():
		return int(value) / 10.0

	raise ValueError(f'invalid F2.1 magnitude code: {value!r}')


def build_origin_time(year, month, day, hour, minute, second):
	if None in (year, month, day, hour, minute, second):
		return ''
	second_str = f'{second:.2f}'
	if second < 10.0 and not second_str.startswith('0'):
		second_str = '0' + second_str
	return f'{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second_str}'


def build_phase_time(
	year_str: str,
	month_str: str,
	day_str: str,
	hour_str: str,
	minute_str: str,
	second_str: str,
):
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

	year = int(year_str)
	if year < 100:
		year = 1900 + year if year >= 70 else 2000 + year

	base = datetime(year, int(month_str), int(day_str))
	total_seconds = (
		int(hour_str) * 3600
		+ int(minute_str) * 60
		+ parse_fxx_2_field(second_str)
	)
	return base + timedelta(seconds=total_seconds)


def parse_epicenter_line(line: str):
	if len(line) != RECORD_LENGTH:
		raise ValueError(
			f'unexpected line length {len(line)} (expected {RECORD_LENGTH})'
		)

	header = line[0]
	if header not in EPICENTER_HEADERS:
		return None

	fields = split_fixed_width(line, FIELD_WIDTHS)
	year = parse_int_field(fields[1])
	month = parse_int_field(fields[2])
	day = parse_int_field(fields[3])
	hour = parse_int_field(fields[4])
	minute = parse_int_field(fields[5])
	second = parse_fxx_2_field(fields[6])
	origin_std = parse_fxx_2_field(fields[7])

	lat_deg_i = parse_int_field(fields[8])
	lat_min = parse_fxx_2_field(fields[9])
	lat_std_min = parse_fxx_2_field(fields[10])
	lon_deg_i = parse_int_field(fields[11])
	lon_min = parse_fxx_2_field(fields[12])
	lon_std_min = parse_fxx_2_field(fields[13])

	depth = parse_depth_field(fields[14])
	depth_std = parse_fxx_2_field(fields[15])

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

	return {
		'record_type': header,
		'origin_time': build_origin_time(year, month, day, hour, minute, second),
		'origin_time_std_s': origin_std,
		'latitude_deg': latitude_deg,
		'latitude_std_deg': lat_std_min / 60.0 if lat_std_min is not None else None,
		'longitude_deg': longitude_deg,
		'longitude_std_deg': lon_std_min / 60.0 if lon_std_min is not None else None,
		'depth_km': depth,
		'depth_std_km': depth_std,
		'mag1': parse_mag_field(fields[16]),
		'mag1_type': fields[17].strip(),
		'mag2': parse_mag_field(fields[18]),
		'mag2_type': fields[19].strip(),
		'tt_table': fields[20].strip(),
		'hypocenter_eval': fields[21].strip(),
		'hypocenter_info': fields[22].strip(),
		'max_intensity': fields[23].strip(),
		'damage_scale': fields[24].strip(),
		'tsunami_scale': fields[25].strip(),
		'region_large_code': fields[26].strip(),
		'region_small_code': fields[27].strip(),
		'epicenter_name': fields[28].rstrip(),
		'station_count': parse_int_field(fields[29]),
		'hypocenter_flag': fields[30].strip(),
	}


def parse_measure_line(line: str):
	if len(line) < RECORD_LENGTH:
		line = line.ljust(RECORD_LENGTH)
	elif len(line) > RECORD_LENGTH:
		raise ValueError(
			f'unexpected line length {len(line)} (expected {RECORD_LENGTH})'
		)

	if line[0] not in MEASUREMENT_HEADERS:
		return None

	return split_fixed_width(line, FIELD_WIDTHS_MEAS)


def format_float(value, digits: int):
	if value is None:
		return ''
	return f'{value:.{digits}f}'


def format_datetime(value: datetime | None) -> str:
	if value is None:
		return ''
	return value.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-4]


def build_measurement_row(event_id: int, line: str) -> list[str] | None:
	fields = parse_measure_line(line)
	if fields is None:
		return None

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
	year_str = fields[31].strip()
	month_str = fields[32].strip()

	phase1_time = build_phase_time(
		year_str,
		month_str,
		onset_day,
		onset_hour_1,
		onset_minute_1,
		onset_second_1,
	)
	phase2_time = build_phase_time(
		year_str,
		month_str,
		onset_day,
		onset_hour_1,
		onset_minute_2,
		onset_second_2,
	)
	if phase1_time is not None and phase2_time is not None and phase2_time < phase1_time:
		phase2_time = phase2_time + timedelta(hours=1)

	return [
		str(event_id),
		record_type_header,
		station_code,
		station_number,
		blank,
		sensor_type,
		phase_name_1,
		format_datetime(phase1_time),
		phase_name_2,
		format_datetime(phase2_time),
		fields[13].strip(),
		fields[14].strip(),
		fields[15].strip(),
		fields[16].strip(),
		fields[17].strip(),
		fields[18].strip(),
		fields[19].strip(),
		fields[20].strip(),
		fields[21].strip(),
		fields[22].strip(),
		fields[23].strip(),
		fields[24].strip(),
		fields[25].strip(),
		fields[26].strip(),
		fields[27].strip(),
		fields[28].strip(),
		fields[29].strip(),
		fields[30].strip(),
		fields[33].strip(),
		fields[34].strip(),
		fields[35].strip(),
		fields[36].strip(),
		fields[37].strip(),
	]


def build_epicenter_row(event_id: int, record: dict[str, object]) -> list[str]:
	station_count = record['station_count']
	return [
		str(event_id),
		record['record_type'],
		record['origin_time'],
		format_float(record['origin_time_std_s'], 2),
		format_float(record['latitude_deg'], 6),
		format_float(record['latitude_std_deg'], 6),
		format_float(record['longitude_deg'], 6),
		format_float(record['longitude_std_deg'], 6),
		format_float(record['depth_km'], 2),
		format_float(record['depth_std_km'], 2),
		format_float(record['mag1'], 1),
		record['mag1_type'],
		format_float(record['mag2'], 1),
		record['mag2_type'],
		record['tt_table'],
		record['hypocenter_eval'],
		record['hypocenter_info'],
		record['max_intensity'],
		record['damage_scale'],
		record['tsunami_scale'],
		record['region_large_code'],
		record['region_small_code'],
		record['epicenter_name'],
		'' if station_count is None else str(station_count),
		record['hypocenter_flag'],
	]


def iter_arrivetime_event_records(
	input_paths: Iterable[str | Path],
) -> Iterator[tuple[int, dict[str, object], list[list[str]]]]:
	event_id = 1
	for input_path in input_paths:
		src = Path(input_path)
		current_event_id = None
		current_epicenter = None
		current_measurements: list[list[str]] = []

		with src.open('r', encoding='utf-8', errors='strict') as fin:
			for raw_line in fin:
				line = raw_line.rstrip('\r\n')
				if not line:
					continue

				header = line[0]
				if header in EPICENTER_HEADERS:
					if current_event_id is not None and current_epicenter is not None:
						yield current_event_id, current_epicenter, current_measurements

					current_event_id = event_id
					current_epicenter = parse_epicenter_line(line)
					current_measurements = []
					event_id += 1
					continue

				if header == 'E':
					if current_event_id is not None and current_epicenter is not None:
						yield current_event_id, current_epicenter, current_measurements
					current_event_id = None
					current_epicenter = None
					current_measurements = []
					continue

				if current_event_id is None:
					continue

				measurement_row = build_measurement_row(current_event_id, line)
				if measurement_row is not None:
					current_measurements.append(measurement_row)

		if current_event_id is not None and current_epicenter is not None:
			yield current_event_id, current_epicenter, current_measurements


def convert_epicenter_to_csv(input_paths, output_path: str | Path):
	dst = Path(output_path)
	dst.parent.mkdir(parents=True, exist_ok=True)
	with dst.open('w', encoding='utf-8', newline='') as fout:
		writer = csv.writer(fout)
		writer.writerow(EVENT_FIELD_NAMES)
		for event_id, epicenter_record, _ in iter_arrivetime_event_records(input_paths):
			writer.writerow(build_epicenter_row(event_id, epicenter_record))


def convert_measure_to_csv(input_paths, output_path: str | Path):
	dst = Path(output_path)
	dst.parent.mkdir(parents=True, exist_ok=True)
	with dst.open('w', encoding='utf-8', newline='') as fout:
		writer = csv.writer(fout)
		writer.writerow(MEAS_FIELD_NAMES)
		for _, _, measurement_rows in iter_arrivetime_event_records(input_paths):
			for row in measurement_rows:
				writer.writerow(row)
