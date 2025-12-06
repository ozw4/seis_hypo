# src/catalog/jma_phases.py
from collections.abc import Sequence

import pandas as pd

from common.core import validate_columns


def map_pick_weight(flags: Sequence[object]) -> int:
	for f in flags:
		if isinstance(f, str):
			v = f.strip().upper()
			if v == 'M':
				return 0
			if v == 'A':
				return 1
			if v == 'R':
				return 2
	return 0


def extract_phase_records(meas_df: pd.DataFrame) -> list[dict]:
	validate_columns(
		meas_df,
		[
			'event_id',
			'station_code',
			'phase_name_1',
			'phase_name_2',
			'phase1_time',
			'phase2_time',
			'pick_flag_1',
			'pick_flag_2',
			'pick_flag_3',
			'pick_flag_4',
		],
		'measurements CSV',
	)

	records: list[dict] = []

	for _, row in meas_df.iterrows():
		eid = int(row['event_id'])

		phase1_raw = row['phase_name_1']
		phase2_raw = row['phase_name_2']

		phase1 = phase1_raw.strip().upper() if isinstance(phase1_raw, str) else ''
		phase2 = phase2_raw.strip().upper() if isinstance(phase2_raw, str) else ''

		t1 = None
		t2 = None

		v1 = row['phase1_time']
		if pd.notna(v1):
			t1 = pd.to_datetime(v1)

		v2 = row['phase2_time']
		if pd.notna(v2):
			t2 = pd.to_datetime(v2)

		flags = [
			row['pick_flag_1'],
			row['pick_flag_2'],
			row['pick_flag_3'],
			row['pick_flag_4'],
		]

		# P フェーズ（1 本目）
		if 'P' in phase1 and 'S' not in phase1 and t1 is not None:
			w = map_pick_weight(flags)
			records.append(
				{
					'event_id': eid,
					'station_code': row['station_code'],
					'phase_type': 'P',
					'weight': w,
					'time': t1,
				}
			)

		# S フェーズ（2 本目優先）
		if 'S' in phase2 and 'P' not in phase2 and t2 is not None:
			flags_s = [
				row['pick_flag_2'],
				row['pick_flag_1'],
				row['pick_flag_3'],
				row['pick_flag_4'],
			]
			w = map_pick_weight(flags_s)
			records.append(
				{
					'event_id': eid,
					'station_code': row['station_code'],
					'phase_type': 'S',
					'weight': w,
					'time': t2,
				}
			)
		elif 'S' in phase1 and 'P' not in phase1 and t1 is not None:
			w = map_pick_weight(flags)
			records.append(
				{
					'event_id': eid,
					'station_code': row['station_code'],
					'phase_type': 'S',
					'weight': w,
					'time': t1,
				}
			)

	return records
