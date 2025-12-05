# %%
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from util import validate_columns


def parse_station_code(code: str):
	s = str(code).strip().upper()
	if not s:
		return None, None
	sta = s[:5]
	if not sta:
		return None, None
	net = s[5] if len(s) == 6 else ''
	return sta, net


def build_station_meta(station_csv: str):
	station_df = pd.read_csv(station_csv)
	validate_columns(
		station_df,
		['station_code', 'Latitude_deg', 'Longitude_deg'],
		'station CSV',
	)
	meta = {}
	for _, row in station_df.iterrows():
		code = str(row['station_code']).strip()
		if not code:
			continue
		sta, net = parse_station_code(code)
		if sta is None:
			continue

		lat = row['Latitude_deg']
		lon = row['Longitude_deg']
		lat_val = float(lat) if pd.notna(lat) else None
		lon_val = float(lon) if pd.notna(lon) else None

		meta[code] = {
			'net': net,
			'sta': sta,
			'lat': lat_val,
			'lon': lon_val,
		}
	return meta


def map_pick_weight(flags):
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


def extract_phase_records(meas_df: pd.DataFrame):
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

	records = []

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


def format_depth_f5_for_hypo(depth_km: float) -> str:
	if depth_km < 0:
		raise ValueError(f'depth_km は負ではいけません: {depth_km}')

	if depth_km < 100.0:
		s = f'{depth_km:5.2f}'
	elif depth_km < 1000.0:
		s = f'{depth_km:5.1f}'
	else:
		raise ValueError(f'depth_km が大きすぎて 5桁に収まりません: {depth_km}')

	if len(s) != 5:
		raise ValueError(f'F5 フィールド長が 5 ではありません: "{s}" (len={len(s)})')

	return s


def write_hypoinverse_arc_from_phases(
	epic_df: pd.DataFrame,
	phases: list[dict],
	station_csv: str | Path,
	output_arc: str | Path,
	*,
	default_depth_km: float = 10.0,
	use_jma_flag: bool = False,
	p_centroid_top_n: int = 5,
	origin_time_offset_sec: float = 3.0,
	fix_depth: bool = False,
) -> None:
	"""epic_df + 任意の phases(list[dict]) から ARC を書き出す汎用版。

	phases は以下のキーを持つ dict のリストを想定:
	    - event_id     : int (JMA event_id)
	    - station_code : str
	    - phase_type   : 'P' or 'S'
	    - weight       : int
	    - time         : pd.Timestamp

	他のロジックは既存 write_hypoinverse_arc と同じ。
	"""
	required_epic_cols = {
		'event_id',
		'origin_time',
		'latitude_deg',
		'longitude_deg',
		'depth_km',
	}
	missing = required_epic_cols.difference(epic_df.columns)
	if missing:
		raise ValueError(f'epic_df に必要な列がありません: {missing}')

	if not phases:
		raise RuntimeError('phases にフェーズが 1 本もありません')

	station_meta = build_station_meta(station_csv)

	events = epic_df.copy()
	events['origin_dt'] = pd.to_datetime(events['origin_time'])
	events = events.sort_values('origin_dt').reset_index(drop=True)

	by_event: dict[int, list[dict]] = defaultdict(list)
	for r in phases:
		eid = int(r['event_id'])
		by_event[eid].append(r)

	out_path = Path(output_arc)
	with out_path.open('w', encoding='ascii', newline='\n') as f:
		for _, ev in events.iterrows():
			eid = int(ev['event_id'])
			picks = by_event.get(eid)
			if not picks:
				continue

			p_picks = [r for r in picks if str(r['phase_type']).upper() == 'P']
			if not p_picks:
				continue

			lat_jma = float(ev['latitude_deg'])
			lon_jma = float(ev['longitude_deg'])
			depth_jma = float(ev['depth_km'])
			origin_jma = pd.to_datetime(ev['origin_time'])

			# --- 初期震源の決め方 ---
			if use_jma_flag:
				if not np.isfinite(lat_jma) or not np.isfinite(lon_jma):
					print(f'[WARN] event_id={eid}: JMA 緯度経度が不正なのでスキップ')
					continue
				if not np.isfinite(depth_jma):
					print(f'[WARN] event_id={eid}: JMA 深さが不正なのでスキップ')
					continue
				if pd.isna(origin_jma):
					print(
						f'[WARN] event_id={eid}: JMA origin_time が不正なのでスキップ'
					)
					continue
				origin_dt = origin_jma
				lat0 = lat_jma
				lon0 = lon_jma
				depth = depth_jma
			else:
				p_picks_sorted = sorted(
					p_picks,
					key=lambda r: (r['time'], str(r['station_code'])),
				)
				first_p_time = p_picks_sorted[0]['time']
				origin_dt = first_p_time - pd.Timedelta(
					seconds=float(origin_time_offset_sec)
				)

				top_picks = p_picks_sorted[: int(p_centroid_top_n)]
				lat_list: list[float] = []
				lon_list: list[float] = []

				for r in top_picks:
					full = str(r['station_code']).strip()
					meta = station_meta.get(full)
					if meta is None:
						continue
					lat_sta = meta.get('lat')
					lon_sta = meta.get('lon')
					if lat_sta is None or lon_sta is None:
						continue
					if pd.isna(lat_sta) or pd.isna(lon_sta):
						continue
					lat_list.append(float(lat_sta))
					lon_list.append(float(lon_sta))

				if not lat_list or not lon_list:
					print(
						f'[WARN] event_id={eid}: 局座標から初期震央が求められないのでスキップ'
					)
					continue

				lat0 = float(np.mean(lat_list))
				lon0 = float(np.mean(lon_list))
				depth = float(default_depth_km)

			if not np.isfinite(depth) or depth < 0:
				print(
					f'[WARN] event_id={eid}: 深さ値が不正なのでスキップ depth={depth}'
				)
				continue

			depth_out = float(depth)
			try:
				depth_str = format_depth_f5_for_hypo(depth_out)
			except ValueError as e:
				print(f'[WARN] event_id={eid}: 深さフォーマット失敗でスキップ: {e}')
				continue

			dt0 = origin_dt
			event_time = dt0.strftime('%Y%m%d%H%M%S%f')[:-4]

			lat_deg = int(abs(lat0))
			lat_min = (abs(lat0) - lat_deg) * 60 * 100
			south = 'S' if lat0 < 0 else ' '

			lon_deg = int(abs(lon0))
			lon_min = (abs(lon0) - lon_deg) * 60 * 100
			east = 'E' if lon0 >= 0 else 'W'

			header = (
				f'{event_time}'
				f'{lat_deg:2d}{south}{lat_min:4.0f}'
				f'{lon_deg:3d}{east}{lon_min:4.0f}'
				f'{depth_str}'
			)
			f.write(header + '\n')

			picks_sorted = sorted(
				picks, key=lambda r: (r['time'], str(r['station_code']))
			)
			for r in picks_sorted:
				full = str(r['station_code']).strip()
				meta = station_meta.get(full)
				if meta is not None:
					sta = meta['sta']
					net = meta['net']
				else:
					sta, net = parse_station_code(full)
					if sta is None:
						continue

				comp_code = ''
				channel_code = 'HHZ'
				sta_block = f'{sta:<5}{net:<2} {comp_code:<1}{channel_code:<3}'

				t = r['time']
				minute_str = t.strftime('%Y%m%d%H%M')
				sec_str = t.strftime('%S%f')[:-4]

				w = int(r['weight'])

				if str(r['phase_type']).upper() == 'P':
					line = f'{sta_block:<13} P {w:<1d}{minute_str} {sec_str}'
				else:
					line = (
						f'{sta_block:<13}   4{minute_str} {"":<12}{sec_str} S {w:<1d}'
					)

				f.write(line + '\n')

			# terminator 行: 深さフリーでも常に trial depth を渡す
			try:
				trial_depth_str = format_depth_f5_for_hypo(depth)
			except ValueError as e:
				print(f'[WARN] event_id={eid}: terminator 用深さフォーマット失敗: {e}')
				f.write('\n')
				continue

			col35 = '-' if fix_depth else ' '
			terminator = ' ' * 29 + trial_depth_str + col35
			event_id_str = str(eid)[-10:]
			terminator = terminator.ljust(62) + f'{event_id_str:>10s}'
			f.write(terminator + '\n')

			f.write('\n')


def write_hypoinverse_arc(
	epic_df: pd.DataFrame,
	meas_df: pd.DataFrame,
	station_csv: str | Path,
	output_arc: str | Path,
	*,
	default_depth_km: float = 10.0,
	use_jma_flag: bool = False,
	p_centroid_top_n: int = 5,
	origin_time_offset_sec: float = 3.0,
	fix_depth: bool = False,
) -> None:
	phases = extract_phase_records(meas_df)
	return write_hypoinverse_arc_from_phases(
		epic_df,
		phases,
		station_csv,
		output_arc,
		default_depth_km=default_depth_km,
		use_jma_flag=use_jma_flag,
		p_centroid_top_n=p_centroid_top_n,
		origin_time_offset_sec=origin_time_offset_sec,
		fix_depth=fix_depth,
	)


if __name__ == '__main__':
	epcs_df = pd.read_csv('/workspace/data/arrivetime/arrivetime_epicenters_1day.csv')
	meas_df = pd.read_csv('/workspace/data/arrivetime/arrivetime_measurements_1day.csv')

	write_hypoinverse_arc(
		epcs_df,
		meas_df,
		'/workspace/data/station/station.csv',
		'hypoinverse_input.arc',
		default_depth_km=10.0,
		use_jma_flag=False,
		p_centroid_top_n=5,
		origin_time_offset_sec=3.0,
	)
