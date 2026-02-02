# src/hypo/arc.py
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hypo.phase_jma import extract_phase_records
from hypo.station_meta import build_station_meta, parse_station_code


def format_depth_f5_for_hypo(depth_km: float) -> str:
	"""Format a depth value as a 5-character HypoInverse depth field (F5).

	Parameters
	----------
	depth_km : float
		Depth in kilometers. Must be finite and non-negative, and must fit in a 5-char
		field according to the formatting rules below.

	Returns
	-------
	str
		A 5-character string suitable for HypoInverse ARC headers/terminators.

	Raises
	------
	ValueError
		If `depth_km` is negative, too large to fit in 5 characters, or the formatted
		result is not exactly 5 characters.

	"""
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
	phases: list[dict[str, Any]],
	station_csv: str | Path,
	output_arc: str | Path,
	*,
	default_depth_km: float = 10.0,
	use_jma_flag: bool = False,
	p_centroid_top_n: int = 5,
	origin_time_offset_sec: float = 3.0,
	fix_depth: bool = False,
) -> None:
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

	by_event: dict[int, list[dict[str, Any]]] = defaultdict(list)
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
			depth_str = format_depth_f5_for_hypo(depth_out)

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

			trial_depth_str = format_depth_f5_for_hypo(depth)
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
