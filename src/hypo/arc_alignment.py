import pandas as pd


# ========= JMA 側 (epicenter / measurement) から arc と同じ順番を再現 =========
def _collect_phase_flags(meas_df: pd.DataFrame) -> dict[int, dict[str, bool]]:
	"""JMA 検測 CSV から event_id ごとの
	・any_phase: 少なくとも1本以上の有効フェーズがあるか
	・has_p    : 少なくとも1本以上の P があるか
	を、write_hypoinverse_arc() と同じ条件で集計する。
	"""
	flags: dict[int, dict[str, bool]] = {}

	for _, row in meas_df.iterrows():
		if 'event_id' not in row:
			raise ValueError('meas_df に event_id カラムがありません')

		eid = int(row['event_id'])

		if eid not in flags:
			flags[eid] = {'any_phase': False, 'has_p': False}

		phase1 = (
			str(row['phase_name_1']).strip().upper()
			if isinstance(row.get('phase_name_1'), str)
			else ''
		)
		phase2 = (
			str(row['phase_name_2']).strip().upper()
			if isinstance(row.get('phase_name_2'), str)
			else ''
		)

		t1_raw = row.get('phase1_time')
		t2_raw = row.get('phase2_time')
		t1 = str(t1_raw).strip() if isinstance(t1_raw, str) else ''
		t2 = str(t2_raw).strip() if isinstance(t2_raw, str) else ''

		any_phase = False

		# P フェーズ（extract_phase_records と同じ条件）
		if 'P' in phase1 and 'S' not in phase1 and t1:
			flags[eid]['has_p'] = True
			any_phase = True

		# S フェーズ（2本目優先）
		if ('S' in phase2 and 'P' not in phase2 and t2) or (
			'S' in phase1 and 'P' not in phase1 and t1
		):
			any_phase = True

		if any_phase:
			flags[eid]['any_phase'] = True

	return flags


def build_arc_event_map(
	epic_df: pd.DataFrame,
	meas_df: pd.DataFrame,
) -> pd.DataFrame:
	"""JMA epicenter / measurement DataFrame から、
	hypoinverse_input.arc に書かれたのと同じ順番のイベント一覧を作る。

	戻り値カラム:
	  seq              : hypoinverse / .arc 内の通し番号 (1..N)
	  event_id         : JMA 側 event_id
	  origin_time_jma  : Timestamp
	  lat_deg_jma      : 緯度 (deg)
	  lon_deg_jma      : 経度 (deg)
	  depth_km_jma     : 深さ (km, NaN は 0.0 に揃える)
	  mag1_jma         : JMA 側 mag1
	  mag1_type_jma    : JMA 側 mag1_type
	  mag2_jma         : JMA 側 mag2
	  mag2_type_jma    : JMA 側 mag2_type
	"""
	if 'event_id' not in epic_df.columns:
		raise ValueError('epic_df に event_id カラムがありません')
	if 'origin_time' not in epic_df.columns:
		raise ValueError('epic_df に origin_time カラムがありません')

	epic = epic_df.copy()
	epic['origin_dt'] = pd.to_datetime(epic['origin_time'])
	epic = epic.sort_values('origin_dt').reset_index(drop=True)

	phase_flags = _collect_phase_flags(meas_df)

	rows: list[dict] = []
	seq = 1

	for _, ev in epic.iterrows():
		eid = int(ev['event_id'])
		flag = phase_flags.get(eid)

		if flag is None or not flag['any_phase']:
			continue
		if not flag['has_p']:
			continue

		lat = ev.get('latitude_deg')
		lon = ev.get('longitude_deg')
		if pd.isna(lat) or pd.isna(lon):
			continue

		depth = ev.get('depth_km')
		depth_val = 0.0 if pd.isna(depth) else float(depth)

		mag1 = ev.get('mag1') if 'mag1' in ev else None
		mag1_type = ev.get('mag1_type') if 'mag1_type' in ev else None
		mag2 = ev.get('mag2') if 'mag2' in ev else None
		mag2_type = ev.get('mag2_type') if 'mag2_type' in ev else None

		rows.append(
			{
				'seq': seq,
				'event_id': eid,
				'origin_time_jma': pd.to_datetime(ev['origin_time']),
				'lat_deg_jma': float(lat),
				'lon_deg_jma': float(lon),
				'depth_km_jma': depth_val,
				'mag1_jma': mag1,
				'mag1_type_jma': mag1_type,
				'mag2_jma': mag2,
				'mag2_type_jma': mag2_type,
			}
		)
		seq += 1

	if not rows:
		raise RuntimeError('arc に相当するイベントが 1 件も見つかりませんでした')

	return pd.DataFrame(rows)
