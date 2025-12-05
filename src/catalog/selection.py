import pandas as pd

from common.geo import haversine_distance_km


def extract_events_in_region(
	epic_df: pd.DataFrame,
	meas_df: pd.DataFrame | None,
	*,
	start_time: str | pd.Timestamp | None,
	end_time: str | pd.Timestamp | None,
	mag_min: float | None,
	mag_max: float | None,
	center_lat: float,
	center_lon: float,
	radius_km: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
	"""epic_df を時刻・マグ・空間条件でフィルタし、
	対応する event_id を持つ meas_df も同時に絞る.

	条件:
	- start_time が None でなければ origin_time >= start_time
	- end_time が None でなければ origin_time <= end_time
	- mag_min が None でなければ mag1 >= mag_min
	- mag_max が None でなければ mag1 <= mag_max
	- radius_km が None でなければ (center_lat, center_lon) から radius_km 以内

	戻り値:
	(epic_sub, meas_sub)
	meas_df が None のとき meas_sub は None
	"""  # noqa: D205
	required_cols = {
		'event_id',
		'origin_time',
		'latitude_deg',
		'longitude_deg',
		'depth_km',
		'mag1',
	}
	missing = [c for c in required_cols if c not in epic_df.columns]
	if missing:
		raise ValueError(f'epic_df に必要な列がありません: {missing}')

	df = epic_df.copy()

	df['origin_time_dt'] = pd.to_datetime(df['origin_time'])
	df['mag1'] = df['mag1'].astype(float)

	mask = pd.Series(True, index=df.index)

	if start_time is not None:
		t_start = pd.to_datetime(start_time)
		mask &= df['origin_time_dt'] >= t_start

	if end_time is not None:
		t_end = pd.to_datetime(end_time)
		mask &= df['origin_time_dt'] <= t_end

	if mag_min is not None:
		mask &= df['mag1'] >= float(mag_min)

	if mag_max is not None:
		mask &= df['mag1'] <= float(mag_max)

	if radius_km is not None:
		lat = df['latitude_deg'].to_numpy(dtype=float)
		lon = df['longitude_deg'].to_numpy(dtype=float)
		dist_km = haversine_distance_km(center_lat, center_lon, lat, lon)
		mask &= dist_km <= float(radius_km)

	epic_sub = df.loc[mask].copy()
	epic_sub = epic_sub.drop(columns=['origin_time_dt'])

	if epic_sub.empty:
		raise RuntimeError('指定した条件を満たすイベントがありません')

	if meas_df is None:
		return epic_sub, None

	if 'event_id' not in meas_df.columns:
		raise ValueError('meas_df に event_id 列がありません')

	ids = epic_sub['event_id'].unique()
	meas_sub = meas_df[meas_df['event_id'].isin(ids)].copy()

	# meas 側に 1 件も無い場合は気づきたいのでエラーにする
	if meas_sub.empty:
		raise RuntimeError('measurement 側に対応する event_id がありません')

	return epic_sub, meas_sub


def filter_by_das_score(
	epic_df: pd.DataFrame,
	meas_df: pd.DataFrame | None = None,
	das_min: float | None = None,
	das_max: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
	"""epic_df / meas_df を das_score でフィルタする.

	- 両方 None のときはそのまま返す
	- どちらか指定されていれば [das_min, das_max] 範囲で絞る
	  （「ちょうどこの値だけ」は das_min=das_max=その値 で表現）
	- meas_df が None の場合は epic_df のみフィルタして (epic_sub, None) を返す
	"""
	if das_min is None and das_max is None:
		return epic_df, meas_df

	if 'das_score' not in epic_df.columns:
		raise ValueError('epic_df に das_score 列がありません')
	if 'event_id' not in epic_df.columns:
		raise ValueError('epic_df に event_id 列がありません')
	if meas_df is not None and 'event_id' not in meas_df.columns:
		raise ValueError('meas_df に event_id 列がありません')

	mask = pd.Series(True, index=epic_df.index)
	if das_min is not None:
		mask &= epic_df['das_score'] >= float(das_min)
	if das_max is not None:
		mask &= epic_df['das_score'] <= float(das_max)

	epic_sub = epic_df[mask].copy()
	if epic_sub.empty:
		raise RuntimeError('指定した DAS 条件を満たすイベントがありません')

	if meas_df is None:
		return epic_sub, None

	ids = epic_sub['event_id'].unique()
	meas_sub = meas_df[meas_df['event_id'].isin(ids)].copy()
	if meas_sub.empty:
		raise RuntimeError('measurement 側に対応する event_id がありません')

	return epic_sub, meas_sub
