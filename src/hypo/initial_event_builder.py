import pandas as pd


def build_initial_events_from_ml_picks(
	eqt_df: pd.DataFrame,
	station_df: pd.DataFrame,
	depth_km: float = 10.0,
	mode: str = 'first_p',  # 'network_center' or 'first_p'
) -> pd.DataFrame:
	"""EqTピックCSVとstation_dfから、自前イベント表(df_epic_eqt)を作る。

	Hypoinverse 用に、列名は
		event_id, origin_time, latitude_deg, longitude_deg, depth_km
	にそろえる。

	Parameters
	----------
	eqt_df : DataFrame
		EqTピックCSV。
		必須列: event_id, station_code, Phase, pick_time, event_time_peak
	station_df : DataFrame
		局情報。必須列: station, lat, lon
	depth_km : float, default 10.0
		全イベント共通の初期深さ[km]。
	mode : {'network_center', 'first_p'}
		'network_center' :
			全イベントの緯度経度をネットワーク中心(全局平均)にする。
		'first_p' :
			各イベントの最初のPピック局の座標を使い、
			Pが無いイベントはネットワーク中心でフォールバックする。

	"""
	required_eqt_cols = [
		'event_id',
		'station_code',
		'Phase',
		'pick_time',
		'event_time_peak',
	]
	for c in required_eqt_cols:
		if c not in eqt_df.columns:
			raise ValueError(f"eqt_df に '{c}' 列がありません。")

	required_sta_cols = ['station', 'lat', 'lon']
	for c in required_sta_cols:
		if c not in station_df.columns:
			raise ValueError(f"station_df に '{c}' 列がありません。")

	eqt_df = eqt_df.copy()
	eqt_df['pick_time'] = pd.to_datetime(eqt_df['pick_time'])
	eqt_df['event_time_peak'] = pd.to_datetime(eqt_df['event_time_peak'])

	# event_id ごとの代表時刻（とりあえず event_time_peak の先頭）
	g = eqt_df.groupby('event_id', sort=False)
	origin_time_all = g['event_time_peak'].first()  # index: 全 event_id
	idx_all = origin_time_all.index

	# ネットワーク中心（フォールバック用も含めて先に計算）
	lat0 = float(station_df['lat'].mean())
	lon0 = float(station_df['lon'].mean())

	if mode == 'network_center':
		origin_time = origin_time_all
		lat = pd.Series(lat0, index=idx_all)
		lon = pd.Series(lon0, index=idx_all)

	elif mode == 'first_p':
		is_p = eqt_df['Phase'].astype(str).str.upper() == 'P'
		eqt_p = eqt_df[is_p].copy()

		if eqt_p.empty:
			# Pが1つも無いなら、全イベントネットワーク中心にフォールバック
			origin_time = origin_time_all
			lat = pd.Series(lat0, index=idx_all)
			lon = pd.Series(lon0, index=idx_all)
		else:
			eqt_p = eqt_p.sort_values(['event_id', 'pick_time'])
			first_p = eqt_p.groupby('event_id', sort=False).first()

			sta = station_df.set_index('station')[['lat', 'lon']]
			first_p = first_p.join(sta, on='station_code', how='left')

			if first_p['lat'].isna().any():
				raise ValueError(
					'first_p の station_code に station_df で見つからないものがあります。'
				)

			origin_time = origin_time_all
			lat = pd.Series(lat0, index=idx_all)
			lon = pd.Series(lon0, index=idx_all)

			lat.loc[first_p.index] = first_p['lat'].to_numpy()
			lon.loc[first_p.index] = first_p['lon'].to_numpy()

	else:
		raise ValueError("mode は 'network_center' か 'first_p' を指定してください。")

	depth = pd.Series(float(depth_km), index=idx_all)

	df_epic_eqt = pd.DataFrame(
		{
			'event_id': idx_all.astype(int),
			'origin_time': origin_time.to_numpy(),
			'latitude_deg': lat.to_numpy(),
			'longitude_deg': lon.to_numpy(),
			'depth_km': depth.to_numpy(),
		}
	)

	return df_epic_eqt
