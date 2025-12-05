from __future__ import annotations

from pathlib import Path

import pandas as pd

from common.geo import haversine_distance_km

COLS18 = [
	'ch_hex',  # [1] 16進チャネル番号
	'rec_flag',  # [2]
	'line_delay_ms',  # [3]
	'station',  # [4]
	'component',  # [5]
	'monitor_gain_idx',  # [6]
	'adc_bits',  # [7]
	'sensor_sensitivity',  # [8]
	'input_unit',  # [9]
	'nat_period_s',  # [10]
	'damping',  # [11]
	'preamp_gain_db',  # [12]
	'ad_lsb_delta_v',  # [13]
	'lat',  # [14]
	'lon',  # [15]
	'elevation_m',  # [16]
	'tt_corr_p',  # [17] 観測走時の観測点補正（P）※単位は元表に従う（多くは秒）
	'tt_corr_s',  # [18] 観測走時の観測点補正（S）
]


def read_hinet_channel_table(path: str | Path) -> pd.DataFrame:
	"""Hi-net チャネルファイル(hinet_channelstbl_####)を DataFrame に。
	- 先頭18列を読み込み・命名([17]=P補正, [18]=S補正)
	- 型変換と conv_coeff を付与
	conv_coeff = [13] / ([8] * 10**([12]/20))   # AD値 I → 物理量 v の比例係数: v = I * conv_coeff
	"""
	path = Path(path)
	df = pd.read_table(
		path, sep=r'\s+', engine='python', comment='#', header=None, dtype=str
	)
	if df.shape[1] < 18:
		raise ValueError(f'expected >=18 columns, got {df.shape[1]} in {path}')
	df = df.iloc[:, :18].copy()
	df.columns = COLS18

	df['ch_hex'] = df['ch_hex'].str.upper()
	df['ch_int'] = df['ch_hex'].apply(lambda s: int(s, 16))

	for c in ['rec_flag', 'line_delay_ms', 'monitor_gain_idx', 'adc_bits']:
		df[c] = df[c].astype(int)

	for c in [
		'sensor_sensitivity',
		'nat_period_s',
		'damping',
		'preamp_gain_db',
		'ad_lsb_delta_v',
		'lat',
		'lon',
		'elevation_m',
		'tt_corr_p',
		'tt_corr_s',
	]:
		df[c] = df[c].astype(float)

	df['conv_coeff'] = df['ad_lsb_delta_v'] / (
		df['sensor_sensitivity'] * (10.0 ** (df['preamp_gain_db'] / 20.0))
	)

	cols = [
		'ch_hex',
		'ch_int',
		'station',
		'component',
		'input_unit',
		'lat',
		'lon',
		'elevation_m',
		'rec_flag',
		'line_delay_ms',
		'monitor_gain_idx',
		'adc_bits',
		'sensor_sensitivity',
		'preamp_gain_db',
		'ad_lsb_delta_v',
		'nat_period_s',
		'damping',
		'tt_corr_p',
		'tt_corr_s',
		'conv_coeff',
	]
	return df[cols]


def stations_within_radius(
	lat: float,
	lon: float,
	radius_km: float,
	hinet_table_path: str
	| Path = '/workspace/proc/util/hinet_util/hinet_channelstbl_20251007',
) -> list[str]:
	"""(lat, lon) から半径 radius_km 以内の局名リストを返す（ハーサイン距離）。"""
	df = read_hinet_channel_table(hinet_table_path)
	lat_arr = df['lat'].to_numpy(dtype=float)
	lon_arr = df['lon'].to_numpy(dtype=float)

	dist_km = haversine_distance_km(
		lat0_deg=lat,
		lon0_deg=lon,
		lat_deg=lat_arr,
		lon_deg=lon_arr,
	)
	return df.loc[dist_km <= radius_km, 'station'].tolist()
