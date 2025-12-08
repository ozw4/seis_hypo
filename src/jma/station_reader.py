from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from common.geo import haversine_distance_km

# Hi-net チャネルテーブル (hinet_channelstbl_YYYYMMDD) の
# 先頭 18 列に対応するカラム名
COLS18 = [
	'ch_hex',  # [1] 16進チャネル番号
	'rec_flag',  # [2]
	'line_delay_ms',  # [3]
	'station',  # [4] 局名
	'component',  # [5] 成分 (U/N/E など)
	'monitor_gain_idx',  # [6]
	'adc_bits',  # [7]
	'sensor_sensitivity',  # [8]
	'input_unit',  # [9]
	'nat_period_s',  # [10]
	'damping',  # [11]
	'preamp_gain_db',  # [12]
	'ad_lsb_delta_v',  # [13]
	'lat',  # [14] 緯度 (deg)
	'lon',  # [15] 経度 (deg)
	'elevation_m',  # [16]
	'tt_corr_p',  # [17] P 走時計算の局補正
	'tt_corr_s',  # [18] S 走時計算の局補正
]


def read_hinet_channel_table(path: str | Path) -> pd.DataFrame:
	"""Hi-net チャネルテーブルを読み込んで標準化した DataFrame を返す.

	対象ファイル:
		hinet_channelstbl_YYYYMMDD などの 18 列以上あるテキストファイル。

	処理内容:
	- 先頭 18 列を読み込んで COLS18 というカラム名を付与
	- 型変換:
		* ch_hex -> 大文字 16進文字列
		* ch_int -> ch_hex を int にした列を追加
		* rec_flag, line_delay_ms, monitor_gain_idx, adc_bits -> int
		* それ以外の数値列は float
	- conv_coeff 列を付与:
		conv_coeff = ad_lsb_delta_v / (sensor_sensitivity * 10**(preamp_gain_db/20))
		これで AD 値 I から物理量 v を v = I * conv_coeff で計算できる。

	戻り値 DataFrame の主なカラム:
		ch_hex, ch_int, station, component, input_unit,
		lat, lon, elevation_m, rec_flag, line_delay_ms, monitor_gain_idx,
		adc_bits, sensor_sensitivity, preamp_gain_db, ad_lsb_delta_v,
		nat_period_s, damping, tt_corr_p, tt_corr_s, conv_coeff
	"""
	path = Path(path)

	df_raw = pd.read_table(
		path,
		sep=r'\s+',
		engine='python',
		comment='#',
		header=None,
		dtype=str,
	)

	if df_raw.shape[1] < 18:
		raise ValueError(f'expected >=18 columns, got {df_raw.shape[1]} in {path}')

	df = df_raw.iloc[:, :18].copy()
	df.columns = COLS18

	# チャネル番号: 16進文字列と int を両方持っておく
	df['ch_hex'] = df['ch_hex'].str.upper()
	df['ch_int'] = df['ch_hex'].apply(lambda s: int(s, 16))

	# int 型カラム
	for col in ['rec_flag', 'line_delay_ms', 'monitor_gain_idx', 'adc_bits']:
		df[col] = df[col].astype(int)

	# float 型カラム
	for col in [
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
		df[col] = df[col].astype(float)

	# AD -> 物理量変換係数
	df['conv_coeff'] = df['ad_lsb_delta_v'] / (
		df['sensor_sensitivity'] * (10.0 ** (df['preamp_gain_db'] / 20.0))
	)

	cols_out = [
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
	return df[cols_out]


def stations_within_radius(
	lat: float,
	lon: float,
	radius_km: float,
	channel_table_path: str
	| Path = '/workspace/proc/util/hinet_util/hinet_channelstbl_20251007',
	*,
	output: Literal['list', 'rows', 'both'] = 'list',
) -> list[str] | pd.DataFrame | tuple[list[str], pd.DataFrame]:
	"""(lat, lon) から半径 radius_km 以内の Hi-net 局を抽出する統合関数。

	output:
		"list" : station 名のリストを返す（従来の stations_within_radius 相当）
		"rows" : station 単位の代表座標 DataFrame を返す（従来の station_rows_within_radius 相当）
		"both" : (list, DataFrame) のタプルを返す

	DataFrame の列:
		station, lat, lon, elevation_m
	"""
	df = read_hinet_channel_table(channel_table_path)

	lat_arr = df['lat'].to_numpy(dtype=float)
	lon_arr = df['lon'].to_numpy(dtype=float)

	dist_km = haversine_distance_km(
		lat0_deg=lat,
		lon0_deg=lon,
		lat_deg=lat_arr,
		lon_deg=lon_arr,
	)

	df_hit = df.loc[
		dist_km <= radius_km, ['station', 'lat', 'lon', 'elevation_m']
	].copy()

	if df_hit.empty:
		raise ValueError('no stations found within the specified radius')

	# station 単位に代表行へ集約（component 等の重複を除去）
	df_sta = (
		df_hit.groupby('station', as_index=False)
		.agg({'lat': 'first', 'lon': 'first', 'elevation_m': 'first'})
		.sort_values('station')
		.reset_index(drop=True)
	)

	stations = df_sta['station'].tolist()

	if output == 'list':
		return stations
	if output == 'rows':
		return df_sta
	if output == 'both':
		return stations, df_sta

	raise ValueError(f'unsupported output: {output}')
