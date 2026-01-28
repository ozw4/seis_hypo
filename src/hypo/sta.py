# src/hypo/sta.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from common.core import validate_columns
from hypo.station_meta import parse_station_code


def decimal_deg_to_degmin_fields(
	dd: Any,
	*,
	pos_hemi: str,
	neg_hemi: str,
	deg_width: int,
) -> tuple[int | None, float | None, str | None]:
	"""10進度 -> (deg, minutes, hemisphere)"""
	if pd.isna(dd):
		return None, None, None

	v = float(dd)
	hemi = neg_hemi if v < 0.0 else pos_hemi
	a = abs(v)
	deg = int(a)
	minutes = (a - deg) * 60.0
	return deg, minutes, hemi


def decimal_deg_to_lat_fields(lat: Any) -> tuple[int | None, float | None, str | None]:
	"""10進緯度 -> (deg, minutes, 'N'/'S')"""
	return decimal_deg_to_degmin_fields(
		lat,
		pos_hemi='N',
		neg_hemi='S',
		deg_width=2,
	)


def decimal_deg_to_lon_fields(lon: Any) -> tuple[int | None, float | None, str | None]:
	"""10進経度 -> (deg, minutes, 'E'/'W')"""
	return decimal_deg_to_degmin_fields(
		lon,
		pos_hemi='E',
		neg_hemi='W',
		deg_width=3,
	)


def _format_elevation_i4_and_neg_flag(elevation_m: int) -> tuple[int, str]:
	"""Station format #2 の I4 elevation と col 86 の負フラグを決める。

	仕様:
	- elevation 欄は I4 なので、値が -1000 以下だと固定幅が崩れる。
	- col 86 に '-' を立てると、elevation の符号に関係なく負の深さとして解釈される。
	  (HypoInverse-2000 User's Guide: station format #2 col 86)
	"""
	if abs(elevation_m) > 9999:
		raise ValueError(f'Elevation_m is out of I4 range: {elevation_m}')

	if elevation_m < -999:
		return abs(elevation_m), '-'
	return elevation_m, ' '


def format_station_line(
	site: str,
	net: str,
	lat_deg: int,
	lat_min: float,
	lat_hemi: str,
	lon_deg: int,
	lon_min: float,
	lon_hemi: str,
	elevation_m: int = 0,
	comp1: str = 'Z',
	chan: str = 'HHZ',
	weight_code: str = ' ',
	default_period: float = 1.0,
	pdelay1: float = 0.0,
	pdelay2: float = 0.0,
	amag_corr: float = 0.0,
	amag_w: str = ' ',
	dmag_corr: float = 0.0,
	dmag_w: str = ' ',
	inst_type: int = 0,
	cal_factor: float = 1.0,
	loc_code: str = '  ',
	alt_chan: str = '   ',
) -> str:
	"""Hypoinverse station format #2 の1行を組み立てる。"""
	# 固定幅が崩れないよう、明示的に切り詰め/パディングする
	site = str(site)[:5]
	net = str(net)[:2]
	comp1 = str(comp1)[:1] if comp1 is not None else ' '
	chan = str(chan)[:3]
	weight_code = str(weight_code)[:1] if weight_code is not None else ' '
	loc_code = str(loc_code).ljust(2)[:2]
	alt_chan = str(alt_chan).ljust(3)[:3]

	elev_i4, neg_flag = _format_elevation_i4_and_neg_flag(int(elevation_m))

	line = ''

	# 1-5: site, 6: space
	line += f'{site:<5} '
	# 7-8: net, 9: space
	line += f'{net:<2} '
	# 10: comp1, 11-13: chan, 14: space
	line += f'{comp1:1s}{chan:<3} '
	# 15: station weight code
	line += f'{weight_code:1s}'
	# 16-17: lat deg, 18: space
	line += f'{lat_deg:2d} '
	# 19-25: lat minutes F7.4, 26: N/S
	line += f'{lat_min:7.4f}{lat_hemi:1s}'
	# 27-29: lon deg, 30: space
	line += f'{lon_deg:3d} '
	# 31-37: lon minutes F7.4, 38: E/W
	line += f'{lon_min:7.4f}{lon_hemi:1s}'
	# 39-42: elevation (I4)
	line += f'{elev_i4:4d}'
	# 43-45: default period F3.1, 46-47: spaces
	line += f'{default_period:3.1f}  '
	# 48: alternate crust model flag
	line += ' '
	# 49: station remark
	line += ' '
	# 50-54: P delay set1 F5.2, 55: space
	line += f'{pdelay1:5.2f} '
	# 56-60: P delay set2 F5.2, 61: space
	line += f'{pdelay2:5.2f} '
	# 62-66: amplitude magnitude correction F5.2
	line += f'{amag_corr:5.2f}'
	# 67: amplitude magnitude weight code
	line += f'{amag_w:1s}'
	# 68-72: duration magnitude correction F5.2
	line += f'{dmag_corr:5.2f}'
	# 73: duration magnitude weight code
	line += f'{dmag_w:1s}'
	# 74: instrument type I1
	line += f'{inst_type:1d}'
	# 75-80: calibration factor F6.2
	line += f'{cal_factor:6.2f}'
	# 81-82: location code A2
	line += f'{loc_code:2s}'
	# 83-85: alternate component/channel code A3
	line += f'{alt_chan:3s}'
	# 86: '-' to force negative depth
	line += f'{neg_flag:1s}'

	if len(line) != 86:
		raise ValueError(
			f'Invalid station line length (expected 86): {len(line)}: {line!r}'
		)

	return line


def write_hypoinverse_sta(
	station_csv: str | Path,
	output_sta: str | Path,
	default_channel: str = 'HHZ',
	default_comp1: str = 'Z',
	default_period: float = 1.0,
	*,
	force_zero_pdelays: bool = False,
) -> None:
	"""station.csv から Hypoinverse #2 station file (.sta) を生成する。

	前提カラム:
	- station_code      (例: 'N.KKGH', 'ABASH2', 'S4N04' など)
	- Latitude_deg      (10進度)
	- Longitude_deg     (10進度)

	任意カラム（あれば補正量として使用する。無い/NaN は 0 or 空文字）:
	- pdelay1, pdelay2        : P delay set1/set2 [s]
	- amag_corr, amag_w       : 振幅マグニチュード補正と重み
	- dmag_corr, dmag_w       : 継続時間マグニチュード補正と重み
	- inst_type               : 計器種別コード (int)
	- cal_factor              : キャリブレーション係数
	- weight_code             : ステーション重みコード (col 15)
	- loc_code                : ロケーションコード (2文字)
	- channel                 : チャンネル名（例 'HHZ'）※あれば default_channel を上書き
	- comp1                   : comp1（例 'Z'）※あれば default_comp1 を上書き
	- default_period          : ステーションごとの default period（あれば引数を上書き）

	force_zero_pdelays=True の場合、CSV 内の pdelay1/pdelay2 を無視して 0.0 を出力する。
	(CRE で station elevation を travel-time 計算に含める場合など、station delay の二重補正を避けたい用途向け)
	"""
	df = pd.read_csv(station_csv)

	validate_columns(
		df,
		['station_code', 'Latitude_deg', 'Longitude_deg'],
		'station CSV',
	)

	out_path = Path(output_sta)
	lines: list[str] = []

	for _, row in df.iterrows():
		site, net = parse_station_code(row.get('station_code', ''))
		if site is None:
			continue

		lat_deg, lat_min, lat_hemi = decimal_deg_to_lat_fields(row.get('Latitude_deg'))
		lon_deg, lon_min, lon_hemi = decimal_deg_to_lon_fields(row.get('Longitude_deg'))
		if lat_deg is None or lon_deg is None:
			continue

		chan = (
			str(row['channel']).strip()
			if 'channel' in row and pd.notna(row['channel'])
			else default_channel
		)
		comp1 = (
			str(row['comp1']).strip()
			if 'comp1' in row and pd.notna(row['comp1'])
			else default_comp1
		)

		weight_code = (
			str(row['weight_code']).strip()[:1]
			if 'weight_code' in row and pd.notna(row['weight_code'])
			else ' '
		)

		period = (
			float(row['default_period'])
			if 'default_period' in row and pd.notna(row['default_period'])
			else float(default_period)
		)

		if force_zero_pdelays:
			pdelay1 = 0.0
			pdelay2 = 0.0
		else:
			pdelay1 = (
				float(row['pdelay1'])
				if 'pdelay1' in row and pd.notna(row['pdelay1'])
				else 0.0
			)
			pdelay2 = (
				float(row['pdelay2'])
				if 'pdelay2' in row and pd.notna(row['pdelay2'])
				else 0.0
			)

		amag_corr = (
			float(row['amag_corr'])
			if 'amag_corr' in row and pd.notna(row['amag_corr'])
			else 0.0
		)
		amag_w = (
			str(row['amag_w']).strip()[:1]
			if 'amag_w' in row and pd.notna(row['amag_w'])
			else ' '
		)

		dmag_corr = (
			float(row['dmag_corr'])
			if 'dmag_corr' in row and pd.notna(row['dmag_corr'])
			else 0.0
		)
		dmag_w = (
			str(row['dmag_w']).strip()[:1]
			if 'dmag_w' in row and pd.notna(row['dmag_w'])
			else ' '
		)

		inst_type = (
			int(row['inst_type'])
			if 'inst_type' in row and pd.notna(row['inst_type'])
			else 0
		)
		cal_factor = (
			float(row['cal_factor'])
			if 'cal_factor' in row and pd.notna(row['cal_factor'])
			else 1.0
		)

		loc_code = (
			str(row['loc_code']).ljust(2)[:2]
			if 'loc_code' in row and pd.notna(row['loc_code'])
			else '  '
		)

		elevation_m = (
			int(row['Elevation_m'])
			if 'Elevation_m' in row and pd.notna(row['Elevation_m'])
			else 0
		)

		line = format_station_line(
			site=site,
			net=net or '',
			lat_deg=lat_deg,
			lat_min=lat_min,
			lat_hemi=lat_hemi,
			lon_deg=lon_deg,
			lon_min=lon_min,
			lon_hemi=lon_hemi,
			elevation_m=elevation_m,
			comp1=comp1,
			chan=chan,
			weight_code=weight_code,
			default_period=period,
			pdelay1=pdelay1,
			pdelay2=pdelay2,
			amag_corr=amag_corr,
			amag_w=amag_w,
			dmag_corr=dmag_corr,
			dmag_w=dmag_w,
			inst_type=inst_type,
			cal_factor=cal_factor,
			loc_code=loc_code,
		)
		lines.append(line)

	with out_path.open('w', encoding='ascii', newline='\n') as f:
		for line in lines:
			f.write(line)
			f.write('\n')
