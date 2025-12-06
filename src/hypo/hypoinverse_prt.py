import re
from pathlib import Path

import pandas as pd


# ========= hypoinverse .prt パース =========
def parse_summary_line(line: str) -> dict:
	"""Hypoinverse .prt の summary 行を固定カラムでパースする。

	例:
	2002-06-03  0002 29.67  34 41.93  132E 3.01  14.21  0.10  0.38  1.01                      14.21
	"""
	s = line  # 行頭スペースはそのまま

	if len(s) < 70:
		raise ValueError(f'summary line too short: {len(s)} | {s!r}')

	# --- 日時 ---
	#  1–10 : 'YYYY-MM-DD'
	# 13–16 : hhmm
	# 18–22 : 秒 (F5.2, 例 '29.67')
	date_str = s[1:11]
	hhmm_str = s[13:17].strip()
	sec_str = s[18:23].strip()

	if not date_str or not hhmm_str or not sec_str:
		raise ValueError(f'invalid datetime fields in summary line: {s!r}')

	year, month, day = map(int, date_str.split('-'))

	if len(hhmm_str) != 4 or not hhmm_str.isdigit():
		raise ValueError(f'invalid hhmm field: {hhmm_str!r}')

	hour = int(hhmm_str[:2])
	minute = int(hhmm_str[2:4])

	sec = float(sec_str)
	sec_int = int(sec)
	micro = int(round((sec - sec_int) * 1_000_000))

	origin_time = pd.Timestamp(
		year=year,
		month=month,
		day=day,
		hour=hour,
		minute=minute,
		second=sec_int,
		microsecond=micro,
	)

	# --- 緯度 ---
	# 25–26 : 緯度度 (I2)
	# 28–32/33 : 緯度分 (F5.2 相当, 例 '41.93')
	lat_deg_str = s[25:27].strip()
	lat_min_str = s[28:33].strip()

	if not lat_deg_str:
		raise ValueError(f'missing latitude degree field: {s!r}')

	lat_deg = float(lat_deg_str)
	lat_min = float(lat_min_str) if lat_min_str else 0.0
	lat = lat_deg + lat_min / 60.0

	# --- 経度 ---
	# 35–37 : 経度度 (I3)
	# 38    : E/W
	# 39–43/44 : 経度分 (F5.2 相当, 例 '52.34' や ' 3.01')
	lon_deg_str = s[35:38].strip()
	hemi = s[38]
	lon_min_str = s[39:44].strip()

	if not lon_deg_str or hemi not in ('E', 'W'):
		raise ValueError(f'invalid longitude fields: {s!r}')

	lon_deg = float(lon_deg_str)
	lon_min = float(lon_min_str) if lon_min_str else 0.0
	lon = lon_deg + lon_min / 60.0
	if hemi == 'W':
		lon = -lon

	# --- depth / RMS / ERH / ERZ ---
	# 46–50 : depth_km (F5.2, 例 '12.00')
	# 53–56 : RMS (F4.2, 例 '0.13')
	# 59–62 : ERH (F4.2, 例 '0.25')
	# 65–68 : ERZ (F4.2, 例 '0.88')
	depth_str = s[44:51].strip()
	rms_str = s[53:57].strip()
	erh_str = s[59:63].strip()
	erz_str = s[65:69].strip()

	if not depth_str:
		raise ValueError(f'missing depth field: {s!r}')

	depth_km = float(depth_str)
	rms = float(rms_str) if rms_str else None
	erh = float(erh_str) if erh_str else None
	erz = float(erz_str) if erz_str else None

	return {
		'origin_time_hyp': origin_time,
		'lat_deg_hyp': lat,
		'lon_deg_hyp': lon,
		'depth_km_hyp': depth_km,
		'RMS': rms,
		'ERH': erh,
		'ERZ': erz,
	}


def parse_nsta_line(line: str) -> dict:
	"""NSTA NPHS 行の「値の行」を split ベースでパースする。

	例:
	  NSTA NPHS  DMIN MODEL GAP ITR NFM NWR NWS NVR ...
		15   15  11.8  JMAS  97   8   0  15   8  15
	"""
	parts = line.split()
	if len(parts) < 10:
		raise ValueError(f'NSTA line too short: {line!r}')

	nsta = int(parts[0])
	nphs = int(parts[1])
	dmin = float(parts[2])
	model = parts[3]
	gap = int(parts[4])
	itr = int(parts[5])
	nfm = int(parts[6])
	nwr = int(parts[7])
	nws = int(parts[8])
	nvr = int(parts[9])

	return {
		'NSTA': nsta,
		'NPHS': nphs,
		'DMIN': dmin,
		'MODEL': model,
		'GAP': gap,
		'ITR': itr,
		'NFM': nfm,
		'NWR': nwr,
		'NWS': nws,
		'NVR': nvr,
	}


def load_hypoinverse_summary_from_prt(prt_path: str | Path) -> pd.DataFrame:
	"""hypoinverse_run.prt からイベント summary + 幾何情報を DataFrame にする。

	カラム:
	  origin_time_hyp, lat_deg_hyp, lon_deg_hyp, depth_km_hyp,
	  RMS, ERH, ERZ,
	  NSTA, NPHS, DMIN, MODEL, GAP, ITR, NFM, NWR, NWS, NVR
	"""
	prt_path = Path(prt_path)
	text = prt_path.read_text(encoding='ascii', errors='strict')
	lines = text.splitlines()

	summary_lines: list[str] = []
	nsta_value_lines: list[str] = []

	for i, line in enumerate(lines):
		# summary 行: 先頭近くに YYYY-MM-DD がある行
		if re.match(r'^\s*\d{4}-\d{2}-\d{2}', line):
			summary_lines.append(line)
		# NSTA NPHS 行の直後の数値行
		if 'NSTA NPHS' in line and i + 1 < len(lines):
			nsta_value_lines.append(lines[i + 1].strip())

	n = min(len(summary_lines), len(nsta_value_lines))
	records: list[dict] = []

	for idx in range(n):
		sline = summary_lines[idx]
		nline = nsta_value_lines[idx]
		rec = parse_summary_line(sline)
		rec.update(parse_nsta_line(nline))
		records.append(rec)

	if not records:
		raise RuntimeError('no events parsed from .prt')

	df = pd.DataFrame(records)
	# hypoinverse 側の通し番号 seq（.arc のイベント順と対応させる）
	df['seq'] = range(1, len(df) + 1)
	return df
