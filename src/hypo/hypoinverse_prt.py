import re
from pathlib import Path

import pandas as pd

from hypo.uncertainty_ellipsoid import ELLIPSE_COLS


# ========= hypoinverse .prt パース =========
_SUMMARY_RE = re.compile(r'^\s*\d{4}-\d{2}-\d{2}')
_NSTA_HEADER_RE = re.compile(r'^\s*NSTA\s+NPHS\b', re.IGNORECASE)
_ELL_TRIPLET_RE = re.compile(
	r'<\s*([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+)\s+([+-]?\d+)\s*>'
)


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


def parse_error_ellipse_line(line: str) -> dict:
	"""Hypoinverse .prt の ERROR ELLIPSE 行をパースする。

	例:
	ERROR ELLIPSE: <SERR AZ DIP>-<   0.12  91 21>-<   0.09   0  0>-<   0.08 271 68>
	"""
	if 'ERROR ELLIPSE' not in line.upper():
		raise ValueError(f'not an ERROR ELLIPSE line: {line!r}')

	triples = _ELL_TRIPLET_RE.findall(line)
	if len(triples) != 3:
		raise ValueError(f'ERROR ELLIPSE must contain 3 numeric triplets: {line!r}')

	s1, az1, dip1 = triples[0]
	s2, az2, dip2 = triples[1]
	s3, az3, dip3 = triples[2]

	return {
		'ell_s1_km': float(s1),
		'ell_az1_deg': int(az1),
		'ell_dip1_deg': int(dip1),
		'ell_s2_km': float(s2),
		'ell_az2_deg': int(az2),
		'ell_dip2_deg': int(dip2),
		'ell_s3_km': float(s3),
		'ell_az3_deg': int(az3),
		'ell_dip3_deg': int(dip3),
	}


def parse_eigenvalues_block(lines: list[str], i: int) -> tuple[dict, int]:
	"""EIGENVALUES 行 + 次行の (a b c d) をパースして返す。

	EIGENVALUES が存在しない場合に呼ばないこと。
	"""
	if i < 0 or i >= len(lines):
		raise IndexError(f'line index out of range: {i}')
	if lines[i].strip().upper() != 'EIGENVALUES':
		raise ValueError(f'not an EIGENVALUES line: {lines[i]!r}')
	if i + 1 >= len(lines):
		raise ValueError('EIGENVALUES block is truncated: missing values line')

	s = lines[i + 1]
	m = re.search(r'\(([^)]*)\)', s)
	if m is None:
		raise ValueError(f'EIGENVALUES values line must contain parentheses: {s!r}')

	parts = m.group(1).split()
	if len(parts) != 4:
		raise ValueError(f'EIGENVALUES must contain 4 values: {s!r}')

	vals = [float(x) for x in parts]
	return (
		{
			'eig_adj1': vals[0],
			'eig_adj2': vals[1],
			'eig_adj3': vals[2],
			'eig_adj4': vals[3],
		},
		i + 2,
	)


def load_hypoinverse_summary_from_prt(prt_path: str | Path) -> pd.DataFrame:
	"""hypoinverse_run.prt からイベント summary + 幾何情報を DataFrame にする。

	カラム:
	  origin_time_hyp, lat_deg_hyp, lon_deg_hyp, depth_km_hyp,
	  RMS, ERH, ERZ,
	  NSTA, NPHS, DMIN, MODEL, GAP, ITR, NFM, NWR, NWS, NVR,
	  ell_s1_km, ell_az1_deg, ell_dip1_deg,
	  ell_s2_km, ell_az2_deg, ell_dip2_deg,
	  ell_s3_km, ell_az3_deg, ell_dip3_deg,
	  eig_adj1, eig_adj2, eig_adj3, eig_adj4
	"""
	prt_path = Path(prt_path)
	text = prt_path.read_text(encoding='ascii', errors='strict')
	lines = text.splitlines()

	pending_error_ellipse: dict | None = None
	pending_eigenvalues: dict | None = None
	records: list[dict] = []

	i = 0
	while i < len(lines):
		line = lines[i]
		s = line.strip()
		u = s.upper()

		if u == 'EIGENVALUES':
			if pending_eigenvalues is not None:
				raise ValueError('multiple EIGENVALUES blocks before an event summary')
			vals, next_i = parse_eigenvalues_block(lines, i)
			pending_eigenvalues = vals
			i = next_i
			continue

		if 'ERROR ELLIPSE' in u:
			if pending_error_ellipse is not None:
				raise ValueError('multiple ERROR ELLIPSE lines before an event summary')
			pending_error_ellipse = parse_error_ellipse_line(line)
			i += 1
			continue

		if _SUMMARY_RE.match(line):
			if pending_error_ellipse is None:
				raise ValueError(f'ERROR ELLIPSE is missing for event summary: {line!r}')

			rec = parse_summary_line(line)
			rec.update(pending_error_ellipse)
			if pending_eigenvalues is not None:
				rec.update(pending_eigenvalues)
			else:
				rec.update(
					{
						'eig_adj1': None,
						'eig_adj2': None,
						'eig_adj3': None,
						'eig_adj4': None,
					}
				)

			records.append(rec)
			pending_error_ellipse = None
			pending_eigenvalues = None
			i += 1
			continue

		if _NSTA_HEADER_RE.match(line):
			if not records:
				raise ValueError('NSTA/NPHS block found before any event summary')
			if i + 1 >= len(lines):
				raise ValueError('NSTA/NPHS header line is truncated: missing values line')
			records[-1].update(parse_nsta_line(lines[i + 1]))
			i += 2
			continue

		i += 1

	if pending_error_ellipse is not None:
		raise ValueError('found ERROR ELLIPSE without a following event summary')

	if not records:
		raise RuntimeError('no events parsed from .prt')

	required_nsta_keys = [
		'NSTA',
		'NPHS',
		'DMIN',
		'MODEL',
		'GAP',
		'ITR',
		'NFM',
		'NWR',
		'NWS',
		'NVR',
	]
	required_ell_keys = list(ELLIPSE_COLS)

	for j, rec in enumerate(records, start=1):
		missing = [k for k in required_ell_keys if k not in rec]
		if missing:
			raise RuntimeError(f'missing error-ellipse fields for event {j}: {missing}')
		missing = [k for k in required_nsta_keys if k not in rec]
		if missing:
			raise RuntimeError(f'missing NSTA/NPHS fields for event {j}: {missing}')

	df = pd.DataFrame(records)
	# hypoinverse 側の通し番号 seq（.arc のイベント順と対応させる）
	df['seq'] = range(1, len(df) + 1)
	return df
