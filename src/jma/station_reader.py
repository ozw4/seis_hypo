from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import pandas as pd

from common.geo import haversine_distance_km

COLS18 = [
	'ch_hex',
	'rec_flag',
	'line_delay_ms',
	'station',
	'component',
	'monitor_gain_idx',
	'adc_bits',
	'sensor_sensitivity',
	'input_unit',
	'nat_period_s',
	'damping',
	'preamp_gain_db',
	'ad_lsb_delta_v',
	'lat',
	'lon',
	'elevation_m',
	'tt_corr_p',
	'tt_corr_s',
]

_INT_COLS = ['rec_flag', 'line_delay_ms', 'monitor_gain_idx', 'adc_bits']
_FLOAT_COLS = [
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
]

_DASH_TRANSLATION = str.maketrans(
	{
		'−': '-',  # U+2212
		'－': '-',  # U+FF0D
		'–': '-',  # U+2013
		'—': '-',  # U+2014
	}
)

# damping と preamp_gain_db が "0.70-10" みたいに結合されるケースを展開する
# 例: "0.70-10" -> ("0.70", "-10")
_DAMP_PREAMP_FUSED = re.compile(r'^(?P<damp>\d+(?:\.\d+)?)(?P<pre>[+-]\d+(?:\.\d+)?)$')


def _first_18_fields_by_spans(line: str, lineno_1based: int, path: Path) -> list[str]:
	line = line.translate(_DASH_TRANSLATION)

	spans = [(m.start(), m.end()) for m in re.finditer(r'\S+', line)]
	if len(spans) < 18:
		raise ValueError(
			f'expected >=18 fields, got {len(spans)} at line={lineno_1based} in {path}'
		)

	fields = [line[a:b] for a, b in spans]

	# damping位置（0-based index=10）で fused を展開して列ズレを修正
	if len(fields) >= 11:
		m = _DAMP_PREAMP_FUSED.match(fields[10])
		if m:
			fields = fields[:10] + [m.group('damp'), m.group('pre')] + fields[11:]

	if len(fields) < 18:
		raise ValueError(
			f'expected >=18 fields after normalization, got {len(fields)} at line={lineno_1based} in {path}'
		)

	return [s.strip() for s in fields[:18]]


def read_hinet_channel_table(path: str | Path) -> pd.DataFrame:
	path = Path(path)
	if not path.is_file():
		raise FileNotFoundError(f'.ch not found: {path}')

	rows: list[list[str]] = []
	with open(path, encoding='utf-8', errors='ignore') as f:
		for lineno_1based, raw in enumerate(f, start=1):
			line = raw.rstrip('\n')
			if not line.strip():
				continue
			if line.lstrip().startswith('#'):
				continue
			rows.append(_first_18_fields_by_spans(line, lineno_1based, path))

	if not rows:
		raise ValueError(f'no data rows found in {path}')

	df = pd.DataFrame(rows, columns=COLS18, dtype=str)

	df['ch_hex'] = df['ch_hex'].str.upper()
	df['ch_int'] = df['ch_hex'].map(lambda s: int(s, 16))

	for c in _INT_COLS:
		df[c] = df[c].astype(int)

	for c in _FLOAT_COLS:
		df[c] = df[c].str.replace('D', 'E', regex=False).astype(float)

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
	| Path = '/workspace/data/station/jma/hinet_channelstbl_20251007',
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


def _decode_lon_lat_from_13digits(code13: str) -> tuple[float, float]:
	if len(code13) != 13 or not code13.isdigit():
		msg = f'invalid lon/lat code: {code13!r}'
		raise ValueError(msg)

	v = int(code13)

	lon_raw = v // 10**6  # 上位7桁
	lat_raw = v % 10**6  # 下位6桁

	lon_deg = lon_raw // 10000
	lon_min = (lon_raw % 10000) / 100.0
	lat_deg = lat_raw // 10000
	lat_min = (lat_raw % 10000) / 100.0

	lon = lon_deg + lon_min / 60.0
	lat = lat_deg + lat_min / 60.0

	return lon, lat


def parse_network_codes_from_client_info(info_text: object) -> list[str]:
	"""client.info() の出力から network_code を抽出する（6桁/末尾英字対応）。

	例:
	0101   : ...
	0103A  : ...
	010501 : ...
	0402AN : ...  # 末尾2文字も来るので許容
	"""
	if info_text is None:
		raise ValueError('client.info() returned None')

	if isinstance(info_text, bytes):
		text = info_text.decode('utf-8', errors='ignore')
	elif isinstance(info_text, dict):
		text = '\n'.join(f'{k}: {v}' for k, v in info_text.items())
	elif isinstance(info_text, (list, tuple, set)):
		text = '\n'.join(str(x) for x in info_text)
	else:
		text = str(info_text)

	# 4桁 or 6桁 + 末尾英数0〜2文字（A, AN など）
	code_pat = r'([0-9]{4}(?:[0-9]{2})?[0-9A-Z]{0,2})'

	out: list[str] = []
	seen: set[str] = set()

	for raw in text.splitlines():
		ln = raw.strip()
		if not ln:
			continue

		# 行頭 "CODE : " / "CODE ： " / "CODE  " を拾う
		m = re.match(rf'^{code_pat}\s*(?:[:：]|\s+)', ln)
		if m:
			c = m.group(1)
			if c not in seen:
				seen.add(c)
				out.append(c)
			continue

		# 念のため行中も拾う（境界つき）
		m2 = re.search(rf'\b{code_pat}\b', ln)
		if m2:
			c = m2.group(1)
			if c not in seen:
				seen.add(c)
				out.append(c)

	if not out:
		head = text[:200].replace('\n', '\\n')
		raise ValueError(
			f'no network codes parsed from client.info() text; head="{head}"'
		)

	return out


def build_station_names_by_network(
	client,
	network_codes: list[str],
) -> dict[str, set[str]]:
	"""network_code -> station name set を構築する（get_station_list 前提）。"""
	if not network_codes:
		raise ValueError('network_codes is empty')

	out: dict[str, set[str]] = {}
	for code in network_codes:
		stations = client.get_station_list(str(code))
		names: set[str] = set()
		for s in stations:
			name = getattr(s, 'name', None)
			if name is None:
				name = str(s)
			names.add(str(name))
		if not names:
			raise ValueError(f'empty station list for network_code={code}')
		out[str(code)] = names
	return out


def assign_network_code_by_membership(
	station_codes: list[str],
	*,
	station_names_by_network: dict[str, set[str]],
	priority: list[str] | None = None,
	default_network_code: str = '',
) -> dict[str, list[str]]:
	"""station_code を station_names_by_network の所属で network に割り当てる。"""
	if not station_codes:
		raise ValueError('station_codes is empty')
	if not station_names_by_network:
		raise ValueError('station_names_by_network is empty')

	priority = [str(x) for x in (priority or [])]
	known_nets = set(station_names_by_network.keys())
	for p in priority:
		if p not in known_nets:
			raise ValueError(
				f'priority network_code not in info/get_station_list results: {p}'
			)

	def pick_network(candidates: list[str]) -> str:
		if len(candidates) == 1:
			return candidates[0]
		for p in priority:
			if p in candidates:
				return p
		raise ValueError(
			f'station appears in multiple networks but no priority match: {candidates}'
		)

	out: dict[str, set[str]] = {}
	missing: list[str] = []
	for sta in station_codes:
		sta = str(sta)
		cands = [net for net, names in station_names_by_network.items() if sta in names]
		if not cands:
			if default_network_code:
				out.setdefault(str(default_network_code), set()).add(sta)
			else:
				missing.append(sta)
			continue
		net = pick_network(cands)
		out.setdefault(net, set()).add(sta)

	if missing:
		ex = ', '.join(missing[:20])
		raise ValueError(
			f'stations not found in any network station list: {ex} (total={len(missing)})'
		)

	return {k: sorted(v) for k, v in sorted(out.items(), key=lambda kv: kv[0])}
