# %%
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import pandas as pd

# ==========================
# 設定（ここだけ直に編集）
# ==========================
YEARLY_ROOT = Path('snapshots/yearly')
MONTHLY_ROOT = Path('snapshots/monthly')

OUT_PRESENCE = MONTHLY_ROOT / 'monthly_presence.csv'
OUT_LIFESPANS = MONTHLY_ROOT / 'monthly_lifespans.csv'

COL_NETWORK = 'network_code'
COL_STATION = 'station'

# stations_*.csv に含まれるメタ情報（monthly_presence.csv にも持たせる）
COL_STATION_NAME = 'station_name'
COL_LAT = 'lat'
COL_LON = 'lon'
COL_ELEV_M = 'elevation_m'

# monthly の YYYY-MM ディレクトリ
_RE_MONTH_DIR = re.compile(r'^\d{4}-\d{2}$')
# stations_YYYYMMDDHHMM.csv や、親ディレクトリ名 YYYYMMDDHHMM から拾う
_RE_TS12 = re.compile(r'(\d{12})')


def _parse_ts12(ts12: str) -> dt.datetime:
	y = int(ts12[0:4])
	m = int(ts12[4:6])
	d = int(ts12[6:8])
	hh = int(ts12[8:10])
	mm = int(ts12[10:12])
	return dt.datetime(y, m, d, hh, mm)


def _floor_month(d: dt.datetime) -> dt.date:
	return dt.date(d.year, d.month, 1)


def _add_months(d: dt.date, n: int) -> dt.date:
	y = d.year
	m = d.month + n
	y += (m - 1) // 12
	m = (m - 1) % 12 + 1
	return dt.date(y, m, 1)


def _month_range(start: dt.date, end_inclusive: dt.date) -> list[dt.date]:
	if end_inclusive < start:
		raise ValueError(f'invalid range: start={start} end={end_inclusive}')
	out: list[dt.date] = []
	cur = dt.date(start.year, start.month, 1)
	last = dt.date(end_inclusive.year, end_inclusive.month, 1)
	while cur <= last:
		out.append(cur)
		cur = _add_months(cur, 1)
	return out


def _month_label(m: dt.date) -> str:
	return f'{m.year:04d}-{m.month:02d}'


def _canon_network_code(x: object) -> str:
	s = '' if x is None else str(x).strip()
	if s == '' or s.lower() == 'nan':
		raise ValueError('empty network_code')

	# "801.0" みたいなのが紛れた場合だけ救う（本来は文字列で来るのが正）
	m = re.fullmatch(r'(\d+)\.0+', s)
	if m:
		s = m.group(1)

	# 英字入り（0103Aなど）は触らない
	if not re.fullmatch(r'\d+', s):
		return s

	# 数字だけはゼロ埋めで正規化
	if len(s) <= 3:
		return s.zfill(4)  # 801 -> 0801
	if len(s) == 4:
		return s
	if len(s) == 5:
		return s.zfill(6)  # 10501 -> 010501
	if len(s) == 6:
		return s
	return s


def _load_station_sets(csv_path: Path) -> dict[str, set[str]]:
	if not csv_path.is_file():
		raise FileNotFoundError(f'missing stations csv: {csv_path}')

	df = pd.read_csv(csv_path, dtype=str)
	if COL_NETWORK not in df.columns or COL_STATION not in df.columns:
		raise ValueError(
			f'{csv_path} missing required columns: '
			f'need={[COL_NETWORK, COL_STATION]} got={list(df.columns)}'
		)

	d = df[[COL_NETWORK, COL_STATION]].copy()
	d[COL_NETWORK] = d[COL_NETWORK].map(_canon_network_code)
	d[COL_STATION] = d[COL_STATION].astype(str).str.strip()
	d = d[(d[COL_NETWORK] != '') & (d[COL_STATION] != '')]

	out: dict[str, set[str]] = {}
	for net, g in d.groupby(COL_NETWORK, sort=False):
		out[net] = set(g[COL_STATION].tolist())
	return out


def _load_station_meta(csv_path: Path) -> dict[str, dict[str, object]]:
	"""stations_*.csv から (net:station) -> メタ情報（座標など）を作る"""
	if not csv_path.is_file():
		raise FileNotFoundError(f'missing stations csv: {csv_path}')

	h = pd.read_csv(csv_path, nrows=0)
	cols = list(h.columns)

	need = [COL_NETWORK, COL_STATION, COL_LAT, COL_LON]
	missing = [c for c in need if c not in cols]
	if missing:
		raise ValueError(
			f'{csv_path} missing required columns for coords: '
			f'missing={missing} got={cols}'
		)

	usecols = [COL_NETWORK, COL_STATION, COL_LAT, COL_LON]
	if COL_STATION_NAME in cols:
		usecols.append(COL_STATION_NAME)
	if COL_ELEV_M in cols:
		usecols.append(COL_ELEV_M)

	df = pd.read_csv(csv_path, usecols=usecols, dtype=str)
	df[COL_NETWORK] = df[COL_NETWORK].map(_canon_network_code)
	df[COL_STATION] = df[COL_STATION].astype(str).str.strip()

	if COL_STATION_NAME in df.columns:
		df[COL_STATION_NAME] = df[COL_STATION_NAME].astype(str).str.strip()
	else:
		df[COL_STATION_NAME] = ''

	df[COL_LAT] = pd.to_numeric(df[COL_LAT], errors='raise')
	df[COL_LON] = pd.to_numeric(df[COL_LON], errors='raise')
	if COL_ELEV_M in df.columns:
		df[COL_ELEV_M] = pd.to_numeric(df[COL_ELEV_M], errors='raise')
	else:
		df[COL_ELEV_M] = pd.NA

	df = df[(df[COL_NETWORK] != '') & (df[COL_STATION] != '')]
	if df.empty:
		return {}

	df['__key'] = df[COL_NETWORK] + ':' + df[COL_STATION]

	out: dict[str, dict[str, object]] = {}
	for key, g in df.groupby('__key', sort=False):
		row0 = g.iloc[0]
		lat_vals = g[COL_LAT].unique()
		lon_vals = g[COL_LON].unique()
		if len(lat_vals) != 1 or len(lon_vals) != 1:
			raise ValueError(f'{csv_path} has inconsistent coords for key={key}')

		elev = None
		elev_vals = g[COL_ELEV_M].dropna().unique()
		if len(elev_vals) == 1:
			elev = float(elev_vals[0])
		elif len(elev_vals) > 1:
			raise ValueError(f'{csv_path} has inconsistent elevation for key={key}')

		out[key] = {
			'network_code': row0[COL_NETWORK],
			'station': row0[COL_STATION],
			'station_name': row0[COL_STATION_NAME],
			'lat': float(lat_vals[0]),
			'lon': float(lon_vals[0]),
			'elevation_m': elev,
		}
	return out


def _merge_station_meta(
	dst: dict[str, dict[str, object]],
	src: dict[str, dict[str, object]],
	*,
	max_jump_deg: float = 0.02,
	max_jump_m: float = 50000.0,
) -> None:
	"""同一 key を上書き更新する。

	座標が大きく飛ぶ場合はデータ破損の可能性が高いので即落とす。
	"""
	for key, m in src.items():
		if key not in dst:
			dst[key] = m
			continue

		d = dst[key]
		dlat = abs(float(d['lat']) - float(m['lat']))
		dlon = abs(float(d['lon']) - float(m['lon']))
		if dlat > max_jump_deg or dlon > max_jump_deg:
			raise ValueError(
				'conflicting station coords: '
				f'key={key} '
				f'dst(lat,lon)=({d["lat"]},{d["lon"]}) '
				f'src(lat,lon)=({m["lat"]},{m["lon"]}) '
				f'max_jump_deg={max_jump_deg}'
			)

		d_e = d.get('elevation_m')
		s_e = m.get('elevation_m')
		if d_e is not None and s_e is not None:
			delev = abs(float(d_e) - float(s_e))
			if delev > max_jump_m:
				raise ValueError(
					'conflicting station elevation: '
					f'key={key} dst={d_e} src={s_e} max_jump_m={max_jump_m}'
				)

		# src が欠損なら dst を引き継ぐ
		if not m.get('station_name') and d.get('station_name'):
			m['station_name'] = d['station_name']
		if m.get('elevation_m') is None and d.get('elevation_m') is not None:
			m['elevation_m'] = d['elevation_m']

		# 新しいメタ情報で上書き
		dst[key] = m


def _infer_when_from_path(p: Path) -> dt.datetime:
	m = _RE_TS12.search(p.stem)
	if m:
		return _parse_ts12(m.group(1))

	for part in p.parts[::-1]:
		m2 = _RE_TS12.fullmatch(part)
		if m2:
			return _parse_ts12(part)

	raise ValueError(f'cannot infer timestamp from path: {p}')


def _list_yearly_anchors() -> list[tuple[dt.datetime, Path]]:
	if not YEARLY_ROOT.is_dir():
		raise FileNotFoundError(f'YEARLY_ROOT not found: {YEARLY_ROOT}')

	files = sorted(YEARLY_ROOT.rglob('stations_*.csv'))
	if not files:
		raise ValueError(f'no stations_*.csv under: {YEARLY_ROOT}')

	anchors: list[tuple[dt.datetime, Path]] = []
	for p in files:
		when = _infer_when_from_path(p)
		anchors.append((when, p))

	anchors.sort(key=lambda x: x[0])
	return anchors


def _pick_latest_snapshot_dir(month_dir: Path) -> Path:
	# monthly/YYYY-MM/ 配下に複数スナップショットがある場合、最大 ts を採用
	subs = [p for p in month_dir.iterdir() if p.is_dir() and _RE_TS12.fullmatch(p.name)]
	if not subs:
		raise ValueError(f'no snapshot dirs (YYYYMMDDHHMM) under: {month_dir}')
	return max(subs, key=lambda p: int(p.name))


def _list_monthly_snapshots() -> dict[str, Path]:
	if not MONTHLY_ROOT.is_dir():
		raise FileNotFoundError(f'MONTHLY_ROOT not found: {MONTHLY_ROOT}')

	month_dirs = [
		p for p in MONTHLY_ROOT.iterdir() if p.is_dir() and _RE_MONTH_DIR.match(p.name)
	]
	month_dirs.sort(key=lambda p: p.name)

	out: dict[str, Path] = {}
	for md in month_dirs:
		snap = _pick_latest_snapshot_dir(md)
		stations = sorted(snap.glob('stations_*.csv'))
		if not stations:
			raise ValueError(f'no stations_*.csv under: {snap}')
		# 複数あれば最大 ts を採用
		stations.sort(key=lambda p: int(_RE_TS12.search(p.stem).group(1)))
		out[md.name] = stations[-1]
	return out


def _build_all_keys(
	yearly_sets: dict[dt.datetime, dict[str, set[str]]],
	monthly_sets: dict[str, dict[str, set[str]]],
) -> tuple[list[str], dict[str, list[int]], dict[str, int]]:
	keyset: set[str] = set()

	for _, by_net in yearly_sets.items():
		for net, stas in by_net.items():
			for sta in stas:
				keyset.add(f'{net}:{sta}')

	for _, by_net in monthly_sets.items():
		for net, stas in by_net.items():
			for sta in stas:
				keyset.add(f'{net}:{sta}')

	all_keys = sorted(keyset)

	key_to_idx: dict[str, int] = {k: i for i, k in enumerate(all_keys)}
	net_to_indices: dict[str, list[int]] = {}
	for i, k in enumerate(all_keys):
		net = k.split(':', 1)[0]
		net_to_indices.setdefault(net, []).append(i)

	return all_keys, net_to_indices, key_to_idx


def _anchor_bracket(
	anchors: list[dt.datetime], t: dt.datetime
) -> tuple[dt.datetime | None, dt.datetime | None]:
	# prev <= t < next
	prev: dt.datetime | None = None
	next_: dt.datetime | None = None
	for a in anchors:
		if a <= t:
			prev = a
		else:
			next_ = a
			break
	return prev, next_


def main() -> None:
	anchors = _list_yearly_anchors()
	anchor_times = [t for t, _ in anchors]

	# 年次アンカーの station set をロード
	yearly_sets: dict[dt.datetime, dict[str, set[str]]] = {}
	for when, p in anchors:
		yearly_sets[when] = _load_station_sets(p)

	# 月次 stations をロード（存在する月だけ）
	monthly_station_csv = _list_monthly_snapshots()
	monthly_sets: dict[str, dict[str, set[str]]] = {}
	for label, p in monthly_station_csv.items():
		monthly_sets[label] = _load_station_sets(p)

	# stations_*.csv から座標メタ情報を集約（monthly_presence.csv に埋め込む）
	meta_sources: list[tuple[dt.datetime, Path]] = []
	for when, p in anchors:
		meta_sources.append((when, p))
	for p in monthly_station_csv.values():
		meta_sources.append((_infer_when_from_path(p), p))
	meta_sources.sort(key=lambda x: x[0])

	station_meta: dict[str, dict[str, object]] = {}
	for _, p in meta_sources:
		_merge_station_meta(station_meta, _load_station_meta(p))

	# 月列の範囲：monthly の月 + yearly の範囲 を全部カバー
	min_month = None
	max_month = None

	if monthly_station_csv:
		mm = [dt.date(int(k[0:4]), int(k[5:7]), 1) for k in monthly_station_csv.keys()]
		min_month = min(mm)
		max_month = max(mm)

	min_anchor_month = _floor_month(anchor_times[0])
	max_anchor_month = _floor_month(anchor_times[-1])

	if min_month is None or min_anchor_month < min_month:
		min_month = min_anchor_month
	if max_month is None or max_anchor_month > max_month:
		max_month = max_anchor_month

	months = _month_range(min_month, max_month)
	month_labels = [_month_label(m) for m in months]

	all_keys, net_to_indices, key_to_idx = _build_all_keys(yearly_sets, monthly_sets)
	if not all_keys:
		raise ValueError('no station keys found from yearly/monthly')

	# presence table 初期化（NAで埋める）
	nets: list[str] = []
	stas: list[str] = []
	for k in all_keys:
		net, sta = k.split(':', 1)
		nets.append(net)
		stas.append(sta)

	keys = [f'{n}:{s}' for n, s in zip(nets, stas)]
	missing_meta = [k for k in keys if k not in station_meta]
	if missing_meta:
		raise ValueError(
			f'missing station meta for {len(missing_meta)} keys '
			f'(example: {missing_meta[:5]})'
		)

	presence = pd.DataFrame(
		{
			'network_code': nets,
			'station': stas,
			'station_name': [station_meta[k].get('station_name', '') for k in keys],
			'lat': [station_meta[k]['lat'] for k in keys],
			'lon': [station_meta[k]['lon'] for k in keys],
			'elevation_m': [station_meta[k].get('elevation_m') for k in keys],
		}
	)
	for label in month_labels:
		presence[label] = pd.Series([pd.NA] * len(all_keys), dtype='Int8')

	# 各月を構築：monthly 観測（上書き） + yearly 推定（前後アンカー一致のネットのみ）
	for mdate, label in zip(months, month_labels):
		t = dt.datetime(mdate.year, mdate.month, 1, 0, 0)
		prev_a, next_a = _anchor_bracket(anchor_times, t)

		observed_by_net = monthly_sets.get(label, {})
		observed_nets = set(observed_by_net.keys())

		inferred_by_net: dict[str, set[str]] = {}
		inferred_nets: set[str] = set()

		if prev_a is not None:
			prev_sets = yearly_sets[prev_a]
			if next_a is not None:
				next_sets = yearly_sets[next_a]
				cand_nets = set(prev_sets.keys()) | set(next_sets.keys())
				for net in cand_nets:
					if (
						net in prev_sets
						and net in next_sets
						and prev_sets[net] == next_sets[net]
					):
						inferred_by_net[net] = prev_sets[net]
						inferred_nets.add(net)
			else:
				# 最終アンカー以降（月列は最終アンカー月までなので実質1回くらい）
				for net, stas_set in prev_sets.items():
					inferred_by_net[net] = stas_set
					inferred_nets.add(net)

		# この月に「確定で0/1を入れていい」ネットワーク
		known_nets = inferred_nets | observed_nets
		if not known_nets:
			continue

		col = label

		# まず known_nets の全stationを 0 で埋める（= その月の不在を 0 として確定させる）
		for net in known_nets:
			idxs = net_to_indices.get(net)
			if idxs:
				presence.iloc[idxs, presence.columns.get_loc(col)] = 0

		# 次に present を 1 で上書き（monthly があればそっち優先）
		def mark_present(net: str, stas_set: set[str]) -> None:
			idxs = []
			for sta in stas_set:
				k = f'{net}:{sta}'
				i = key_to_idx.get(k)
				if i is not None:
					idxs.append(i)
			if idxs:
				presence.iloc[idxs, presence.columns.get_loc(col)] = 1

		for net, stas_set in inferred_by_net.items():
			if net not in observed_nets:
				mark_present(net, stas_set)

		for net, stas_set in observed_by_net.items():
			mark_present(net, stas_set)

	MONTHLY_ROOT.mkdir(parents=True, exist_ok=True)
	presence.to_csv(OUT_PRESENCE, index=False, encoding='utf-8')
	print(f'[INFO] wrote: {OUT_PRESENCE}')

	# lifespans（1の連続区間。NA は “未観測” なので区間を切る）
	mat = presence[month_labels].to_numpy()
	liferows: list[dict[str, str]] = []

	for i, k in enumerate(all_keys):
		net, sta = k.split(':', 1)
		row = mat[i, :]

		run_start: int | None = None
		for j in range(len(months)):
			v = row[j]
			if pd.isna(v):
				# 未観測は区間を切る
				if run_start is not None:
					start_m = months[run_start]
					end_excl = months[j]
					liferows.append(
						{
							'network_code': net,
							'station': sta,
							'start_month': _month_label(start_m),
							'end_month': _month_label(_add_months(end_excl, -1)),
							'start_date': start_m.isoformat(),
							'end_date': end_excl.isoformat(),
						}
					)
					run_start = None
				continue

			vv = int(v)
			if vv == 1 and run_start is None:
				run_start = j
			if vv == 0 and run_start is not None:
				start_m = months[run_start]
				end_excl = months[j]
				liferows.append(
					{
						'network_code': net,
						'station': sta,
						'start_month': _month_label(start_m),
						'end_month': _month_label(_add_months(end_excl, -1)),
						'start_date': start_m.isoformat(),
						'end_date': end_excl.isoformat(),
					}
				)
				run_start = None

		if run_start is not None:
			start_m = months[run_start]
			liferows.append(
				{
					'network_code': net,
					'station': sta,
					'start_month': _month_label(start_m),
					'end_month': '',
					'start_date': start_m.isoformat(),
					'end_date': '',
				}
			)

	lifespans = pd.DataFrame(
		liferows,
		columns=[
			'network_code',
			'station',
			'start_month',
			'end_month',
			'start_date',
			'end_date',
		],
	)
	lifespans.to_csv(OUT_LIFESPANS, index=False, encoding='utf-8')
	print(f'[INFO] wrote: {OUT_LIFESPANS}')


if __name__ == '__main__':
	main()
