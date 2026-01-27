# file: src/jma/monthly_ch_index.py
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import pandas as pd

from jma.chk_network_station import parse_station_names_from_ch
from jma.station_reader import read_hinet_channel_table
from jma.stationcode_common import canon_network_code

_RE_NET_IN_NAME = re.compile(
	r'(?:^|[_-])(probe|win)_(?P<net>[0-9]{4,6}[A-Za-z]?)(?:[_-]|$)'
)
_RE_NET_DIR = re.compile(r'^[0-9]{4,6}[A-Za-z]?$')
_RE_MONTH_DIR = re.compile(r'^\d{4}-\d{2}$')


def _canon_station(x: object) -> str:
	s = '' if x is None else str(x)
	s = s.strip()
	if not s:
		return ''
	return s.upper()


def infer_month_label_from_snapshot_dir(snapshot_dir: str | Path) -> str:
	p = Path(snapshot_dir)
	for part in p.parts[::-1]:
		if _RE_MONTH_DIR.fullmatch(part):
			return part
	raise ValueError(f'cannot infer month label (YYYY-MM) from path: {p}')


def _infer_network_code_from_ch_path(ch_path: Path) -> str:
	p = Path(ch_path)

	m = _RE_NET_IN_NAME.search(p.stem)
	if m:
		return canon_network_code(m.group('net'))

	for part in p.parts[::-1]:
		if _RE_NET_DIR.fullmatch(part) and not re.fullmatch(r'\d{12}', part):
			return canon_network_code(part)

	raise ValueError(f'cannot infer network_code from ch_path: {p}')


def scan_channels_from_ch_files(
	*,
	ch_files: list[Path],
) -> pd.DataFrame:
	if not ch_files:
		raise ValueError('ch_files is empty')

	rows: list[pd.DataFrame] = []
	for ch_path in ch_files:
		ch_path = Path(ch_path)
		if not ch_path.is_file():
			raise FileNotFoundError(f'.ch not found: {ch_path}')

		network_code = _infer_network_code_from_ch_path(ch_path)
		df = read_hinet_channel_table(ch_path)

		name_map = parse_station_names_from_ch(ch_path)

		df = df.copy()
		df.insert(0, 'network_code', network_code)
		df['station'] = df['station'].map(_canon_station)
		df['component'] = df['component'].astype(str).str.strip()
		df['channel_id'] = df['station'].astype(str) + '.' + df['component'].astype(str)
		df['station_name'] = df['station'].map(name_map).fillna('')
		df['ch_path'] = str(ch_path)

		cols = [
			'network_code',
			'station',
			'station_name',
			'component',
			'channel_id',
			'lat',
			'lon',
			'elevation_m',
			'ch_hex',
			'ch_int',
			'ch_path',
		]
		missing = [c for c in cols if c not in df.columns]
		if missing:
			raise ValueError(f'unexpected columns missing in channel table: {missing}')

		rows.append(df[cols])

	out = pd.concat(rows, ignore_index=True)

	out['network_code'] = out['network_code'].map(canon_network_code)
	out['station'] = out['station'].map(_canon_station)
	out = out[(out['network_code'] != '') & (out['station'] != '')].reset_index(
		drop=True
	)
	if out.empty:
		raise ValueError('no valid channel rows after normalization')

	out = out.drop_duplicates(
		subset=['network_code', 'station', 'component', 'channel_id', 'ch_hex'],
		keep='first',
	).reset_index(drop=True)

	return out


def build_station_summary_from_channels(channels_df: pd.DataFrame) -> pd.DataFrame:
	required = [
		'network_code',
		'station',
		'component',
		'lat',
		'lon',
		'elevation_m',
		'station_name',
	]
	missing = [c for c in required if c not in channels_df.columns]
	if missing:
		raise ValueError(f'channels_df missing columns: {missing}')

	df = channels_df.copy()
	df['network_code'] = df['network_code'].map(canon_network_code)
	df['station'] = df['station'].map(_canon_station)

	g = (
		df.groupby(['network_code', 'station'], as_index=False)
		.agg(
			station_name=('station_name', 'first'),
			lat=('lat', 'first'),
			lon=('lon', 'first'),
			elevation_m=('elevation_m', 'first'),
			components=('component', lambda x: ','.join(sorted(set(map(str, x))))),
			n_components=('component', lambda x: len(set(map(str, x)))),
		)
		.sort_values(['network_code', 'station'])
		.reset_index(drop=True)
	)
	return g


def _assert_station_meta_consistent(
	*,
	old: pd.Series,
	new: pd.Series,
	max_jump_deg: float = 0.02,
	max_jump_m: float = 500.0,
) -> None:
	dlat = abs(float(old['lat']) - float(new['lat']))
	dlon = abs(float(old['lon']) - float(new['lon']))
	if dlat > max_jump_deg or dlon > max_jump_deg:
		raise ValueError(
			'conflicting station coords: '
			f'net={old["network_code"]} station={old["station"]} '
			f'old(lat,lon)=({old["lat"]},{old["lon"]}) '
			f'new(lat,lon)=({new["lat"]},{new["lon"]}) '
			f'max_jump_deg={max_jump_deg}'
		)

	oe = old.get('elevation_m')
	ne = new.get('elevation_m')
	if pd.notna(oe) and pd.notna(ne):
		de = abs(float(oe) - float(ne))
		if de > max_jump_m:
			raise ValueError(
				'conflicting station elevation: '
				f'net={old["network_code"]} station={old["station"]} '
				f'old={oe} new={ne} max_jump_m={max_jump_m}'
			)


def update_station_network_map_csv(
	*,
	stations_df: pd.DataFrame,
	out_csv: str | Path,
	month_label: str,
) -> pd.DataFrame:
	required = ['network_code', 'station', 'station_name', 'lat', 'lon', 'elevation_m']
	missing = [c for c in required if c not in stations_df.columns]
	if missing:
		raise ValueError(f'stations_df missing columns: {missing}')

	out_csv = Path(out_csv)
	out_csv.parent.mkdir(parents=True, exist_ok=True)

	new = stations_df.copy()
	new['network_code'] = new['network_code'].map(canon_network_code)
	new['station'] = new['station'].map(_canon_station)
	new['first_seen_month'] = month_label
	new['last_seen_month'] = month_label

	key_cols = ['network_code', 'station']
	keep_cols = key_cols + [
		'station_name',
		'lat',
		'lon',
		'elevation_m',
		'first_seen_month',
		'last_seen_month',
	]

	if not out_csv.is_file():
		out = new[keep_cols].copy()
		out.to_csv(out_csv, index=False, encoding='utf-8')
		return out

	old = pd.read_csv(out_csv, dtype=str)
	req_old = key_cols + [
		'lat',
		'lon',
		'elevation_m',
		'station_name',
		'first_seen_month',
		'last_seen_month',
	]
	missing_old = [c for c in req_old if c not in old.columns]
	if missing_old:
		raise ValueError(f'{out_csv} missing required columns: {missing_old}')

	old['network_code'] = old['network_code'].map(canon_network_code)
	old['station'] = old['station'].map(_canon_station)
	old['lat'] = pd.to_numeric(old['lat'], errors='raise')
	old['lon'] = pd.to_numeric(old['lon'], errors='raise')
	old['elevation_m'] = pd.to_numeric(old['elevation_m'], errors='coerce')

	new['lat'] = pd.to_numeric(new['lat'], errors='raise')
	new['lon'] = pd.to_numeric(new['lon'], errors='raise')
	new['elevation_m'] = pd.to_numeric(new['elevation_m'], errors='coerce')

	old['__key'] = old['network_code'] + ':' + old['station']
	new['__key'] = new['network_code'] + ':' + new['station']

	old_idx = {k: i for i, k in enumerate(old['__key'].tolist())}

	out = old.copy()
	for _, r in new.iterrows():
		k = r['__key']
		if k not in old_idx:
			out = pd.concat([out, r[old.columns].to_frame().T], ignore_index=True)
			continue

		i = old_idx[k]
		old_row = out.iloc[i]
		_assert_station_meta_consistent(old=old_row, new=r)

		if (not str(r['station_name']).strip()) and str(
			old_row['station_name']
		).strip():
			r['station_name'] = old_row['station_name']

		r_first = str(old_row['first_seen_month']).strip()
		if r_first:
			r['first_seen_month'] = r_first

		out.at[i, 'station_name'] = str(r['station_name'])
		out.at[i, 'lat'] = float(r['lat'])
		out.at[i, 'lon'] = float(r['lon'])
		out.at[i, 'elevation_m'] = (
			r['elevation_m'] if pd.notna(r['elevation_m']) else ''
		)
		out.at[i, 'first_seen_month'] = str(r['first_seen_month'])
		out.at[i, 'last_seen_month'] = month_label

	out = out.drop(columns=['__key']).copy()
	out = out.sort_values(['network_code', 'station']).reset_index(drop=True)
	out.to_csv(out_csv, index=False, encoding='utf-8')
	return out


def update_monthly_presence_csv(
	*,
	stations_df: pd.DataFrame,
	out_csv: str | Path,
	month_label: str,
	assume_complete_coverage_by_network: bool = True,
) -> pd.DataFrame:
	required = [
		'network_code',
		'station',
		'station_name',
		'lat',
		'lon',
		'elevation_m',
		'components',
		'n_components',
	]
	missing = [c for c in required if c not in stations_df.columns]
	if missing:
		raise ValueError(f'stations_df missing columns: {missing}')

	out_csv = Path(out_csv)
	out_csv.parent.mkdir(parents=True, exist_ok=True)

	new = stations_df.copy()
	new['network_code'] = new['network_code'].map(canon_network_code)
	new['station'] = new['station'].map(_canon_station)
	new['lat'] = pd.to_numeric(new['lat'], errors='raise')
	new['lon'] = pd.to_numeric(new['lon'], errors='raise')
	new['elevation_m'] = pd.to_numeric(new['elevation_m'], errors='coerce')
	new['n_components'] = pd.to_numeric(new['n_components'], errors='raise').astype(int)
	new['components'] = new['components'].astype(str).str.strip()
	new['station_name'] = new['station_name'].astype(str).str.strip()

	scanned_networks = sorted(set(new['network_code'].tolist()))
	if not scanned_networks:
		raise ValueError('no scanned networks in stations_df')

	key = new['network_code'] + ':' + new['station']
	new['__key'] = key

	if not out_csv.is_file():
		out = new[
			[
				'network_code',
				'station',
				'station_name',
				'lat',
				'lon',
				'elevation_m',
				'components',
				'n_components',
			]
		].copy()
		out[month_label] = pd.Series([1] * len(out), dtype='Int8')
		out = out.sort_values(['network_code', 'station']).reset_index(drop=True)
		out.to_csv(out_csv, index=False, encoding='utf-8')
		return out

	old = pd.read_csv(out_csv, dtype=str)
	meta_cols = [
		'network_code',
		'station',
		'station_name',
		'lat',
		'lon',
		'elevation_m',
		'components',
		'n_components',
	]
	missing_old = [c for c in meta_cols if c not in old.columns]
	if missing_old:
		raise ValueError(f'{out_csv} missing required columns: {missing_old}')

	old['network_code'] = old['network_code'].map(canon_network_code)
	old['station'] = old['station'].map(_canon_station)
	old['lat'] = pd.to_numeric(old['lat'], errors='raise')
	old['lon'] = pd.to_numeric(old['lon'], errors='raise')
	old['elevation_m'] = pd.to_numeric(old['elevation_m'], errors='coerce')
	old['n_components'] = pd.to_numeric(old['n_components'], errors='raise').astype(int)
	old['components'] = old['components'].astype(str).str.strip()
	old['station_name'] = old['station_name'].astype(str).str.strip()

	if month_label not in old.columns:
		old[month_label] = pd.Series([pd.NA] * len(old), dtype='Int8')
	else:
		old[month_label] = pd.to_numeric(old[month_label], errors='coerce').astype(
			'Int8'
		)

	old['__key'] = old['network_code'] + ':' + old['station']
	old_idx = {k: i for i, k in enumerate(old['__key'].tolist())}

	if assume_complete_coverage_by_network:
		mask_net = old['network_code'].isin(scanned_networks)
		old.loc[mask_net, month_label] = 0

	for _, r in new.iterrows():
		k = r['__key']
		if k not in old_idx:
			row = {
				'network_code': r['network_code'],
				'station': r['station'],
				'station_name': r['station_name'],
				'lat': float(r['lat']),
				'lon': float(r['lon']),
				'elevation_m': r['elevation_m'] if pd.notna(r['elevation_m']) else '',
				'components': r['components'],
				'n_components': int(r['n_components']),
				month_label: 1,
			}
			for c in old.columns:
				if c in row:
					continue
				if c == '__key':
					continue
				if c not in meta_cols and c != month_label:
					row[c] = pd.NA
			old = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
			old_idx[k] = len(old) - 1
			continue

		i = old_idx[k]
		old_row = old.iloc[i]
		_assert_station_meta_consistent(old=old_row, new=r)

		if (not str(r['station_name']).strip()) and str(
			old_row['station_name']
		).strip():
			r['station_name'] = old_row['station_name']

		old.at[i, 'station_name'] = str(r['station_name'])
		old.at[i, 'components'] = str(r['components'])
		old.at[i, 'n_components'] = int(r['n_components'])
		old.at[i, month_label] = 1

	old = old.drop(columns=['__key']).copy()
	old = old.sort_values(['network_code', 'station']).reset_index(drop=True)
	old.to_csv(out_csv, index=False, encoding='utf-8')
	return old


def index_snapshot_dir(
	*,
	snapshot_dir: str | Path,
	month_label: str | None = None,
	stamp: str | None = None,
	out_channels_csv: str | Path | None = None,
	out_stations_csv: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	snapshot_dir = Path(snapshot_dir)
	if not snapshot_dir.is_dir():
		raise FileNotFoundError(f'snapshot_dir not found: {snapshot_dir}')

	ch_files = sorted(snapshot_dir.rglob('*.ch'))
	if not ch_files:
		raise ValueError(f'no .ch files under: {snapshot_dir}')

	ch_df = scan_channels_from_ch_files(ch_files=ch_files)
	st_df = build_station_summary_from_channels(ch_df)

	if month_label is None:
		month_label = infer_month_label_from_snapshot_dir(snapshot_dir)

	if stamp is None:
		stamp = dt.datetime.now().strftime('%Y%m%d%H%M')

	if out_channels_csv is not None:
		p = Path(out_channels_csv)
		p.parent.mkdir(parents=True, exist_ok=True)
		ch_df.to_csv(p, index=False, encoding='utf-8')

	if out_stations_csv is not None:
		p = Path(out_stations_csv)
		p.parent.mkdir(parents=True, exist_ok=True)
		st_df.to_csv(p, index=False, encoding='utf-8')

	return ch_df, st_df
