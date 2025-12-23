# %%
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from jma.station_reader import read_hinet_channel_table

_COMMENT_RE = re.compile(
	r'^\s*#\s*(?P<sta>[A-Za-z0-9]+\.[A-Za-z0-9]+)\s+(?P<name>.+?)\s*$'
)


def parse_station_names_from_ch(ch_path: str | Path) -> dict[str, str]:
	"""`.ch` のコメント行 `# tt.wkg Niijimawakago` から station->name を作る。

	返り値のキーはデータ行と合わせて大文字 station（例: 'TT.WKG'）
	"""
	ch_path = Path(ch_path)
	if not ch_path.is_file():
		raise FileNotFoundError(f'.ch not found: {ch_path}')

	out: dict[str, str] = {}
	for line in ch_path.read_text(encoding='utf-8', errors='ignore').splitlines():
		m = _COMMENT_RE.match(line)
		if not m:
			continue
		sta = m.group('sta').strip().upper()
		name = m.group('name').strip()
		if sta and name:
			out[sta] = name
	return out


def _pick_one_ch_file(network_dir: Path) -> Path:
	"""ネットワーク配下から `.ch` を1つ選ぶ（probe_*.ch優先、無ければ*.chのうち最新）。"""
	ch_probe = sorted(network_dir.glob('probe_*.ch'))
	if ch_probe:
		return ch_probe[-1]

	ch_all = sorted(network_dir.glob('*.ch'))
	if not ch_all:
		raise FileNotFoundError(f'no .ch files found under: {network_dir}')

	# mtimeで最新
	ch_all_sorted = sorted(ch_all, key=lambda p: p.stat().st_mtime)
	return ch_all_sorted[-1]


def export_channels_from_probe_ch_dirs(
	*,
	base_probe_dir: str | Path,
	out_csv: str | Path,
) -> pd.DataFrame:
	"""base_probe_dir/<network_code>/*.ch を走査し、networkごとのchを読み込んでCSV化する。

	出力DF（行=チャンネル）:
	- network_code
	- station, component
	- channel_id (例: TT.WKG.U)
	- station_name（コメント行から取れたときのみ、無いなら空）
	- lat, lon, elevation_m など read_hinet_channel_table の列
	- ch_path（元ファイル）
	"""
	base_probe_dir = Path(base_probe_dir)
	if not base_probe_dir.is_dir():
		raise FileNotFoundError(f'base_probe_dir not found: {base_probe_dir}')

	rows: list[pd.DataFrame] = []

	for network_dir in sorted([p for p in base_probe_dir.iterdir() if p.is_dir()]):
		print(f'Processing network dir: {network_dir}')
		network_code = network_dir.name.strip()
		if not network_code:
			continue

		ch_path = _pick_one_ch_file(network_dir)
		try:
			df = read_hinet_channel_table(ch_path)
		except Exception as e:
			raise RuntimeError(
				f'failed to read_hinet_channel_table for network_code={network_code} at {ch_path}: {e}'
			) from e
		station_name_map = parse_station_names_from_ch(ch_path)

		df = df.copy()
		df.insert(0, 'network_code', network_code)
		df['station'] = df['station'].astype(str).str.strip()
		df['component'] = df['component'].astype(str).str.strip()
		df['channel_id'] = df['station'].astype(str) + '.' + df['component'].astype(str)

		df['station_name'] = df['station'].map(station_name_map).fillna('')
		df['ch_path'] = str(ch_path)

		# 出力列（必要なら増やしてOK）
		cols = [
			'network_code',
			'station',
			'station_name',
			'component',
			'channel_id',
			'lat',
			'lon',
			'elevation_m',
			'input_unit',
			'rec_flag',
			'adc_bits',
			'sensor_sensitivity',
			'preamp_gain_db',
			'ad_lsb_delta_v',
			'conv_coeff',
			'ch_hex',
			'ch_int',
			'ch_path',
		]
		missing = [c for c in cols if c not in df.columns]
		if missing:
			raise ValueError(
				f'unexpected columns missing from read_hinet_channel_table: {missing}'
			)

		rows.append(df[cols])

	if not rows:
		raise ValueError(f'no network dirs with .ch found under: {base_probe_dir}')

	out_df = pd.concat(rows, ignore_index=True)

	out_csv = Path(out_csv)
	out_csv.parent.mkdir(parents=True, exist_ok=True)
	out_df.to_csv(out_csv, index=False, encoding='utf-8')
	return out_df


def export_station_summary_from_channels(
	channels_df: pd.DataFrame,
	*,
	out_csv: str | Path,
) -> pd.DataFrame:
	"""上のチャンネルCSVから station単位に集約したサマリCSVも作る（任意）。"""
	for c in (
		'network_code',
		'station',
		'component',
		'station_name',
		'lat',
		'lon',
		'elevation_m',
	):
		if c not in channels_df.columns:
			raise ValueError(f'channels_df missing column: {c}')

	df = channels_df.copy()
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

	out_csv = Path(out_csv)
	out_csv.parent.mkdir(parents=True, exist_ok=True)
	g.to_csv(out_csv, index=False, encoding='utf-8')
	return g
