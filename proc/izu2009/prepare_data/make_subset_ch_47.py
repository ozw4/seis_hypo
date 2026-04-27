# %%
from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

from jma.prepare.station_subset_ch import (
	read_station_list_txt,
	write_station_subset_ch_dir,
)
from jma.station_reader import read_hinet_channel_table


def _sha256_hex(text: str) -> str:
	return hashlib.sha256(text.encode('utf-8')).hexdigest()


def ch_signatures(ch_path: Path) -> tuple[str, str, int, int]:
	"""return:
	sig_channels: ch_hex 集合が同じなら一致（コメント差は無視）
	sig_full: チャンネル行の主要情報まで一致（より厳密）
	n_channels: 行数
	n_stations: station ユニーク数

	"""
	df = read_hinet_channel_table(ch_path)

	n_channels = len(df)
	n_stations = int(df['station'].nunique())

	# (1) チャンネルID集合ベース：実運用で「同じ .ch」と見なすならこれが最重要
	ch_list_sorted = sorted(str(x).strip().upper() for x in df['ch_hex'].tolist())
	sig_channels = _sha256_hex('\n'.join(ch_list_sorted))

	# (2) 主要列まで含める：座標や成分まで同一かチェックしたい場合
	df2 = df[
		['ch_int', 'ch_hex', 'station', 'component', 'lat', 'lon', 'elevation_m']
	].copy()

	df2['ch_hex'] = df2['ch_hex'].astype(str).str.strip().str.upper()
	df2['station'] = df2['station'].astype(str).str.strip().str.upper()
	df2['component'] = df2['component'].astype(str).str.strip().str.upper()

	df2 = df2.sort_values(['ch_int', 'component']).reset_index(drop=True)

	lines: list[str] = []
	for r in df2.itertuples(index=False):
		lines.append(
			f'{r.ch_hex}\t{r.station}\t{r.component}\t{float(r.lat):.6f}\t{float(r.lon):.6f}\t{float(r.elevation_m):.1f}'
		)

	sig_full = _sha256_hex('\n'.join(lines))

	return sig_channels, sig_full, n_channels, n_stations


def report_ch_uniqueness(out_base_dir: Path, report_csv: Path) -> None:
	ch_files = sorted(out_base_dir.rglob('*.ch'))
	if not ch_files:
		raise FileNotFoundError(f'no .ch files under: {out_base_dir}')

	rows: list[tuple[str, str, str, int, int]] = []
	groups = defaultdict(list)

	for p in ch_files:
		sig_channels, sig_full, n_ch, n_sta = ch_signatures(p)
		rows.append((str(p), sig_channels, sig_full, n_ch, n_sta))
		groups[sig_channels].append(p)

	report_csv.parent.mkdir(parents=True, exist_ok=True)
	with open(report_csv, 'w', newline='', encoding='utf-8') as f:
		w = csv.writer(f)
		w.writerow(['path', 'sig_channels', 'sig_full', 'n_channels', 'n_stations'])
		w.writerows(rows)

	dup_groups = [(sig, ps) for sig, ps in groups.items() if len(ps) >= 2]
	dup_groups.sort(key=lambda x: len(x[1]), reverse=True)

	print(f'[report] .ch files={len(ch_files)} unique(sig_channels)={len(groups)}')
	print(f'[report] csv={report_csv}')

	if not dup_groups:
		print('[report] no duplicates by sig_channels')
		return

	print('[report] duplicates by sig_channels (same channel-id set):')
	for sig, ps in dup_groups:
		print(f'  sig_channels={sig} n_files={len(ps)}')
		for p in ps:
			print(f'    {p}')


if __name__ == '__main__':
	base_cont_dir = Path('../../../data/izu2009/continuous')  # DL済み .ch がある場所
	stations_dir = Path('./profile/stations47')  # stations_0101.txt 等がある場所
	out_base_dir = Path('./download_continuous/continuous_ch47')  # 出力先（新規）

	networks = ['0101', '0203', '0207', '0301']
	# =======================

	for net in networks:
		in_dir = base_cont_dir / net
		out_dir = out_base_dir / net
		sta_txt = stations_dir / f'stations_{net}.txt'

		keep_stations = read_station_list_txt(sta_txt)

		write_station_subset_ch_dir(
			in_dir=in_dir,
			out_dir=out_dir,
			keep_stations=keep_stations,
			pattern='*.ch',
			skip_if_exists=True,
		)

	report_ch_uniqueness(
		out_base_dir=out_base_dir,
		report_csv=out_base_dir / '_report_ch_uniqueness.csv',
	)
