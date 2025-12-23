# %%
from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path

import pandas as pd

from jma.chk_network_station import (
	export_channels_from_probe_ch_dirs,
	export_station_summary_from_channels,
)
from jma.chk_newwork_downloadable import probe_networks_by_get_continuous_waveform

# ==========================
# 設定（ここだけ直に編集）
# ==========================
NETWORK_INFO_CSV = Path(
	'/workspace/data/station/network_info.csv'
)  # columns: network_code, network_name
OUT_ROOT = Path(
	'snapshots/yearly'
)  # snapshots/yearly/YYYY/ で運用（年フォルダ直下に日時サブフォルダを切る）

START_YEAR = 2011
END_YEAR = 2025

# 毎年この時刻でスナップショット（timezone-naive。JST想定）
CHECK_MONTH = 1
CHECK_DAY = 1
CHECK_HOUR = 0
CHECK_MINUTE = 0

# 追加スナップショット（指定どおり）
EXTRA_WHENS = [
	# dt.datetime(2004, 4, 1, 0, 0),
	dt.datetime(2025, 12, 1, 0, 0),
]

SPAN_MIN = 1
THREADS = 4
CLEANUP = True
KEEP_CNT = False

SKIP_IF_EXISTS = (
	False  # その日時フォルダに stations/channels/probe_results が揃ってたらスキップ
)


def load_network_info_csv(path: Path) -> dict[str, str]:
	if not path.is_file():
		raise FileNotFoundError(f'network_info csv not found: {path}')

	df = pd.read_csv(path)
	required = ['network_code', 'network_name']
	missing = [c for c in required if c not in df.columns]
	if missing:
		raise ValueError(
			f'network_info csv missing columns: {missing}; got={list(df.columns)}'
		)

	out: dict[str, str] = {}
	for _, r in df.iterrows():
		code = str(r['network_code']).strip()
		name = str(r['network_name']).strip()
		if not code:
			raise ValueError('empty network_code in network_info csv')
		out[code] = name

	if not out:
		raise ValueError('network_info csv parsed empty')
	return out


def ok_mask(series: pd.Series) -> pd.Series:
	s = series.copy()
	if s.dtype == bool:
		return s
	return s.astype(str).str.strip().str.lower().isin(['true', '1', 't', 'yes', 'y'])


def remove_failed_network_dirs(probe_root: Path, probe_df: pd.DataFrame) -> None:
	if 'network_code' not in probe_df.columns or 'ok' not in probe_df.columns:
		raise ValueError(
			f'probe_df must contain columns: network_code, ok; got={list(probe_df.columns)}'
		)

	m = ok_mask(probe_df['ok'])
	failed_codes = probe_df.loc[~m, 'network_code'].astype(str).tolist()

	for code in failed_codes:
		d = probe_root / code
		if d.exists():
			shutil.rmtree(d)


def build_snapshot_whens() -> list[dt.datetime]:
	whens: list[dt.datetime] = []
	for year in range(int(START_YEAR), int(END_YEAR) + 1):
		whens.append(
			dt.datetime(
				year,
				int(CHECK_MONTH),
				int(CHECK_DAY),
				int(CHECK_HOUR),
				int(CHECK_MINUTE),
			)
		)
	whens.extend(EXTRA_WHENS)

	# unique + sort
	uniq = sorted(set(whens))
	return uniq


def snapshot_dir_for(when: dt.datetime) -> Path:
	# 年フォルダの下に日時サブフォルダを作る（同一年に複数スナップショットがあっても衝突しない）
	stamp = when.strftime('%Y%m%d%H%M')
	return OUT_ROOT / f'{when.year:04d}' / stamp


def main() -> None:
	network_info = load_network_info_csv(NETWORK_INFO_CSV)

	OUT_ROOT.mkdir(parents=True, exist_ok=True)

	summary_rows: list[dict[str, object]] = []

	for when in build_snapshot_whens():
		snap_dir = snapshot_dir_for(when)
		snap_dir.mkdir(parents=True, exist_ok=True)

		probe_root = snap_dir / 'probe'
		probe_root.mkdir(parents=True, exist_ok=True)

		stamp = when.strftime('%Y%m%d%H%M')
		channels_csv = snap_dir / f'channels_{stamp}.csv'
		stations_csv = snap_dir / f'stations_{stamp}.csv'
		probe_results_csv = snap_dir / f'probe_results_{stamp}.csv'

		if (
			SKIP_IF_EXISTS
			and channels_csv.exists()
			and stations_csv.exists()
			and probe_results_csv.exists()
		):
			print(f'[INFO] skip existing: {snap_dir}')
			continue

		probe_df = probe_networks_by_get_continuous_waveform(
			network_info=network_info,
			when=when,
			base_outdir=probe_root,
			span_min=int(SPAN_MIN),
			threads=int(THREADS),
			cleanup=bool(CLEANUP),
			keep_cnt=bool(KEEP_CNT),
		)
		probe_df.to_csv(probe_results_csv, index=False, encoding='utf-8')

		remove_failed_network_dirs(probe_root, probe_df)

		ch_df = export_channels_from_probe_ch_dirs(
			base_probe_dir=probe_root,
			out_csv=channels_csv,
		)
		st_df = export_station_summary_from_channels(
			ch_df,
			out_csv=stations_csv,
		)

		m_ok = ok_mask(probe_df['ok'])
		n_ok = int(m_ok.sum())
		n_all = len(probe_df)
		n_station = int(st_df.shape[0])

		summary_rows.append(
			{
				'when': when.strftime('%Y-%m-%d %H:%M'),
				'stamp': stamp,
				'year': when.year,
				'n_network_total': n_all,
				'n_network_ok': n_ok,
				'n_network_fail': n_all - n_ok,
				'n_station': n_station,
				'snapshot_dir': str(snap_dir),
			}
		)

		print(
			f'[INFO] done: {when.strftime("%Y-%m-%d %H:%M")}  networks ok={n_ok}/{n_all}  stations={n_station}'
		)

	if summary_rows:
		summary_df = (
			pd.DataFrame(summary_rows)
			.sort_values(['when', 'snapshot_dir'], kind='stable')
			.reset_index(drop=True)
		)
		summary_csv = OUT_ROOT / 'yearly_scan_summary.csv'
		summary_df.to_csv(summary_csv, index=False, encoding='utf-8')
		print(f'[INFO] wrote: {summary_csv}')


if __name__ == '__main__':
	main()
