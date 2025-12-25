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
CANDIDATES_CSV = Path('snapshots/yearly/refine_candidates.csv')  # input
NETWORK_INFO_CSV = Path(
	'/workspace/data/station/jma/network_info.csv'
)  # columns: network_code, network_name

OUT_ROOT = Path('snapshots/monthly')  # outputs: snapshots/monthly/YYYY-MM/<stamp>/...

# 月次スナップショット時刻（JST想定のnaiveで統一）
CHECK_HOUR = 0
CHECK_MINUTE = 0

SPAN_MIN = 1
THREADS = 4
CLEANUP = True
KEEP_CNT = False

SKIP_IF_EXISTS = True  # 同じstampの成果物が揃ってたらスキップ


COL_NETWORK = 'network_code'
COL_START = 'start'
COL_END = 'end'


def load_network_info_csv(path: Path) -> dict[str, str]:
	if not path.is_file():
		raise FileNotFoundError(f'network_info csv not found: {path}')

	df = pd.read_csv(path, dtype=str)
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


def parse_date_iso(s: str, field: str) -> dt.date:
	ss = str(s).strip()
	if not ss:
		raise ValueError(f'empty date for {field}')
	try:
		return dt.date.fromisoformat(ss)
	except ValueError as e:
		raise ValueError(f'invalid ISO date for {field}: {ss}') from e


def month_starts_between(start: dt.date, end: dt.date) -> list[dt.date]:
	# [start, end) の範囲を「各月1日」で列挙（end は排他的）
	if end <= start:
		raise ValueError(f'end must be > start: start={start} end={end}')

	cur = dt.date(start.year, start.month, 1)
	out: list[dt.date] = []
	while cur < end:
		out.append(cur)
		if cur.month == 12:
			cur = dt.date(cur.year + 1, 1, 1)
		else:
			cur = dt.date(cur.year, cur.month + 1, 1)
	return out


def load_candidates(path: Path) -> pd.DataFrame:
	if not path.is_file():
		raise FileNotFoundError(f'refine_candidates not found: {path}')

	df = pd.read_csv(path, dtype=str)
	required = [COL_NETWORK, COL_START, COL_END]
	missing = [c for c in required if c not in df.columns]
	if missing:
		raise ValueError(
			f'refine_candidates missing columns: {missing}; got={list(df.columns)}'
		)

	df = df[required].copy()
	df[COL_NETWORK] = df[COL_NETWORK].astype(str).str.strip()
	df[COL_START] = df[COL_START].astype(str).str.strip()
	df[COL_END] = df[COL_END].astype(str).str.strip()

	if (df[COL_NETWORK] == '').any():
		raise ValueError('refine_candidates contains empty network_code')
	if (df[COL_START] == '').any() or (df[COL_END] == '').any():
		raise ValueError('refine_candidates contains empty start/end')

	return df.drop_duplicates().reset_index(drop=True)


def build_month_to_networks(cand: pd.DataFrame) -> dict[dt.date, set[str]]:
	m2n: dict[dt.date, set[str]] = {}
	for _, r in cand.iterrows():
		code = str(r[COL_NETWORK]).strip()
		start = parse_date_iso(str(r[COL_START]), 'start')
		end = parse_date_iso(str(r[COL_END]), 'end')
		for m in month_starts_between(start, end):
			if m not in m2n:
				m2n[m] = set()
			m2n[m].add(code)
	return m2n


def month_key(m: dt.date) -> str:
	return f'{m.year:04d}-{m.month:02d}'


def snapshot_dir_for(m: dt.date) -> tuple[dt.datetime, Path]:
	when = dt.datetime(m.year, m.month, 1, int(CHECK_HOUR), int(CHECK_MINUTE))
	stamp = when.strftime('%Y%m%d%H%M')
	return when, OUT_ROOT / month_key(m) / stamp


def main() -> None:
	network_info_all = load_network_info_csv(NETWORK_INFO_CSV)
	cand = load_candidates(CANDIDATES_CSV)

	m2n = build_month_to_networks(cand)
	if not m2n:
		raise ValueError('no month targets derived from refine_candidates')

	OUT_ROOT.mkdir(parents=True, exist_ok=True)
	summary_rows: list[dict[str, object]] = []

	for m in sorted(m2n.keys()):
		codes = sorted(m2n[m])
		if not codes:
			continue

		missing = [c for c in codes if c not in network_info_all]
		if missing:
			raise ValueError(f'network_info missing codes: {missing}')

		network_info = {c: network_info_all[c] for c in codes}

		when, snap_dir = snapshot_dir_for(m)
		snap_dir.mkdir(parents=True, exist_ok=True)

		probe_root = snap_dir / 'probe'
		probe_root.mkdir(parents=True, exist_ok=True)

		stamp = when.strftime('%Y%m%d%H%M')
		channels_csv = snap_dir / f'channels_{stamp}.csv'
		stations_csv = snap_dir / f'stations_{stamp}.csv'
		probe_results_csv = snap_dir / f'probe_results_{stamp}.csv'
		networks_csv = snap_dir / f'networks_{stamp}.csv'

		if (
			SKIP_IF_EXISTS
			and channels_csv.exists()
			and stations_csv.exists()
			and probe_results_csv.exists()
		):
			print(f'[INFO] skip existing: {snap_dir}')
			continue

		pd.DataFrame(
			[{'network_code': c, 'network_name': network_info[c]} for c in codes],
			dtype=str,
		).to_csv(networks_csv, index=False, encoding='utf-8')

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
				'month': month_key(m),
				'when': when.strftime('%Y-%m-%d %H:%M'),
				'n_network_target': len(codes),
				'n_network_total_in_probe_df': n_all,
				'n_network_ok': n_ok,
				'n_network_fail': n_all - n_ok,
				'n_station': n_station,
				'snapshot_dir': str(snap_dir),
			}
		)

		print(
			f'[INFO] done: {month_key(m)}  networks ok={n_ok}/{n_all}  stations={n_station}'
		)

	if summary_rows:
		summary_df = (
			pd.DataFrame(summary_rows)
			.sort_values(['month', 'when'], kind='stable')
			.reset_index(drop=True)
		)
		summary_csv = OUT_ROOT / 'monthly_scan_summary.csv'
		summary_df.to_csv(summary_csv, index=False, encoding='utf-8')
		print(f'[INFO] wrote: {summary_csv}')


if __name__ == '__main__':
	main()
