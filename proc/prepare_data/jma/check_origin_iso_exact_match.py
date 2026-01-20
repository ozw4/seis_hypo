# %%
# proc/prepare_data/jma/check_origin_time_match_strict.py
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from jma.prepare.event_txt import read_origin_jst_iso
from jma.prepare.event_paths import resolve_evt_and_txt

# =========================
# 設定（直書き）
# =========================
WIN_EVENT_DIR = Path('/workspace/data/waveform/jma/event').resolve()
EVENT_DIR_GLOB = 'D202301*'
EPI_CSV = Path(
	'/workspace/data/arrivetime/JMA/arrivetime_epicenters_2023.0.csv'
).resolve()

OUT_CSV = Path(
	'/workspace/proc/prepare_data/jma/_tmp/origin_time_strict_match_check.csv'
).resolve()
MAX_PRINT_EXAMPLES = 50


def _iso_to_ns(origin_iso: str) -> int:
	t = pd.to_datetime(origin_iso, format='ISO8601', errors='raise')
	dt64 = np.datetime64(t.to_datetime64())
	return int(dt64.astype('datetime64[ns]').astype('int64'))


def _build_ns_to_rows(
	epi_df: pd.DataFrame,
) -> tuple[dict[int, list[int]], np.ndarray, np.ndarray]:
	req = {'event_id', 'origin_time'}
	if not req.issubset(epi_df.columns):
		raise ValueError(
			f'epicenters csv missing columns: {sorted(req - set(epi_df.columns))}'
		)

	origin_str = epi_df['origin_time'].astype(str)
	dt64 = pd.to_datetime(origin_str, format='ISO8601', errors='raise').to_numpy(
		dtype='datetime64[ns]'
	)
	ns = dt64.astype('int64')

	event_id = epi_df['event_id'].astype(int).to_numpy()
	idx = np.arange(len(epi_df), dtype=int)

	ns_to_rows: dict[int, list[int]] = {}
	for i, k in enumerate(ns.tolist()):
		kk = int(k)
		ns_to_rows.setdefault(kk, []).append(int(idx[i]))

	return ns_to_rows, ns, event_id


def main() -> None:
	if not WIN_EVENT_DIR.is_dir():
		raise FileNotFoundError(WIN_EVENT_DIR)
	if not EPI_CSV.is_file():
		raise FileNotFoundError(EPI_CSV)

	epi_df = pd.read_csv(EPI_CSV, low_memory=False)
	ns_to_rows, epi_ns, epi_event_id = _build_ns_to_rows(epi_df)

	dup_keys = [k for k, rows in ns_to_rows.items() if len(rows) > 1]
	dup_keys_sorted = sorted(dup_keys)

	event_dirs = sorted([p for p in WIN_EVENT_DIR.glob(EVENT_DIR_GLOB) if p.is_dir()])
	if not event_dirs:
		raise RuntimeError(f'no event dirs matched: {WIN_EVENT_DIR}/{EVENT_DIR_GLOB}')

	rows_out: list[dict[str, object]] = []

	n_total = 0
	n_matched_unique = 0
	n_matched_ambiguous = 0
	n_not_found = 0

	for event_dir in event_dirs:
		evt_path, txt_path = resolve_evt_and_txt(event_dir)
		origin_iso = read_origin_jst_iso(txt_path)
		origin_ns = _iso_to_ns(origin_iso)

		n_total += 1

		cands = ns_to_rows.get(origin_ns, [])
		if not cands:
			n_not_found += 1
			rows_out.append(
				{
					'event_dir': event_dir.name,
					'evt_file': evt_path.name,
					'origin_iso': origin_iso,
					'status': 'not_found',
					'n_candidates': 0,
					'event_id_candidates': '',
				}
			)
			continue

		if len(cands) == 1:
			n_matched_unique += 1
			eid = int(epi_event_id[int(cands[0])])
			rows_out.append(
				{
					'event_dir': event_dir.name,
					'evt_file': evt_path.name,
					'origin_iso': origin_iso,
					'status': 'matched_unique',
					'n_candidates': 1,
					'event_id_candidates': str(eid),
				}
			)
			continue

		n_matched_ambiguous += 1
		eids = [int(epi_event_id[int(i)]) for i in cands]
		eids_str = ';'.join([str(x) for x in sorted(set(eids))])
		rows_out.append(
			{
				'event_dir': event_dir.name,
				'evt_file': evt_path.name,
				'origin_iso': origin_iso,
				'status': 'ambiguous',
				'n_candidates': len(cands),
				'event_id_candidates': eids_str,
			}
		)

	OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
	with OUT_CSV.open('w', newline='', encoding='utf-8') as f:
		w = csv.DictWriter(
			f,
			fieldnames=[
				'event_dir',
				'evt_file',
				'origin_iso',
				'status',
				'n_candidates',
				'event_id_candidates',
			],
		)
		w.writeheader()
		w.writerows(rows_out)

	print('[check] origin_time strict match by timestamp(ns) (no time fallback)')
	print(f'  total_events={n_total}')
	print(f'  matched_unique={n_matched_unique}')
	print(f'  matched_ambiguous={n_matched_ambiguous}')
	print(f'  not_found={n_not_found}')
	print(f'  epicenters_dup_origin_time_keys={len(dup_keys_sorted)}')
	print(f'  out_csv={OUT_CSV}')

	if n_matched_ambiguous:
		print('\n[examples] ambiguous:')
		ex = [r for r in rows_out if r['status'] == 'ambiguous']
		for r in ex[: int(min(MAX_PRINT_EXAMPLES, len(ex)))]:
			print(
				f'  {r["event_dir"]}  origin={r["origin_iso"]}  cands={r["event_id_candidates"]}'
			)

	if n_not_found:
		print('\n[examples] not_found:')
		ex = [r for r in rows_out if r['status'] == 'not_found']
		for r in ex[: int(min(MAX_PRINT_EXAMPLES, len(ex)))]:
			print(f'  {r["event_dir"]}  origin={r["origin_iso"]}')


if __name__ == '__main__':
	main()
