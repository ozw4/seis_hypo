# src/jma/missing_continuous.py の run_make_missing_continuous に期間フィルタを追加
from __future__ import annotations

from pathlib import Path

import pandas as pd

from jma.prepare.event_dirs import (
	in_date_range,
	list_event_dirs,
	parse_date_yyyy_mm_dd,
)
from jma.prepare.event_txt import read_origin_jst_iso
from jma.picks import build_pick_table_for_event
from jma.station_reader import read_hinet_channel_table
from jma.stationcode_common import normalize_code, pick_one_network_code
from jma.stationcode_mappingdb import load_mapping_db
from jma.stationcode_presence import load_presence_db

P_PHASES = {'P', 'EP', 'IP'}
S_PHASES = {'S', 'ES', 'IS'}


def find_event_id_by_origin(epi_df: pd.DataFrame, origin_iso: str) -> int:
	hit = epi_df.loc[epi_df['origin_time'] == origin_iso, 'event_id']
	if len(hit) == 1:
		return int(hit.iloc[0])
	if len(hit) > 1:
		raise ValueError(f'multiple event_id matched for origin_time={origin_iso}')

	epi_t = pd.to_datetime(epi_df['origin_time'], format='ISO8601')
	t0 = pd.to_datetime(origin_iso, format='ISO8601')
	dt_s = (epi_t - t0).abs().dt.total_seconds()
	i = int(dt_s.idxmin())
	if float(dt_s.iloc[i]) > 0.5:
		raise ValueError(
			f'no close event_id for origin_time={origin_iso} (closest={float(dt_s.iloc[i]):.3f}s)'
		)
	return int(epi_df.loc[i, 'event_id'])


def run_make_missing_continuous(
	*,
	win_event_dir: str | Path,
	meas_csv: str | Path,
	epi_csv: str | Path,
	pres_csv: str | Path,
	mapping_report_csv: str | Path,
	near0_suggest_csv: str | Path,
	out_missing_csv: str | Path,
	skip_if_no_active_ch: bool = True,
	# 追加：期間フィルタ（YYYY-MM-DD）。両方Noneなら全期間
	date_min: str | None = None,
	date_max: str | None = None,
	# 追加：イベント単位skip
	skip_if_done: bool = True,
	run_tag: str = 'v1',
) -> pd.DataFrame:
	import json
	from datetime import datetime
	from pathlib import Path

	win_event_dir = Path(win_event_dir)
	meas_csv = Path(meas_csv)
	epi_csv = Path(epi_csv)
	out_missing_csv = Path(out_missing_csv)

	for p in [
		win_event_dir,
		meas_csv,
		epi_csv,
		Path(pres_csv),
		Path(mapping_report_csv),
	]:
		if not p.exists():
			raise FileNotFoundError(p)

	dmin = parse_date_yyyy_mm_dd(date_min, allow_slash=True, allow_time=True)
	dmax = parse_date_yyyy_mm_dd(date_max, allow_slash=True, allow_time=True)
	if dmin is not None and dmax is not None and dmax < dmin:
		raise ValueError(f'date_max < date_min: {dmax} < {dmin}')

	meas_df = pd.read_csv(meas_csv, low_memory=False)
	epi_df = pd.read_csv(epi_csv, low_memory=False)

	pdb = load_presence_db(pres_csv)
	mdb = load_mapping_db(mapping_report_csv, near0_suggest_csv)

	missing_rows: list[dict[str, object]] = []

	run_tag2 = str(run_tag).strip()
	if not run_tag2:
		raise ValueError('run_tag must be non-empty')

	for event_dir in list_event_dirs(
		win_event_dir, date_min=dmin, date_max=dmax, invalid_name='keep'
	):
		try:
			for evt_path in sorted(event_dir.glob('*.evt')):
				txt_path = evt_path.with_suffix('.txt')
				active_ch_path = evt_path.with_name(f'{evt_path.stem}_active.ch')

				if not txt_path.is_file():
					print(f'[warn] skip (no .txt): {evt_path}')
					continue

				if not active_ch_path.is_file():
					if skip_if_no_active_ch:
						print(f'[warn] skip (no _active.ch): {evt_path}')
						continue
					raise FileNotFoundError(active_ch_path)

			pick_by_ch, map_log = build_pick_table_for_event(
				meas_df,
				event_id=event_id,
				event_month=event_month,
				mdb=mdb,
				pdb=pdb,
				p_phases=P_PHASES,
				s_phases=S_PHASES,
			)

				origin_iso = read_origin_jst_iso(txt_path)
				origin_ts = pd.to_datetime(origin_iso, format='ISO8601')

				# 厳密フィルタ（ORIGIN_JST）
				if dmin is not None or dmax is not None:
					if not in_date_range(origin_ts, date_min=dmin, date_max=dmax):
						continue

				event_month = origin_iso[:7]
				event_id = find_event_id_by_origin(epi_df, origin_iso)

				pick_by_ch, map_log = build_pick_table_for_event(
					meas_df,
					event_id=event_id,
					event_month=event_month,
					mdb=mdb,
					pdb=pdb,
				)

				station_df = read_hinet_channel_table(active_ch_path)
				st_keys = set(
					station_df['station'].astype(str).map(normalize_code).tolist()
				)

				pick_idx = pick_by_ch.copy()
				pick_idx.index = pick_idx.index.astype(str).map(normalize_code)
				missing_keys = sorted(set(pick_idx.index.tolist()) - st_keys)

				out_txt = event_dir / f'{evt_path.stem}_missing_continuous.txt'
				n_missing = len(missing_keys)

				if missing_keys:
					lines: list[str] = []
					for k in missing_keys:
						r = pick_idx.loc[k]
						net = pick_one_network_code(r.get('preferred_network_code'))
						lines.append(f'{k}\t{net}')
						missing_rows.append(
							{
								'event_dir': str(event_dir),
								'evt_file': evt_path.name,
								'event_id': int(event_id),
								'origin_time': origin_iso,
								'event_month': event_month,
								'ch_station': k,
								'network_code': net,
								'p_time': r.get('p_time'),
								's_time': r.get('s_time'),
							}
						)
					out_txt.write_text('\n'.join(lines) + '\n', encoding='utf-8')
					print(
						f'[missing] {evt_path.name}: n_missing={n_missing} -> {out_txt.name}'
					)
				else:
					if out_txt.is_file():
						out_txt.unlink()
					print(f'[missing] {evt_path.name}: n_missing=0')

				if map_log:
					pd.DataFrame(map_log).to_csv(
						event_dir / f'{evt_path.stem}_mapping_log.csv',
						index=False,
					)

				# doneマーカー（missing=0でも必ず作る）
				done_obj = {
					'evt_file': evt_path.name,
					'origin_time': origin_iso,
					'event_id': int(event_id),
					'event_month': event_month,
					'n_missing': n_missing,
					'run_tag': run_tag2,
					'created_at_utc': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
				}
				done_path.write_text(
					json.dumps(done_obj, ensure_ascii=False, indent=2) + '\n',
					encoding='utf-8',
				)
		except Exception as e:
			print(f'Error in make_missing_continuous: {e}')
			continue

	out_df = pd.DataFrame(missing_rows)
	out_missing_csv.parent.mkdir(parents=True, exist_ok=True)
	out_df.to_csv(out_missing_csv, index=False)
	print(f'[saved] missing log -> {out_missing_csv}  rows={len(out_df)}')
	return out_df
