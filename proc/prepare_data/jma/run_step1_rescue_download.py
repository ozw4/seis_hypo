# %%
# proc/prepare_data/jma/run_step1_rescue_download.py
from __future__ import annotations

import csv
import os
import shutil
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from jma.download import create_hinet_client
from jma.picks import read_origin_iso_from_txt

# =========================
# 設定（直書き）
# =========================

WIN_EVENT_DIR = Path('/workspace/data/waveform/jma/event').resolve()

EPI_CSV = Path(
	'/workspace/data/arrivetime/JMA/arrivetime_epicenters_2023.0.csv'
).resolve()

DATE_MIN: str | None = '2023-01-01'
DATE_MAX: str | None = '2023-01-31'

MIN_MAG: float | None = 1.00
MAX_MAG: float | None = 9.9

REQUEST_WINDOW_MIN = 1

MAX_RETRY_GET_EVENT_WAVEFORM = 5
RETRY_SLEEP_SEC = 2.0

TMP_DOWNLOAD_DIR = Path(
	'/workspace/proc/prepare_data/jma/_tmp/step1_rescue_download'
).resolve()

OUT_RESCUE_TARGETS_CSV = Path(
	'/workspace/proc/prepare_data/jma/_tmp/step1_rescue_targets.csv'
).resolve()
OUT_ORPHAN_DIRS_CSV = Path(
	'/workspace/proc/prepare_data/jma/_tmp/step1_orphan_event_dirs.csv'
).resolve()
OUT_RESCUE_RUN_CSV = Path(
	'/workspace/proc/prepare_data/jma/_tmp/step1_rescue_run.csv'
).resolve()

SKIP_IF_ALREADY_OK = True


# =========================
# 実装
# =========================


def _parse_date_yyyy_mm_dd(s: str | None) -> date | None:
	if s is None:
		return None
	ss = str(s).strip()
	if not ss:
		return None
	y, m, d = ss.split('-')
	return date(int(y), int(m), int(d))


def _in_date_range(d: date, *, dmin: date | None, dmax: date | None) -> bool:
	if dmin is not None and d < dmin:
		return False
	if dmax is not None and d > dmax:
		return False
	return True


def _pick_col(cols: list[str], candidates: list[str]) -> str:
	lower = {c.lower(): c for c in cols}
	for k in candidates:
		if k.lower() in lower:
			return lower[k.lower()]
	raise ValueError(f'missing required column. candidates={candidates} cols={cols}')


def _pick_col_optional(cols: list[str], candidates: list[str]) -> str | None:
	lower = {c.lower(): c for c in cols}
	for k in candidates:
		if k.lower() in lower:
			return lower[k.lower()]
	return None


def _load_epicenters_filtered(
	epi_csv: Path,
	*,
	dmin: date | None,
	dmax: date | None,
	min_mag: float | None,
	max_mag: float | None,
) -> pd.DataFrame:
	if not epi_csv.is_file():
		raise FileNotFoundError(epi_csv)

	df = pd.read_csv(epi_csv, low_memory=False)
	if df.empty:
		raise ValueError(f'empty epicenters csv: {epi_csv}')

	event_id_col = _pick_col(df.columns.tolist(), ['event_id'])
	origin_col = _pick_col(
		df.columns.tolist(), ['origin_time', 'origin_iso', 'origin_time_jst']
	)
	lat_col = _pick_col(df.columns.tolist(), ['latitude_deg'])
	lon_col = _pick_col(df.columns.tolist(), ['longitude_deg'])
	mag_col = _pick_col(df.columns.tolist(), ['mag1'])

	# record_type は無い/型が様々なので optional（文字列で保持）
	rec_col = _pick_col_optional(df.columns.tolist(), ['record_type'])

	origin_ts = pd.to_datetime(df[origin_col], format='ISO8601', errors='raise')

	out = pd.DataFrame(
		{
			'event_id': df[event_id_col].astype('int64'),
			'origin_time': df[origin_col].astype(str),
			'origin_ts': origin_ts,
			'origin_ns': origin_ts.astype('int64'),
			'lat': df[lat_col].astype('float64'),
			'lon': df[lon_col].astype('float64'),
			'magnitude': df[mag_col].astype('float64'),
			'record_type': (df[rec_col].astype(str) if rec_col is not None else ''),
		}
	)

	if dmin is not None or dmax is not None:
		dd = out['origin_ts'].dt.date
		mask = dd.map(lambda x: _in_date_range(x, dmin=dmin, dmax=dmax))
		out = out[mask].copy()

	if min_mag is not None:
		out = out[out['magnitude'] >= float(min_mag)].copy()
	if max_mag is not None:
		out = out[out['magnitude'] <= float(max_mag)].copy()

	out = out.sort_values(['origin_ts', 'event_id'], kind='mergesort').reset_index(
		drop=True
	)
	return out


def _list_event_dirs_in_range(*, dmin: date | None, dmax: date | None) -> list[Path]:
	out: list[Path] = []
	for p in sorted([x for x in WIN_EVENT_DIR.glob('D20*') if x.is_dir()]):
		name = p.name
		if len(name) < 9 or not name.startswith('D'):
			continue
		ymd = name[1:9]
		if not ymd.isdigit():
			continue
		dd = date(int(ymd[0:4]), int(ymd[4:6]), int(ymd[6:8]))
		if dmin is not None or dmax is not None:
			if not _in_date_range(dd, dmin=dmin, dmax=dmax):
				continue
		out.append(p)
	return out


def _minute0_from_event_dir_name(dir_name: str) -> str | None:
	# expected like: DYYYYMMDDHHMMSS_XX (at least D + 14 digits)
	name = str(dir_name).strip()
	if len(name) < 1 + 14:
		return None
	if not name.startswith('D'):
		return None
	ts14 = name[1:15]
	if not ts14.isdigit():
		return None
	# minute: YYYYMMDDHHMM
	return ts14[0:12]


@dataclass(frozen=True)
class Step1Files:
	stem: str
	evt_path: Path
	ch_path: Path
	txt_path: Path
	active_ch_path: Path


def _step1_paths(event_dir: Path) -> Step1Files:
	stem = event_dir.name
	return Step1Files(
		stem=stem,
		evt_path=event_dir / f'{stem}.evt',
		ch_path=event_dir / f'{stem}.ch',
		txt_path=event_dir / f'{stem}.txt',  # ★ここだけを見る
		active_ch_path=event_dir / f'{stem}_active.ch',
	)


def _event_step1_ok(event_dir: Path) -> bool:
	p = _step1_paths(event_dir)
	return p.evt_path.is_file() and p.ch_path.is_file() and p.txt_path.is_file()


def _origin_ns_from_event_txt(txt_path: Path) -> int:
	origin_iso = read_origin_iso_from_txt(txt_path)
	ts = pd.to_datetime(origin_iso, format='ISO8601', errors='raise')
	return int(ts.value)


def _origin_minute_str(ts: pd.Timestamp) -> str:
	dt = ts.to_pydatetime().replace(second=0, microsecond=0)
	return dt.strftime('%Y%m%d%H%M')


def _minute_add(minute_yyyymmddhhmm: str, add_min: int) -> str:
	dt = datetime.strptime(minute_yyyymmddhhmm, '%Y%m%d%H%M')
	dt2 = dt + timedelta(minutes=int(add_min))
	return dt2.strftime('%Y%m%d%H%M')


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open('w', newline='', encoding='utf-8') as f:
		if not rows:
			w = csv.writer(f)
			w.writerow(['(empty)'])
			return

		# 全行のキーを集めて列を決める（行ごとに列が違っても落とさない）
		all_keys: set[str] = set()
		for r in rows:
			all_keys.update([str(k) for k in r.keys()])

		# 主要列は見やすい順に先頭へ（存在するものだけ）
		preferred = [
			'event_id',
			'record_type',
			'origin_time_epi',
			'origin_ns',
			'lat',
			'lon',
			'magnitude',
			'action',
			'event_dir',
			'missing',
		]

		fields: list[str] = []
		rest = set(all_keys)
		for k in preferred:
			if k in rest:
				fields.append(k)
				rest.remove(k)

		# その他の列は末尾に安定ソートで追加
		fields.extend(sorted(rest))

		w = csv.DictWriter(f, fieldnames=fields)
		w.writeheader()
		for r in rows:
			w.writerow(r)


def _clear_tmp_dir(tmp_dir: Path) -> None:
	tmp_dir.mkdir(parents=True, exist_ok=True)
	for p in list(tmp_dir.iterdir()):
		if p.is_dir():
			shutil.rmtree(p)
		else:
			p.unlink()


def _safe_copy(src: Path, dst: Path) -> None:
	dst.parent.mkdir(parents=True, exist_ok=True)
	if dst.exists():
		return
	shutil.copy2(src, dst)


def _scan_orphan_dirs(
	*, dmin: date | None, dmax: date | None
) -> list[dict[str, object]]:
	rows: list[dict[str, object]] = []
	for d in _list_event_dirs_in_range(dmin=dmin, dmax=dmax):
		p = _step1_paths(d)
		missing = []
		if not p.txt_path.is_file():
			missing.append('txt')
		if not p.evt_path.is_file():
			missing.append('evt')
		if not p.ch_path.is_file():
			missing.append('ch')

		if not missing:
			continue

		minute0_from_dir = _minute0_from_event_dir_name(d.name) or ''

		origin_ns = ''
		origin_iso = ''
		if p.txt_path.is_file():
			try:
				origin_iso = read_origin_iso_from_txt(p.txt_path)
				origin_ns = int(_origin_ns_from_event_txt(p.txt_path))
			except Exception:
				origin_iso = ''
				origin_ns = ''

		rows.append(
			{
				'event_dir': str(d),
				'missing': ','.join(missing),
				'minute0_from_dir': minute0_from_dir,
				'origin_iso_from_dir_txt': origin_iso,
				'origin_ns_from_dir_txt': origin_ns,
			}
		)
	return rows


def _build_origin_ns_to_dir_map(
	*, dmin: date | None, dmax: date | None
) -> dict[int, Path]:
	out: dict[int, Path] = {}
	for d in _list_event_dirs_in_range(dmin=dmin, dmax=dmax):
		p = _step1_paths(d)
		if not p.txt_path.is_file():
			continue
		try:
			ns = int(_origin_ns_from_event_txt(p.txt_path))
		except Exception:
			continue
		if ns not in out:
			out[ns] = d
	return out


def main() -> None:
	if not WIN_EVENT_DIR.is_dir():
		raise FileNotFoundError(WIN_EVENT_DIR)

	dmin = _parse_date_yyyy_mm_dd(DATE_MIN)
	dmax = _parse_date_yyyy_mm_dd(DATE_MAX)
	if dmin is not None and dmax is not None and dmax < dmin:
		raise ValueError(f'DATE_MAX < DATE_MIN: {dmax} < {dmin}')

	if int(REQUEST_WINDOW_MIN) <= 0:
		raise ValueError('REQUEST_WINDOW_MIN must be >= 1')
	if int(MAX_RETRY_GET_EVENT_WAVEFORM) <= 0:
		raise ValueError('MAX_RETRY_GET_EVENT_WAVEFORM must be >= 1')

	epi = _load_epicenters_filtered(
		EPI_CSV, dmin=dmin, dmax=dmax, min_mag=MIN_MAG, max_mag=MAX_MAG
	)
	print(
		f'[epi] filtered rows={len(epi)}  date={DATE_MIN}..{DATE_MAX}  mag={MIN_MAG}..{MAX_MAG}',
		flush=True,
	)

	# 既存dirの欠損（evt/txt/ch いずれか欠けている）
	orphan_rows = _scan_orphan_dirs(dmin=dmin, dmax=dmax)
	_write_csv(OUT_ORPHAN_DIRS_CSV, orphan_rows)
	print(f'[out] orphan_dirs={len(orphan_rows)} -> {OUT_ORPHAN_DIRS_CSV}', flush=True)

	origin_ns_to_dir = _build_origin_ns_to_dir_map(dmin=dmin, dmax=dmax)
	orphan_dirname_set = set(Path(str(r['event_dir'])).name for r in orphan_rows)

	# orphan_dirs 由来 minute（dir名から）
	orphan_minute_set: set[str] = set()
	for r in orphan_rows:
		m0 = str(r.get('minute0_from_dir', '')).strip()
		if m0 and len(m0) == 12 and m0.isdigit():
			orphan_minute_set.add(m0)

	target_rows: list[dict[str, object]] = []
	target_origin_ns_set: set[int] = set()

	ok = 0
	missing_dir_or_txt = 0
	missing_evt_or_ch = 0

	for _, row in epi.iterrows():
		ns = int(row['origin_ns'])
		event_id = int(row['event_id'])
		origin_time = str(row['origin_time'])
		record_type = str(row.get('record_type', ''))

		event_dir = origin_ns_to_dir.get(ns)

		if event_dir is None:
			target_rows.append(
				{
					'event_id': event_id,
					'record_type': record_type,
					'origin_time_epi': origin_time,
					'origin_ns': ns,
					'lat': float(row['lat']),
					'lon': float(row['lon']),
					'magnitude': float(row['magnitude']),
					'action': 'missing_dir_or_dir_txt',
					'event_dir': '',
				}
			)
			target_origin_ns_set.add(ns)
			missing_dir_or_txt += 1
			continue

		if SKIP_IF_ALREADY_OK and _event_step1_ok(event_dir):
			ok += 1
			continue

		p = _step1_paths(event_dir)
		missing = []
		if not p.evt_path.is_file():
			missing.append('evt')
		if not p.ch_path.is_file():
			missing.append('ch')

		if missing:
			target_rows.append(
				{
					'event_id': event_id,
					'record_type': record_type,
					'origin_time_epi': origin_time,
					'origin_ns': ns,
					'lat': float(row['lat']),
					'lon': float(row['lon']),
					'magnitude': float(row['magnitude']),
					'action': 'missing_evt_or_ch',
					'event_dir': str(event_dir),
					'missing': ','.join(missing),
				}
			)
			target_origin_ns_set.add(ns)
			missing_evt_or_ch += 1
			continue

		ok += 1

	_write_csv(OUT_RESCUE_TARGETS_CSV, target_rows)
	print(
		f'[out] rescue_targets={len(target_rows)} -> {OUT_RESCUE_TARGETS_CSV}\n'
		f'  ok={ok}\n'
		f'  missing_dir_or_dir_txt={missing_dir_or_txt}\n'
		f'  missing_evt_or_ch={missing_evt_or_ch}',
		flush=True,
	)

	if not target_origin_ns_set and not orphan_dirname_set:
		print('[done] nothing to rescue', flush=True)
		return

	epi_need = epi[epi['origin_ns'].isin(sorted(target_origin_ns_set))].copy()

	epi_minute_set: set[str] = set()
	if not epi_need.empty:
		epi_need['minute0'] = epi_need['origin_ts'].map(lambda x: _origin_minute_str(x))
		epi_minute_set = set(epi_need['minute0'].astype(str).tolist())

	# ★ここが重要：epi起点 + orphan(dir名)起点 を union
	minute_list = sorted(set(epi_minute_set) | set(orphan_minute_set))

	print(
		f'[plan] rescue minutes={len(minute_list)}  '
		f'(from_epi={len(epi_minute_set)} from_orphan_dirname={len(orphan_minute_set)})  '
		f'events_need={len(epi_need)} orphan_dirs={len(orphan_dirname_set)}',
		flush=True,
	)

	if not minute_list:
		print('[done] no minutes to request', flush=True)
		return

	client = create_hinet_client()
	run_rows: list[dict[str, object]] = []

	for i, m0 in enumerate(minute_list, 1):
		m1 = _minute_add(m0, int(REQUEST_WINDOW_MIN))
		print(
			f'\n[req {i}/{len(minute_list)}] get_event_waveform {m0}..{m1}', flush=True
		)

		_clear_tmp_dir(TMP_DOWNLOAD_DIR)

		ok_req = False
		last_err = ''

		for k in range(1, int(MAX_RETRY_GET_EVENT_WAVEFORM) + 1):
			try:
				cwd0 = os.getcwd()
				os.chdir(str(TMP_DOWNLOAD_DIR))
				try:
					kwargs: dict[str, object] = {}
					if MIN_MAG is not None:
						kwargs['minmagnitude'] = float(MIN_MAG)
					if MAX_MAG is not None:
						kwargs['maxmagnitude'] = float(MAX_MAG)
					client.get_event_waveform(m0, m1, **kwargs)
				finally:
					os.chdir(cwd0)
				ok_req = True
				last_err = ''
				break
			except Exception as e:
				last_err = repr(e)
				print(
					f'[warn] get_event_waveform failed -> retry {k}/{MAX_RETRY_GET_EVENT_WAVEFORM}: {last_err}',
					flush=True,
				)
				time.sleep(float(RETRY_SLEEP_SEC) * float(k))

		if not ok_req:
			run_rows.append(
				{
					'minute0': m0,
					'minute1': m1,
					'status': 'request_failed',
					'message': last_err,
					'n_dirs_downloaded': 0,
					'n_dirs_applied': 0,
				}
			)
			continue

		downloaded_dirs = sorted(
			[p for p in TMP_DOWNLOAD_DIR.glob('D20*') if p.is_dir()]
		)
		print(f'[tmp] downloaded event dirs={len(downloaded_dirs)}', flush=True)

		n_applied = 0
		for d in downloaded_dirs:
			p = _step1_paths(d)

			want = False
			origin_ns = None

			if d.name in orphan_dirname_set:
				want = True

			if p.txt_path.is_file():
				try:
					origin_ns = int(_origin_ns_from_event_txt(p.txt_path))
				except Exception:
					origin_ns = None
				if origin_ns is not None and origin_ns in target_origin_ns_set:
					want = True

			if not want:
				continue

			dst_dir = WIN_EVENT_DIR / d.name
			dst_dir.mkdir(parents=True, exist_ok=True)

			src_evt = d / f'{d.name}.evt'
			src_ch = d / f'{d.name}.ch'
			src_txt = d / f'{d.name}.txt'

			dst_evt = dst_dir / f'{d.name}.evt'
			dst_ch = dst_dir / f'{d.name}.ch'
			dst_txt = dst_dir / f'{d.name}.txt'

			if src_evt.is_file():
				_safe_copy(src_evt, dst_evt)
			if src_ch.is_file():
				_safe_copy(src_ch, dst_ch)
			if src_txt.is_file():
				_safe_copy(src_txt, dst_txt)

			n_applied += 1

		run_rows.append(
			{
				'minute0': m0,
				'minute1': m1,
				'status': 'requested',
				'message': '',
				'n_dirs_downloaded': len(downloaded_dirs),
				'n_dirs_applied': n_applied,
			}
		)

	origin_ns_to_dir2 = _build_origin_ns_to_dir_map(dmin=dmin, dmax=dmax)

	n_ok_after = 0
	n_still_missing = 0
	for ns in sorted(target_origin_ns_set):
		d = origin_ns_to_dir2.get(int(ns))
		if d is None:
			n_still_missing += 1
			continue
		if _event_step1_ok(d):
			n_ok_after += 1
		else:
			n_still_missing += 1

	_write_csv(OUT_RESCUE_RUN_CSV, run_rows)
	print(
		f'\n[out] rescue_run rows={len(run_rows)} -> {OUT_RESCUE_RUN_CSV}', flush=True
	)
	print(
		f'[check] after rescue (origin_ns targets): ok={n_ok_after} still_missing={n_still_missing}',
		flush=True,
	)
	print('[done]', flush=True)


if __name__ == '__main__':
	main()

# %%
