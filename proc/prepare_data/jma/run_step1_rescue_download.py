# %%
# proc/prepare_data/jma/run_step1_rescue_download.py
from __future__ import annotations

import csv
import os
import shutil
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from common.load_config import load_config
from common.time_util import floor_minute
from jma.download import create_hinet_client
from jma.prepare.config import JmaStep1RescueDownloadConfig
from jma.prepare.event_dirs import in_date_range, list_event_dirs, parse_date_yyyy_mm_dd
from jma.prepare.event_txt import read_origin_jst_iso
from jma.prepare.step1_paths import build_step1_paths

# =========================
# 設定（YAML から読み込む）
# =========================

YAML_PATH = Path(__file__).resolve().parent / 'config' / 'step1_rescue_download.yaml'
PRESET = 'sample'

# =========================
# 実装
# =========================


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
		mask = dd.map(lambda x: in_date_range(x, date_min=dmin, date_max=dmax))
		out = out[mask].copy()

	if min_mag is not None:
		out = out[out['magnitude'] >= float(min_mag)].copy()
	if max_mag is not None:
		out = out[out['magnitude'] <= float(max_mag)].copy()

	out = out.sort_values(['origin_ts', 'event_id'], kind='mergesort').reset_index(
		drop=True
	)
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
	return ts14[0:12]


def _event_step1_ok(event_dir: Path) -> bool:
	p = build_step1_paths(event_dir)
	return p.evt_path.is_file() and p.ch_path.is_file() and p.txt_path.is_file()


def _origin_ns_from_event_txt(txt_path: Path) -> int:
	origin_iso = read_origin_jst_iso(txt_path)
	ts = pd.to_datetime(origin_iso, format='ISO8601', errors='raise')
	return int(ts.value)


def _origin_minute_str(ts: pd.Timestamp) -> str:
	dt = floor_minute(ts.to_pydatetime())
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

		all_keys: set[str] = set()
		for r in rows:
			all_keys.update([str(k) for k in r.keys()])

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
			'minute0_from_dir',
			'origin_iso_from_dir_txt',
			'origin_ns_from_dir_txt',
			'minute0',
			'minute1',
			'status',
			'message',
			'n_dirs_downloaded',
			'n_dirs_applied',
		]

		fields: list[str] = []
		rest = set(all_keys)
		for k in preferred:
			if k in rest:
				fields.append(k)
				rest.remove(k)
		fields.extend(sorted(rest))

		w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
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
	*, win_event_dir: Path, dmin: date | None, dmax: date | None
) -> list[dict[str, object]]:
	rows: list[dict[str, object]] = []
	for d in list_event_dirs(
		win_event_dir, date_min=dmin, date_max=dmax, invalid_name='skip'
	):
		p = build_step1_paths(d)
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
				origin_iso = read_origin_jst_iso(p.txt_path)
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
	*, win_event_dir: Path, dmin: date | None, dmax: date | None
) -> dict[int, Path]:
	out: dict[int, Path] = {}
	for d in list_event_dirs(
		win_event_dir, date_min=dmin, date_max=dmax, invalid_name='skip'
	):
		p = build_step1_paths(d)
		if not p.txt_path.is_file():
			continue
		try:
			ns = int(_origin_ns_from_event_txt(p.txt_path))
		except Exception:
			continue
		if ns not in out:
			out[ns] = d
	return out


def _count_useful_downloaded_dirs(
	downloaded_dirs: list[Path],
	*,
	target_origin_ns_set: set[int],
	orphan_dirname_set: set[str],
) -> int:
	n = 0
	for d in downloaded_dirs:
		if d.name in orphan_dirname_set:
			n += 1
			continue

		txt_path = d / f'{d.name}.txt'
		if not txt_path.is_file():
			continue
		try:
			ns = int(_origin_ns_from_event_txt(txt_path))
		except Exception:
			continue
		if ns in target_origin_ns_set:
			n += 1
	return n


def main() -> None:
	cfg = load_config(JmaStep1RescueDownloadConfig, YAML_PATH, PRESET)

	if not cfg.win_event_dir.is_dir():
		raise FileNotFoundError(cfg.win_event_dir)

	dmin = parse_date_yyyy_mm_dd(cfg.date_min)
	dmax = parse_date_yyyy_mm_dd(cfg.date_max)
	if dmin is not None and dmax is not None and dmax < dmin:
		raise ValueError(f'DATE_MAX < DATE_MIN: {dmax} < {dmin}')

	if int(cfg.request_window_min) <= 0:
		raise ValueError('REQUEST_WINDOW_MIN must be >= 1')
	if int(cfg.max_retry_get_event_waveform) <= 0:
		raise ValueError('MAX_RETRY_GET_EVENT_WAVEFORM must be >= 1')

	epi = _load_epicenters_filtered(
		cfg.epi_csv, dmin=dmin, dmax=dmax, min_mag=cfg.min_mag, max_mag=cfg.max_mag
	)
	print(
		f'[epi] filtered rows={len(epi)}  date={cfg.date_min}..{cfg.date_max}  '
		f'mag={cfg.min_mag}..{cfg.max_mag}',
		flush=True,
	)

	# 既存dirの欠損（evt/txt/ch いずれか欠けている）
	orphan_rows = _scan_orphan_dirs(
		win_event_dir=cfg.win_event_dir, dmin=dmin, dmax=dmax
	)
	_write_csv(cfg.out_orphan_dirs_csv, orphan_rows)
	print(
		f'[out] orphan_dirs={len(orphan_rows)} -> {cfg.out_orphan_dirs_csv}',
		flush=True,
	)

	origin_ns_to_dir = _build_origin_ns_to_dir_map(
		win_event_dir=cfg.win_event_dir, dmin=dmin, dmax=dmax
	)
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

		if cfg.skip_if_already_ok and _event_step1_ok(event_dir):
			ok += 1
			continue

		p = build_step1_paths(event_dir)
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

	_write_csv(cfg.out_rescue_targets_csv, target_rows)
	print(
		f'[out] rescue_targets={len(target_rows)} -> {cfg.out_rescue_targets_csv}\n'
		f'  ok={ok}\n'
		f'  missing_dir_or_dir_txt={missing_dir_or_txt}\n'
		f'  missing_evt_or_ch={missing_evt_or_ch}',
		flush=True,
	)

	if not target_origin_ns_set and not orphan_dirname_set:
		print('[done] nothing to rescue', flush=True)
		return

	if not cfg.download_run:
		print('[done] download run skipped by config', flush=True)
		return

	epi_need = epi[epi['origin_ns'].isin(sorted(target_origin_ns_set))].copy()

	epi_minute_set: set[str] = set()
	if not epi_need.empty:
		epi_need['minute0'] = epi_need['origin_ts'].map(lambda x: _origin_minute_str(x))
		epi_minute_set = set(epi_need['minute0'].astype(str).tolist())

	# epi起点 + orphan(dir名)起点 を union
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
		m1 = _minute_add(m0, int(cfg.request_window_min))
		print(
			f'\n[req {i}/{len(minute_list)}] get_event_waveform {m0}..{m1}', flush=True
		)

		ok_req = False
		last_err = ''
		downloaded_dirs: list[Path] = []
		useful = 0

		for k in range(1, int(cfg.max_retry_get_event_waveform) + 1):
			print(
				f'[try] {k}/{cfg.max_retry_get_event_waveform}',
				flush=True,
			)

			# retryごとに必ず掃除（残骸で成功扱いにならないようにする）
			_clear_tmp_dir(cfg.tmp_download_dir)

			try:
				cwd0 = os.getcwd()
				os.chdir(str(cfg.tmp_download_dir))
				try:
					kwargs: dict[str, object] = {}
					if cfg.min_mag is not None:
						kwargs['minmagnitude'] = float(cfg.min_mag)
					if cfg.max_mag is not None:
						kwargs['maxmagnitude'] = float(cfg.max_mag)
					client.get_event_waveform(m0, m1, **kwargs)
				finally:
					os.chdir(cwd0)

			except Exception as e:
				last_err = repr(e)
				print(
					f'[warn] get_event_waveform failed -> retry {k}/{cfg.max_retry_get_event_waveform}: {last_err}',
					flush=True,
				)
				time.sleep(float(cfg.retry_sleep_sec) * float(k))
				continue

			downloaded_dirs = sorted(
				[p for p in cfg.tmp_download_dir.glob('D20*') if p.is_dir()]
			)
			useful = _count_useful_downloaded_dirs(
				downloaded_dirs,
				target_origin_ns_set=target_origin_ns_set,
				orphan_dirname_set=orphan_dirname_set,
			)

			print(
				f'[tmp] downloaded event dirs={len(downloaded_dirs)} useful={useful}',
				flush=True,
			)

			# 例外が無くても「欲しいものゼロ」なら retry 扱いにする
			if useful <= 0:
				last_err = 'no_useful_dirs_downloaded'
				print(
					f'[warn] no useful dirs -> retry {k}/{cfg.max_retry_get_event_waveform}',
					flush=True,
				)
				time.sleep(float(cfg.retry_sleep_sec) * float(k))
				continue

			ok_req = True
			last_err = ''
			break

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

		n_applied = 0
		for d in downloaded_dirs:
			p = build_step1_paths(d)

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

			dst_dir = cfg.win_event_dir / d.name
			dst_dir.mkdir(parents=True, exist_ok=True)

			dst_paths = build_step1_paths(dst_dir)

			if p.evt_path.is_file():
				_safe_copy(p.evt_path, dst_paths.evt_path)
			if p.ch_path.is_file():
				_safe_copy(p.ch_path, dst_paths.ch_path)
			if p.txt_path.is_file():
				_safe_copy(p.txt_path, dst_paths.txt_path)

			n_applied += 1

		run_rows.append(
			{
				'minute0': m0,
				'minute1': m1,
				'status': 'requested',
				'message': '',
				'n_dirs_downloaded': len(downloaded_dirs),
				'n_dirs_applied': n_applied,
				'useful': useful,
			}
		)

	origin_ns_to_dir2 = _build_origin_ns_to_dir_map(
		win_event_dir=cfg.win_event_dir, dmin=dmin, dmax=dmax
	)

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

	_write_csv(cfg.out_rescue_run_csv, run_rows)
	print(
		f'\n[out] rescue_run rows={len(run_rows)} -> {cfg.out_rescue_run_csv}',
		flush=True,
	)
	print(
		f'[check] after rescue (origin_ns targets): ok={n_ok_after} still_missing={n_still_missing}',
		flush=True,
	)
	print('[done]', flush=True)


if __name__ == '__main__':
	main()

# %%
