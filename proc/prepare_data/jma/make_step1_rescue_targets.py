# %%
# proc/prepare_data/jma/make_step1_rescue_targets.py
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

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

MIN_MAG: float | None = 1.0
MAX_MAG: float | None = 9.9

OUT_CSV = Path(
	'/workspace/proc/prepare_data/jma/_tmp/step1_rescue_targets.csv'
).resolve()
OUT_ORPHAN_DIRS_CSV = Path(
	'/workspace/proc/prepare_data/jma/_tmp/step1_orphan_event_dirs.csv'
).resolve()

# 対象dirを絞りたい場合（空なら全部）
TARGET_EVENT_DIR_NAMES: list[str] = []
# 例:
# TARGET_EVENT_DIR_NAMES = ["D20230118000041_20"]

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
		}
	)

	if dmin is not None or dmax is not None:
		d = out['origin_ts'].dt.date
		mask = d.map(lambda x: _in_date_range(x, dmin=dmin, dmax=dmax))
		out = out[mask].copy()

	if min_mag is not None:
		out = out[out['magnitude'] >= float(min_mag)].copy()
	if max_mag is not None:
		out = out[out['magnitude'] <= float(max_mag)].copy()

	out = out.sort_values(['origin_ts', 'event_id'], kind='mergesort').reset_index(
		drop=True
	)
	return out


def _list_event_dirs() -> list[Path]:
	if TARGET_EVENT_DIR_NAMES:
		out: list[Path] = []
		for name in TARGET_EVENT_DIR_NAMES:
			p = (WIN_EVENT_DIR / name).resolve()
			if not p.is_dir():
				raise FileNotFoundError(f'event_dir not found: {p}')
			out.append(p)
		return out
	return sorted([p for p in WIN_EVENT_DIR.glob('D20*') if p.is_dir()])


def _evt_count(event_dir: Path) -> int:
	return len(list(event_dir.glob('*.evt')))


def _event_txt_path(event_dir: Path) -> Path:
	return event_dir / f'{event_dir.name}.txt'


def _origin_ns_from_txt(txt_path: Path) -> int:
	origin_iso = read_origin_iso_from_txt(txt_path)
	ts = pd.to_datetime(origin_iso, format='ISO8601', errors='raise')
	return int(ts.value)


@dataclass(frozen=True)
class DirIndexRow:
	event_dir: str
	has_event_txt: bool
	evt_count: int
	origin_ns: int | None
	origin_iso: str | None


def _scan_event_dirs(
	event_dirs: list[Path],
) -> tuple[dict[int, Path], list[DirIndexRow]]:
	origin_to_dirs: dict[int, list[Path]] = {}
	rows: list[DirIndexRow] = []

	for d in event_dirs:
		evtc = _evt_count(d)

		evt_txt = _event_txt_path(d)
		if evt_txt.is_file():
			ns = _origin_ns_from_txt(evt_txt)
			origin_to_dirs.setdefault(ns, []).append(d)
			rows.append(
				DirIndexRow(
					event_dir=str(d),
					has_event_txt=True,
					evt_count=evtc,
					origin_ns=ns,
					origin_iso=read_origin_iso_from_txt(evt_txt),
				)
			)
		else:
			rows.append(
				DirIndexRow(
					event_dir=str(d),
					has_event_txt=False,
					evt_count=evtc,
					origin_ns=None,
					origin_iso=None,
				)
			)

	origin_ns_to_unique_dir: dict[int, Path] = {}
	for ns, dirs in origin_to_dirs.items():
		if len(dirs) == 1:
			origin_ns_to_unique_dir[int(ns)] = dirs[0]

	return origin_ns_to_unique_dir, rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open('w', newline='', encoding='utf-8') as f:
		if not rows:
			w = csv.writer(f)
			w.writerow(['(empty)'])
			return
		fields = list(rows[0].keys())
		w = csv.DictWriter(f, fieldnames=fields)
		w.writeheader()
		for r in rows:
			w.writerow(r)


def main() -> None:
	if not WIN_EVENT_DIR.is_dir():
		raise FileNotFoundError(WIN_EVENT_DIR)

	dmin = _parse_date_yyyy_mm_dd(DATE_MIN)
	dmax = _parse_date_yyyy_mm_dd(DATE_MAX)
	if dmin is not None and dmax is not None and dmax < dmin:
		raise ValueError(f'DATE_MAX < DATE_MIN: {dmax} < {dmin}')

	epi = _load_epicenters_filtered(
		EPI_CSV, dmin=dmin, dmax=dmax, min_mag=MIN_MAG, max_mag=MAX_MAG
	)
	print(
		f'[epi] filtered rows={len(epi)}  '
		f'date={DATE_MIN}..{DATE_MAX}  mag={MIN_MAG}..{MAX_MAG}',
		flush=True,
	)

	event_dirs = _list_event_dirs()
	print(f'[dirs] found={len(event_dirs)} under {WIN_EVENT_DIR}', flush=True)

	origin_ns_to_dir, dir_rows = _scan_event_dirs(event_dirs)

	# 孤児dirの一覧（event_dir名.txt が無い、または .evt が無い）
	orphan_rows: list[dict[str, object]] = []
	for r in dir_rows:
		if (not r.has_event_txt) or (r.evt_count == 0):
			orphan_rows.append(
				{
					'event_dir': r.event_dir,
					'has_event_txt': int(bool(r.has_event_txt)),
					'evt_count': int(r.evt_count),
					'origin_ns': '' if r.origin_ns is None else int(r.origin_ns),
					'origin_iso': '' if r.origin_iso is None else str(r.origin_iso),
				}
			)
	_write_csv(OUT_ORPHAN_DIRS_CSV, orphan_rows)
	print(f'[out] orphan_dirs={len(orphan_rows)} -> {OUT_ORPHAN_DIRS_CSV}', flush=True)

	# 期待イベントに対して「dir無し or (dirはあるが event_dir名.txt が無い)」または「.evt が無い」だけ抽出
	target_rows: list[dict[str, object]] = []
	ok = 0
	miss_dir_or_txt = 0
	miss_evt = 0

	for _, row in epi.iterrows():
		ns = int(row['origin_ns'])
		event_id = int(row['event_id'])

		event_dir = origin_ns_to_dir.get(ns)
		if event_dir is None:
			target_rows.append(
				{
					'event_id': event_id,
					'origin_time': str(row['origin_time']),
					'origin_ns': ns,
					'lat': float(row['lat']),
					'lon': float(row['lon']),
					'magnitude': float(row['magnitude']),
					'action': 'missing_dir_or_txt',
					'event_dir': '',
				}
			)
			miss_dir_or_txt += 1
			continue

		evtc = _evt_count(event_dir)
		if evtc == 0:
			target_rows.append(
				{
					'event_id': event_id,
					'origin_time': str(row['origin_time']),
					'origin_ns': ns,
					'lat': float(row['lat']),
					'lon': float(row['lon']),
					'magnitude': float(row['magnitude']),
					'action': 'missing_evt',
					'event_dir': str(event_dir),
				}
			)
			miss_evt += 1
			continue

		ok += 1

	_write_csv(OUT_CSV, target_rows)
	print(
		f'[out] rescue_targets={len(target_rows)} -> {OUT_CSV}\n'
		f'  ok={ok}\n'
		f'  missing_dir_or_txt={miss_dir_or_txt}\n'
		f'  missing_evt={miss_evt}',
		flush=True,
	)


if __name__ == '__main__':
	main()

# %%
