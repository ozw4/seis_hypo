# %%
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import pandas as pd

# ==========================
# 設定（ここだけ直に編集）
# ==========================
ROOT = Path('snapshots/yearly')
OUT_TRANSITIONS = ROOT / 'yearly_transitions.csv'
OUT_CANDIDATES = ROOT / 'refine_candidates.csv'

# stations CSV に必須の列名（export_station_summary_from_channels の出力想定）
COL_NETWORK = 'network_code'
COL_STATION = 'station'

# 年ごとに複数スナップショットがある場合、最も早い日時のものをその年の代表として使う
# （例: 2025-01-01 と 2025-12-01 が両方あれば 2025-01-01 が採用される）


_RE_STATIONS_STEM = re.compile(r'^stations_(?P<ts>\d{4}|\d{8}|\d{12})$')
_RE_DIR_YEAR = re.compile(r'^\d{4}$')
_RE_DIR_TS12 = re.compile(r'^\d{12}$')


def _parse_ts_token(ts: str) -> dt.datetime:
	if len(ts) == 4:
		y = int(ts)
		return dt.datetime(y, 1, 1, 0, 0)
	if len(ts) == 8:
		y = int(ts[0:4])
		m = int(ts[4:6])
		d = int(ts[6:8])
		return dt.datetime(y, m, d, 0, 0)
	if len(ts) == 12:
		y = int(ts[0:4])
		m = int(ts[4:6])
		d = int(ts[6:8])
		hh = int(ts[8:10])
		mm = int(ts[10:12])
		return dt.datetime(y, m, d, hh, mm)
	raise ValueError(f'unsupported timestamp token: {ts}')


def _infer_snapshot_datetime(p: Path) -> dt.datetime:
	m = _RE_STATIONS_STEM.match(p.stem)
	if m:
		return _parse_ts_token(m.group('ts'))

	parts = list(p.parts)
	ts12: str | None = None
	year4: str | None = None

	for s in parts:
		if year4 is None and _RE_DIR_YEAR.match(s):
			year4 = s
		if _RE_DIR_TS12.match(s):
			ts12 = s

	if ts12 is not None:
		return _parse_ts_token(ts12)
	if year4 is not None:
		return _parse_ts_token(year4)

	raise ValueError(f'cannot infer snapshot datetime from path: {p}')


def _infer_year(p: Path, when: dt.datetime) -> int:
	parts = list(p.parts)
	for s in parts:
		if _RE_DIR_YEAR.match(s):
			return int(s)
	return int(when.year)


def _load_station_sets(csv_path: Path) -> dict[str, set[str]]:
	if not csv_path.is_file():
		raise FileNotFoundError(f'missing stations csv: {csv_path}')

	df = pd.read_csv(csv_path, dtype=str)
	if COL_NETWORK not in df.columns or COL_STATION not in df.columns:
		raise ValueError(
			f'stations csv missing required columns at {csv_path}: '
			f'need={[COL_NETWORK, COL_STATION]} got={list(df.columns)}'
		)

	df = df[[COL_NETWORK, COL_STATION]].copy()
	df[COL_NETWORK] = df[COL_NETWORK].astype(str).str.strip()
	df[COL_STATION] = df[COL_STATION].astype(str).str.strip()

	out: dict[str, set[str]] = {}
	for net, g in df.groupby(COL_NETWORK, sort=False):
		st_set = set(g[COL_STATION].tolist())
		if net == '':
			continue
		out[net] = {s for s in st_set if s != ''}
	return out


def _pick_representative_per_year(
	files: list[Path],
) -> dict[int, tuple[dt.datetime, Path]]:
	rep: dict[int, tuple[dt.datetime, Path]] = {}
	for p in files:
		when = _infer_snapshot_datetime(p)
		year = _infer_year(p, when)

		if year not in rep:
			rep[year] = (when, p)
			continue

		cur_when, _ = rep[year]
		if when < cur_when:
			rep[year] = (when, p)

	return rep


def main() -> None:
	if not ROOT.is_dir():
		raise FileNotFoundError(f'ROOT not found: {ROOT}')

	# 入力：snapshots/yearly 以下を再帰的に探索（年フォルダ直下でも、日時サブフォルダでも拾う）
	station_files = sorted(ROOT.rglob('stations_*.csv'))
	if not station_files:
		raise ValueError(f'no stations_*.csv found under: {ROOT}')

	rep = _pick_representative_per_year(station_files)
	years = sorted(rep.keys())
	if len(years) < 2:
		raise ValueError(f'need >=2 years of snapshots, got years={years}')

	# 年ごとの station 集合（network_code -> set(station)）をロード
	yearly_sets: dict[int, dict[str, set[str]]] = {}
	yearly_when: dict[int, dt.datetime] = {}
	yearly_path: dict[int, Path] = {}

	for y in years:
		when, path = rep[y]
		yearly_when[y] = when
		yearly_path[y] = path
		yearly_sets[y] = _load_station_sets(path)

	rows: list[dict[str, object]] = []
	cand_rows: list[dict[str, str]] = []

	for i in range(len(years) - 1):
		y_from = years[i]
		y_to = years[i + 1]
		when_from = yearly_when[y_from]
		when_to = yearly_when[y_to]

		sets_from = yearly_sets[y_from]
		sets_to = yearly_sets[y_to]

		nets = sorted(set(sets_from.keys()) | set(sets_to.keys()))

		for net in nets:
			has_from = net in sets_from
			has_to = net in sets_to

			s_from = sets_from.get(net, set())
			s_to = sets_to.get(net, set())

			added = sorted(s_to - s_from)
			removed = sorted(s_from - s_to)

			added_count = len(added)
			removed_count = len(removed)

			if has_from and has_to:
				status = (
					'no_change' if (added_count == 0 and removed_count == 0) else 'ok'
				)
			elif has_from and (not has_to):
				status = 'missing_to'
			elif (not has_from) and has_to:
				status = 'missing_from'
			else:
				continue

			rows.append(
				{
					'year_from': y_from,
					'year_to': y_to,
					'network_code': net,
					'added_count': added_count,
					'removed_count': removed_count,
					'added_list': ';'.join(added),
					'removed_list': ';'.join(removed),
					'status': status,
				}
			)

			if (added_count + removed_count) > 0:
				cand_rows.append(
					{
						'network_code': net,
						'start': when_from.date().isoformat(),
						'end': when_to.date().isoformat(),
					}
				)

	df_trans = pd.DataFrame(
		rows,
		columns=[
			'year_from',
			'year_to',
			'network_code',
			'added_count',
			'removed_count',
			'added_list',
			'removed_list',
			'status',
		],
	)

	ROOT.mkdir(parents=True, exist_ok=True)
	df_trans.to_csv(OUT_TRANSITIONS, index=False, encoding='utf-8')

	df_cand = pd.DataFrame(
		cand_rows, columns=['network_code', 'start', 'end']
	).drop_duplicates()
	df_cand.to_csv(OUT_CANDIDATES, index=False, encoding='utf-8')

	print(f'[INFO] wrote: {OUT_TRANSITIONS}')
	print(f'[INFO] wrote: {OUT_CANDIDATES}')
	print(f'[INFO] wrote: {ROOT / "yearly_snapshot_selection.csv"}')


if __name__ == '__main__':
	main()
