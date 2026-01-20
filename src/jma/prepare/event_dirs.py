from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Literal


def parse_date_yyyy_mm_dd(
	s: str | None, *, allow_slash: bool = False, allow_time: bool = False
) -> date | None:
	if s is None:
		return None
	ss = str(s).strip()
	if not ss:
		return None
	if not allow_slash and not allow_time:
		y, m, d = ss.split('-')
		return date(int(y), int(m), int(d))

	ss2 = ss
	if allow_time:
		if 'T' in ss2:
			ss2 = ss2.split('T', 1)[0]
		elif ' ' in ss2:
			ss2 = ss2.split(' ', 1)[0]

	if '-' in ss2:
		parts = ss2.split('-')
	elif allow_slash and '/' in ss2:
		parts = ss2.split('/')
	else:
		raise ValueError(f'invalid date string (expected YYYY-MM-DD): {s!r}')
	if len(parts) != 3:
		raise ValueError(f'invalid date string (expected YYYY-MM-DD): {s!r}')
	y_s, m_s, d_s = parts
	return date(int(y_s), int(m_s), int(d_s))


def in_date_range(
	d: date | datetime,
	*,
	date_min: date | None,
	date_max: date | None,
) -> bool:
	dd = d.date() if isinstance(d, datetime) else d
	if date_min is not None and dd < date_min:
		return False
	if date_max is not None and dd > date_max:
		return False
	return True


def event_dir_date_jst_from_name(
	dirname: str, *, on_invalid: Literal['raise', 'none'] = 'raise'
) -> date | None:
	name = str(dirname).strip()
	if len(name) < 9 or not name.startswith('D'):
		if on_invalid == 'none':
			return None
		raise ValueError(
			f'unexpected event dir name (need DYYYYMMDD...): {dirname!r}'
		)
	ymd = name[1:9]
	if not ymd.isdigit():
		if on_invalid == 'none':
			return None
		raise ValueError(f'unexpected event dir name (bad ymd): {dirname!r}')
	y = int(ymd[0:4])
	m = int(ymd[4:6])
	d = int(ymd[6:8])
	return date(y, m, d)


def list_event_dirs(
	win_event_dir: Path,
	*,
	target_names: Iterable[str] | None = None,
	date_min: date | None = None,
	date_max: date | None = None,
	dir_glob: str = 'D20*',
	invalid_name: Literal['raise', 'skip', 'keep'] = 'raise',
) -> list[Path]:
	base_dir = Path(win_event_dir)
	if target_names:
		out: list[Path] = []
		for name in target_names:
			p = (base_dir / name).resolve()
			if not p.is_dir():
				raise FileNotFoundError(f'event_dir not found: {p}')
			out.append(p)
	else:
		out = sorted([p for p in base_dir.glob(dir_glob) if p.is_dir()])

	if date_min is None and date_max is None and invalid_name != 'skip':
		return out

	filtered: list[Path] = []
	for p in out:
		if invalid_name == 'raise':
			dd = event_dir_date_jst_from_name(p.name, on_invalid='raise')
			if date_min is None and date_max is None:
				filtered.append(p)
				continue
			if in_date_range(dd, date_min=date_min, date_max=date_max):
				filtered.append(p)
			continue

		dd = event_dir_date_jst_from_name(p.name, on_invalid='none')
		if dd is None:
			if invalid_name == 'keep':
				filtered.append(p)
			continue
		if date_min is None and date_max is None:
			filtered.append(p)
			continue
		if in_date_range(dd, date_min=date_min, date_max=date_max):
			filtered.append(p)

	return filtered
