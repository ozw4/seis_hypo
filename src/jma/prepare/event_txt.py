# src/jma/prepare/event_txt.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class EventTxtMeta:
	origin_jst: datetime
	event_month: str  # YYYY-MM
	lat: float
	lon: float


def parse_origin_jst(s: str) -> datetime:
	ss = s.strip()
	parts = ss.split()
	if len(parts) != 2:
		raise ValueError(f'invalid ORIGIN_JST: {s!r}')
	date_part, time_part = parts
	y_s, m_s, d_s = date_part.split('/')
	h_s, mi_s, sec_s = time_part.split(':')

	if '.' in sec_s:
		sec_int_s, frac_s = sec_s.split('.', 1)
		frac_s = ''.join([c for c in frac_s if c.isdigit()])
		frac_s = (frac_s + '000000')[:6]
		micro = int(frac_s)
	else:
		sec_int_s = sec_s
		micro = 0

	return datetime(
		int(y_s),
		int(m_s),
		int(d_s),
		int(h_s),
		int(mi_s),
		int(sec_int_s),
		micro,
	)


def parse_latlon(s: str) -> float:
	ss = s.strip()
	if len(ss) < 2:
		raise ValueError(f'invalid lat/lon token: {s!r}')
	hem = ss[-1].upper()
	val = float(ss[:-1])
	if hem in {'N', 'E'}:
		return val
	if hem in {'S', 'W'}:
		return -val
	raise ValueError(f'invalid hemisphere in lat/lon: {s!r}')


def _read_kv_cp932(path: str | Path) -> dict[str, str]:
	path = Path(path)
	if not path.is_file():
		raise FileNotFoundError(path)

	lines = path.read_text(encoding='cp932', errors='strict').splitlines()
	kv: dict[str, str] = {}
	for raw in lines:
		line = raw.strip()
		if not line or ':' not in line:
			continue
		k, v = line.split(':', 1)
		key = k.strip().upper()
		val = v.strip()
		if key and val:
			kv[key] = val
	return kv


def read_event_txt_meta(path: str | Path) -> EventTxtMeta:
	path = Path(path)
	kv = _read_kv_cp932(path)

	req_keys = {'ORIGIN_JST', 'LATITUDE', 'LONGITUDE'}
	missing = sorted(req_keys - set(kv.keys()))
	if missing:
		raise ValueError(f'missing keys in {path.name}: {missing}')

	origin = parse_origin_jst(kv['ORIGIN_JST'])
	month = f'{origin.year:04d}-{origin.month:02d}'
	lat = parse_latlon(kv['LATITUDE'])
	lon = parse_latlon(kv['LONGITUDE'])

	return EventTxtMeta(origin_jst=origin, event_month=month, lat=lat, lon=lon)


def read_origin_jst_iso(txt_path: str | Path) -> str:
	kv = _read_kv_cp932(txt_path)
	if 'ORIGIN_JST' not in kv:
		raise ValueError(f'ORIGIN_JST not found in {txt_path}')

	dt = parse_origin_jst(kv['ORIGIN_JST'])
	frac2 = f'{dt.microsecond // 10000:02d}'
	return f'{dt:%Y-%m-%dT%H:%M:%S}.{frac2}'
