# output_loki_parse.py（= 今の pasted.txt をこの内容に置き換え）

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class LokiLocRow:
	trial: int
	x_km: float
	y_km: float
	z_km: float
	cmax: float
	# LOKIのバージョンによって追加列があるので保持
	extra: tuple[float, ...] = ()


@dataclass(frozen=True)
class LokiPhsRow:
	station: str
	p_arrival_sec: float
	s_arrival_sec: float


@dataclass(frozen=True)
class LokiEventResult:
	event_name: str
	event_dir: Path
	loc_path: Path
	loc_rows: tuple[LokiLocRow, ...]
	phs_paths: tuple[Path, ...]
	phs_by_trial: dict[int, tuple[LokiPhsRow, ...]]
	corrmatrix_paths: tuple[Path, ...]


@dataclass(frozen=True)
class LokiCatalogueRow:
	origin_time: str
	lat: float
	lon: float
	depth_km: float
	errmax: float
	cb: float
	cmax: float


def _read_text_lines(path: Path) -> list[str]:
	if not path.is_file():
		raise FileNotFoundError(f'file not found: {path}')
	txt = path.read_text(encoding='utf-8', errors='strict')
	lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
	if not lines:
		raise ValueError(f'file is empty: {path}')
	return lines


def parse_loc_file(loc_path: Path) -> tuple[LokiLocRow, ...]:
	""".loc は通常 1行（trialが1つ）だけど、複数trial/複数行でも読めるようにする。
	format: trial x y z cmax [extra...]
	"""
	rows: list[LokiLocRow] = []
	for ln in _read_text_lines(loc_path):
		cols = ln.split()
		if len(cols) < 5:
			raise ValueError(
				f"invalid .loc line (need >=5 cols): {loc_path} line='{ln}'"
			)

		trial = int(float(cols[0]))
		x_km = float(cols[1])
		y_km = float(cols[2])
		z_km = float(cols[3])
		cmax = float(cols[4])
		extra: tuple[float, ...] = (
			tuple(float(c) for c in cols[5:]) if len(cols) > 5 else ()
		)

		rows.append(
			LokiLocRow(
				trial=trial, x_km=x_km, y_km=y_km, z_km=z_km, cmax=cmax, extra=extra
			)
		)

	if not rows:
		raise ValueError(f'no loc rows parsed: {loc_path}')

	rows.sort(key=lambda r: r.trial)
	return tuple(rows)


_TRIAL_FROM_NAME = re.compile(r'_trial(\d+)\.phs$')


def _infer_trial_from_phs_name(phs_path: Path) -> int:
	m = _TRIAL_FROM_NAME.search(phs_path.name)
	if not m:
		raise ValueError(
			f"cannot infer trial from phs filename: {phs_path.name} (need '*_trialN.phs')"
		)
	return int(m.group(1))


def _iter_phs_tokens(phs_path: Path) -> Iterator[tuple[str, str, str]]:
	"""Yield (station, p_token, s_token) triples from a .phs file."""
	for ln in _read_text_lines(phs_path):
		if ln.startswith('#'):
			continue
		cols = ln.split()
		if not cols:
			continue
		if cols[0].lower() == 'station':
			continue
		if len(cols) < 3:
			raise ValueError(
				f"invalid .phs line (need >=3 cols): {phs_path} line='{ln}'"
			)
		yield cols[0], cols[1], cols[2]


def _infer_origin_from_phs_filename(phs_path: Path) -> pd.Timestamp | None:
	"""例: 2020-02-09T15:40:29.255374_trial0.phs
	-> "2020-02-09T15:40:29.255374" を origin として使う
	"""
	name = phs_path.name

	# suffix を落とす
	stem = name.removesuffix('.phs')

	# "_trial" より前がISO時刻の想定
	key = '_trial'
	i = stem.find(key)
	if i <= 0:
		return None

	prefix = stem[:i]
	origin = pd.to_datetime(prefix)
	if pd.isna(origin):
		return None
	return pd.Timestamp(origin)


def _require_event_origin(phs_path: Path) -> pd.Timestamp:
	"""ISO到達時刻を秒へ変換するための基準時刻（origin）を決定する。"""
	phs_path = Path(phs_path)

	origin = _infer_origin_from_phs_filename(phs_path)
	if origin is None:
		raise ValueError(
			'cannot infer event origin time from .phs filename. '
			f"expected '<ISO>_trial*.phs', got filename={phs_path.name!r}"
		)
	return origin


def _phs_token_to_seconds(
	token: str,
	ensure_origin: Callable[[], pd.Timestamp],
) -> float:
	try:
		return float(token)
	except ValueError:
		pass

	arrival = pd.to_datetime(token)
	if pd.isna(arrival):
		raise ValueError(f'failed to parse arrival time token: {token!r}')

	origin = ensure_origin()
	return _timestamp_to_relative_seconds(pd.Timestamp(arrival), origin)


def _timestamp_to_relative_seconds(
	arrival: pd.Timestamp,
	origin: pd.Timestamp,
) -> float:
	arrival_has_tz = arrival.tzinfo is not None
	origin_has_tz = origin.tzinfo is not None
	if arrival_has_tz != origin_has_tz:
		raise ValueError(
			'timezone mismatch between arrival token and event origin; '
			'ensure both encode timezone or both are naive'
		)

	if arrival_has_tz:
		arrival = arrival.tz_convert('UTC')
		origin = origin.tz_convert('UTC')

	return float((arrival - origin).total_seconds())


def parse_phs_file(phs_path: Path) -> tuple[LokiPhsRow, ...]:
	""".phs: station P/S arrival tokens. Returns seconds relative to event origin."""
	phs_path = Path(phs_path)
	rows: list[LokiPhsRow] = []

	origin_cache: pd.Timestamp | None = None

	def _ensure_origin() -> pd.Timestamp:
		nonlocal origin_cache
		if origin_cache is None:
			origin_cache = _require_event_origin(phs_path)
		return origin_cache

	for sta, p_tok, s_tok in _iter_phs_tokens(phs_path):
		p_sec = _phs_token_to_seconds(p_tok, _ensure_origin)
		s_sec = _phs_token_to_seconds(s_tok, _ensure_origin)
		rows.append(
			LokiPhsRow(
				station=sta,
				p_arrival_sec=p_sec,
				s_arrival_sec=s_sec,
			)
		)

	if not rows:
		raise ValueError(f'no phs rows parsed: {phs_path}')

	return tuple(rows)


def parse_catalogue(catalogue_path: Path) -> tuple[LokiCatalogueRow, ...]:
	"""LOKI output_dir/catalogue を読む。
	format: origin_time lat lon depth_km errmax cb cmax
	"""
	rows: list[LokiCatalogueRow] = []
	for ln in _read_text_lines(catalogue_path):
		cols = ln.split()
		if len(cols) < 7:
			raise ValueError(
				f"invalid catalogue line (need >=7 cols): {catalogue_path} line='{ln}'"
			)

		rows.append(
			LokiCatalogueRow(
				origin_time=str(cols[0]),
				lat=float(cols[1]),
				lon=float(cols[2]),
				depth_km=float(cols[3]),
				errmax=float(cols[4]),
				cb=float(cols[5]),
				cmax=float(cols[6]),
			)
		)
	return tuple(rows)


def load_loki_catalogue_as_events_df(loki_output_dir: str | Path) -> pd.DataFrame:
	"""LOKI output_dir/catalogue を DataFrame にする（描画関数にそのまま渡せる列名で返す）"""
	loki_output_dir = Path(loki_output_dir)
	catalogue_path = loki_output_dir / 'catalogue'
	rows = parse_catalogue(catalogue_path)

	df = pd.DataFrame(
		[
			{
				'origin_time': r.origin_time,
				'latitude_deg': r.lat,
				'longitude_deg': r.lon,
				'depth_km': r.depth_km,
				'errmax': r.errmax,
				'cb': r.cb,
				'cmax': r.cmax,
			}
			for r in rows
		]
	)
	df['origin_time'] = pd.to_datetime(df['origin_time'])
	return df


def _find_one_loc(event_out_dir: Path) -> Path:
	locs = sorted(event_out_dir.glob('*.loc'))
	if not locs:
		raise FileNotFoundError(f'no .loc in event dir: {event_out_dir}')
	if len(locs) == 1:
		return locs[0]
	raise ValueError(
		f'multiple .loc found in {event_out_dir}: {[p.name for p in locs]}'
	)


def parse_loki_event_dir(event_out_dir: Path) -> LokiEventResult:
	if not event_out_dir.is_dir():
		raise FileNotFoundError(f'event dir not found: {event_out_dir}')

	loc_path = _find_one_loc(event_out_dir)
	loc_rows = parse_loc_file(loc_path)

	phs_paths = tuple(sorted(event_out_dir.glob('*_trial*.phs')))
	phs_by_trial: dict[int, tuple[LokiPhsRow, ...]] = {}
	for phs in phs_paths:
		trial = _infer_trial_from_phs_name(phs)
		phs_by_trial[trial] = parse_phs_file(phs)

	corr_paths = tuple(sorted(event_out_dir.glob('corrmatrix_trial_*.npy')))

	return LokiEventResult(
		event_name=event_out_dir.name,
		event_dir=event_out_dir,
		loc_path=loc_path,
		loc_rows=loc_rows,
		phs_paths=phs_paths,
		phs_by_trial=phs_by_trial,
		corrmatrix_paths=corr_paths,
	)


def list_event_dirs(loki_output_dir: Path, event_glob: str = '[0-9]*') -> list[Path]:
	if not loki_output_dir.is_dir():
		raise FileNotFoundError(f'loki_output_dir not found: {loki_output_dir}')
	return sorted([p for p in loki_output_dir.glob(event_glob) if p.is_dir()])


def parse_loki_output_dir(
	loki_output_dir: str | Path,
	*,
	event_glob: str = '[0-9]*',
) -> tuple[LokiEventResult, ...]:
	loki_output_dir = Path(loki_output_dir)
	evdirs = list_event_dirs(loki_output_dir, event_glob=event_glob)
	if not evdirs:
		raise ValueError(f"no event dirs in {loki_output_dir} with glob '{event_glob}'")

	return tuple(parse_loki_event_dir(d) for d in evdirs)


@dataclass(frozen=True)
class HeaderOrigin:
	lat0_deg: float
	lon0_deg: float
	x0_km: float = 0.0
	y0_km: float = 0.0


@dataclass(frozen=True)
class LokiHeader:
	origin: HeaderOrigin
	stations_df: pd.DataFrame  # columns: station, lat, lon


def parse_loki_header(header_path: Path) -> LokiHeader:
	"""header.hdr を1回だけ読み、origin と stations を同時に返す。

	固定フォーマット前提:
	0: nx ny nz
	1: x0 y0 z0   (km)
	2: dx dy dz
	3: lat0 lon0
	4+: station lat lon elev
	"""
	header_path = Path(header_path)
	if not header_path.is_file():
		raise FileNotFoundError(f'header not found: {header_path}')

	lines = header_path.read_text(encoding='utf-8', errors='ignore').splitlines()
	lines = [ln.strip() for ln in lines if ln and ln.strip()]

	if len(lines) < 5:
		raise ValueError(f'header too short (<5 lines): {header_path}')

	x0_y0_z0 = lines[1].split()
	lat0_lon0 = lines[3].split()
	if len(x0_y0_z0) < 2 or len(lat0_lon0) < 2:
		raise ValueError(f'invalid header format: {header_path}')

	x0_km = float(x0_y0_z0[0])
	y0_km = float(x0_y0_z0[1])
	lat0 = float(lat0_lon0[0])
	lon0 = float(lat0_lon0[1])
	origin = HeaderOrigin(lat0_deg=lat0, lon0_deg=lon0, x0_km=x0_km, y0_km=y0_km)

	rows: list[dict] = []
	for ln in lines[4:]:
		cols = ln.split()
		if len(cols) < 4:
			continue
		rows.append(
			{
				'station': cols[0],
				'lat': float(cols[1]),
				'lon': float(cols[2]),
			}
		)
	if not rows:
		raise ValueError(f'no station rows parsed from header: {header_path}')

	stations_df = pd.DataFrame(rows)
	return LokiHeader(origin=origin, stations_df=stations_df)


def parse_header_origin(header_path: Path) -> HeaderOrigin:
	return parse_loki_header(header_path).origin


def parse_phs_absolute_times(phs_path: Path, *, tz: str = 'utc') -> pd.DataFrame:
	""".phs -> DataFrame(station, tp, ts) with absolute arrival timestamps."""
	tz_mode = tz.lower()
	if tz_mode not in {'utc', 'naive'}:
		raise ValueError(f"tz must be 'utc' or 'naive', got {tz}")

	def _parse_token(token: str) -> pd.Timestamp:
		# Float seconds are not supported here to avoid implicit assumptions.
		try:
			float(token)
		except ValueError:
			pass
		else:
			raise ValueError(
				f'absolute-time parser expected ISO timestamps, found float token: {token!r}'
			)

		ts = pd.to_datetime(token, utc=True)
		if pd.isna(ts):
			raise ValueError(f'failed to parse arrival time token: {token!r}')

		if tz_mode == 'utc':
			return pd.Timestamp(ts)
		return pd.Timestamp(ts.tz_convert('UTC').tz_localize(None))

	# Collect all picks per station to resolve duplicates deterministically.
	per_sta: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
	for sta, p_tok, s_tok in _iter_phs_tokens(phs_path):
		try:
			tp = _parse_token(p_tok)
			ts = _parse_token(s_tok)
		except ValueError:
			# skip malformed rows; robust to partial data
			continue
		if pd.isna(tp) or pd.isna(ts):
			continue
		per_sta.setdefault(sta, []).append((tp, ts))

	rows: list[dict] = []
	for sta, pairs in per_sta.items():
		if not pairs:
			continue
		# pick earliest tp/ts
		tp_vals = [p for p, _ in pairs if not pd.isna(p)]
		ts_vals = [s for _, s in pairs if not pd.isna(s)]
		if not tp_vals or not ts_vals:
			continue
		tp_min = min(tp_vals)
		ts_min = min(ts_vals)
		rows.append({'station': sta, 'tp': tp_min, 'ts': ts_min})

	if not rows:
		raise ValueError(f'no valid rows in phs: {phs_path}')

	return pd.DataFrame(rows)
