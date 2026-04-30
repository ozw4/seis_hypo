# %%
"""Build Loki-ready WIN32 event directories from Izu2009 GaMMA outputs."""

from __future__ import annotations

import datetime as dt
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _REPO_ROOT / 'src'

for _path in (_REPO_ROOT, _SRC_DIR):
	_path_str = str(_path)
	if _path_str not in sys.path:
		sys.path.insert(0, _path_str)

from common.core import (  # noqa: E402
	load_event_json,
	validate_columns,
	write_event_json_win32_groups,
)
from common.json_io import write_json  # noqa: E402
from jma.station_reader import read_hinet_channel_table  # noqa: E402
from jma.win32_reader import (  # noqa: E402
	compute_event_time_window,
	scan_channel_sampling_rate_map_win32,
)
from pipelines.win32_eqt_continuous_pipelines import (  # noqa: E402
	parse_win32_cnt_filename,
)

GAMMA_EVENTS_CSV = _REPO_ROOT / 'proc/izu2009/association/out/gamma_events.csv'
GAMMA_PICKS_CSV = _REPO_ROOT / 'proc/izu2009/association/out/gamma_picks.csv'
GAMMA_EVENTS_QC_CSV = (
	_REPO_ROOT / 'proc/izu2009/association/qc/gamma_events_qc_flags.csv'
)
GAMMA_EVENTS_LATLON_CSV = (
	_REPO_ROOT / 'proc/izu2009/association/qc/gamma_events_with_latlon.csv'
)
STATIONS47_CSV = (
	_REPO_ROOT / 'proc/izu2009/prepare_data/profile/stations47/stations_47.csv'
)

OUT_BASE_DIR = _REPO_ROOT / 'proc/izu2009/loki/events_from_gamma'

CONT_BASE_DIR_BY_NETWORK = {
	'0101': Path('/workspace/data/izu2009/continuous/0101'),
	'0203': Path('/workspace/data/izu2009/continuous/0203'),
	'0207': Path('/workspace/data/izu2009/continuous/0207'),
	'0301': Path('/workspace/data/izu2009/continuous/0301'),
}

CH47_BASE_DIR_BY_NETWORK = {
	'0101': _REPO_ROOT
	/ 'proc/izu2009/prepare_data/download_continuous/continuous_ch47/0101',
	'0203': _REPO_ROOT
	/ 'proc/izu2009/prepare_data/download_continuous/continuous_ch47/0203',
	'0207': _REPO_ROOT
	/ 'proc/izu2009/prepare_data/download_continuous/continuous_ch47/0207',
	'0301': _REPO_ROOT
	/ 'proc/izu2009/prepare_data/download_continuous/continuous_ch47/0301',
}

PRE_SEC = 20
POST_SEC = 80
SPAN_MIN_DEFAULT = 10
THREADS = 8
USE_SYMLINK = True

EVENT_ID_MODE = 'sequential'
EVENT_ID_START = 1

MAX_EVENTS: int | None = 50

MIN_NUM_PICKS = 15
MIN_NUM_P_PICKS = 5
MIN_NUM_S_PICKS = 5
MAX_SIGMA_TIME = 0.75
REQUIRE_NOT_NEAR_BOUNDARY_1KM = True
MIN_LOKI_STATIONS_PER_EVENT = 3
MIN_LOKI_NETWORKS_PER_EVENT = 1

_JST = dt.timezone(dt.timedelta(hours=9))
_STATION_ID_SPLIT = '__'
_EVENT_TIME_COL_CANDIDATES = ('time', 'timestamp', 'event_time', 'origin_time')
_DROPPED_EVENTS_COLUMNS = [
	'event_index',
	'origin_time_utc',
	'reason',
	'n_station_gamma',
	'n_station_loki',
	'n_network_loki',
	'dropped_station_count_3c',
	'dropped_stations_3c',
	'dropped_station_count_cnt_missing',
	'dropped_stations_cnt_missing',
]


@dataclass(frozen=True, slots=True)
class CntRecord:
	"""One WIN32 continuous tile."""

	path: Path
	network_code: str
	start_jst: dt.datetime
	end_jst: dt.datetime
	span_min: int


@dataclass(frozen=True, slots=True)
class EventBuildResult:
	"""One built event directory and its summary row."""

	event_dir: Path
	summary_row: dict[str, object]


@dataclass(frozen=True, slots=True)
class EventDropResult:
	"""One skipped event and its drop summary row."""

	summary_row: dict[str, object]


def _repo_rel(path: Path) -> str:
	p = Path(path)
	try:
		return str(p.resolve().relative_to(_REPO_ROOT.resolve()))
	except ValueError:
		return str(p)


def _normalize_network_mapping_keys(
	mapping: dict[str, Path],
	*,
	label: str,
) -> dict[str, Path]:
	out: dict[str, Path] = {}
	for net, path in mapping.items():
		code = str(net).strip().upper()
		if not code:
			raise ValueError(f'{label} contains empty network_code')
		if code in out:
			raise ValueError(
				f'{label} has duplicated network_code after upper-normalization: {code}'
			)
		out[code] = Path(path)
	return out


def _to_int_series_strict(series: pd.Series, label: str) -> pd.Series:
	num = pd.to_numeric(series, errors='raise')
	if num.isna().any():
		raise ValueError(f'{label} contains NaN')

	num_arr = num.to_numpy(dtype=float)
	num_int = np.rint(num_arr).astype(np.int64)
	if not np.allclose(num_arr, num_int.astype(float), atol=1e-9):
		bad_idx = np.where(np.abs(num_arr - num_int.astype(float)) > 1e-9)[0][:10]
		bad_vals = [float(num_arr[i]) for i in bad_idx]
		raise ValueError(f'{label} must be integer-like, bad values={bad_vals}')

	return pd.Series(num_int, index=series.index, dtype='int64')


def _to_bool_series_strict(series: pd.Series, label: str) -> pd.Series:
	def parse_one(value: object) -> bool:
		if isinstance(value, bool):
			return bool(value)
		if pd.isna(value):
			raise ValueError(f'{label} contains NaN')
		if isinstance(value, (np.bool_,)):
			return bool(value)
		if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
			return bool(int(value))
		s = str(value).strip().lower()
		if s in {'true', 't', '1'}:
			return True
		if s in {'false', 'f', '0'}:
			return False
		raise ValueError(f'{label} must be boolean-like, got {value!r}')

	return series.map(parse_one).astype(bool)


def _require_unique_event_index(df: pd.DataFrame, label: str) -> None:
	if df['event_index'].duplicated().any():
		dup = df.loc[df['event_index'].duplicated(), 'event_index'].astype(int).tolist()
		raise ValueError(f'duplicated event_index in {label}: {dup[:20]}')


def _resolve_event_time_column(events_df: pd.DataFrame) -> str:
	for col in _EVENT_TIME_COL_CANDIDATES:
		if col in events_df.columns:
			return col
	raise ValueError(
		'gamma_events.csv must contain one of time columns: '
		f'{list(_EVENT_TIME_COL_CANDIDATES)}'
	)


def _load_gamma_events(events_csv: Path) -> pd.DataFrame:
	events = pd.read_csv(events_csv)
	if events.empty:
		raise ValueError(f'gamma_events.csv is empty: {events_csv}')

	validate_columns(
		events,
		['event_index', 'x(km)', 'y(km)', 'z(km)'],
		f'gamma_events.csv: {events_csv}',
	)
	time_col = _resolve_event_time_column(events)

	events = events.copy()
	events['event_index'] = _to_int_series_strict(
		events['event_index'], 'gamma_events.event_index'
	)
	_require_unique_event_index(events, 'gamma_events.csv')
	events['event_time_utc'] = pd.to_datetime(
		events[time_col], utc=True, errors='raise'
	)
	return events


def _split_station_id(station_id: str) -> tuple[str, str]:
	s = str(station_id).strip()
	if s.count(_STATION_ID_SPLIT) != 1:
		raise ValueError(f'invalid station_id format (need <net>__<station>): {s!r}')

	net, sta = s.split(_STATION_ID_SPLIT, 1)
	net = net.strip().upper()
	sta = sta.strip().upper()
	if not net or not sta:
		raise ValueError(f'invalid station_id format (empty token): {s!r}')
	return net, sta


def _load_gamma_picks(picks_csv: Path) -> pd.DataFrame:
	picks = pd.read_csv(picks_csv)
	if picks.empty:
		raise ValueError(f'gamma_picks.csv is empty: {picks_csv}')

	validate_columns(
		picks,
		['station_id', 'phase_type', 'event_index'],
		f'gamma_picks.csv: {picks_csv}',
	)

	picks = picks.copy()
	picks['event_index'] = _to_int_series_strict(
		picks['event_index'], 'gamma_picks.event_index'
	)
	picks['phase_type'] = picks['phase_type'].astype(str).str.strip().str.upper()

	parts = picks['station_id'].astype(str).map(_split_station_id)
	picks['network_code'] = [t[0] for t in parts]
	picks['station_code'] = [t[1] for t in parts]

	picks = picks[picks['event_index'] >= 0].reset_index(drop=True)
	if picks.empty:
		raise ValueError('gamma_picks.csv has no assigned picks (event_index >= 0)')
	return picks


def _load_gamma_qc(qc_csv: Path) -> pd.DataFrame:
	qc = pd.read_csv(qc_csv)
	if qc.empty:
		raise ValueError(f'gamma_events_qc_flags.csv is empty: {qc_csv}')

	required = [
		'event_index',
		'num_picks',
		'num_p_picks',
		'num_s_picks',
		'sigma_time',
		'gamma_score',
		'near_any_boundary_1km',
	]
	validate_columns(qc, required, f'gamma_events_qc_flags.csv: {qc_csv}')

	qc = qc.copy()
	qc['event_index'] = _to_int_series_strict(
		qc['event_index'], 'gamma_events_qc_flags.event_index'
	)
	_require_unique_event_index(qc, 'gamma_events_qc_flags.csv')
	qc['near_any_boundary_1km'] = _to_bool_series_strict(
		qc['near_any_boundary_1km'], 'near_any_boundary_1km'
	)
	return qc


def _load_gamma_latlon(latlon_csv: Path) -> pd.DataFrame:
	latlon = pd.read_csv(latlon_csv)
	if latlon.empty:
		raise ValueError(f'gamma_events_with_latlon.csv is empty: {latlon_csv}')

	validate_columns(
		latlon,
		['event_index', 'latitude_deg', 'longitude_deg', 'depth_km'],
		f'gamma_events_with_latlon.csv: {latlon_csv}',
	)

	latlon = latlon.copy()
	latlon['event_index'] = _to_int_series_strict(
		latlon['event_index'], 'gamma_events_with_latlon.event_index'
	)
	_require_unique_event_index(latlon, 'gamma_events_with_latlon.csv')
	return latlon


def _load_stations47(stations_csv: Path) -> pd.DataFrame:
	stations = pd.read_csv(stations_csv, dtype={'network_code': str, 'station': str})
	if stations.empty:
		raise ValueError(f'stations_47.csv is empty: {stations_csv}')

	validate_columns(
		stations,
		['network_code', 'station'],
		f'stations_47.csv: {stations_csv}',
	)
	stations = stations.copy()
	stations['network_code'] = (
		stations['network_code'].astype(str).str.strip().str.zfill(4).str.upper()
	)
	stations['station'] = stations['station'].astype(str).str.strip().str.upper()
	if stations[['network_code', 'station']].duplicated().any():
		dup = stations.loc[
			stations[['network_code', 'station']].duplicated(),
			['network_code', 'station'],
		]
		raise ValueError(
			'duplicated network/station rows in stations_47.csv: '
			f'{dup.head(20).to_dict(orient="records")}'
		)
	return stations


def _build_cnt_index_for_network(network_code: str, base_dir: Path) -> list[CntRecord]:
	if not base_dir.is_dir():
		raise FileNotFoundError(
			f'continuous base dir not found for {network_code}: {base_dir}'
		)

	records: list[CntRecord] = []
	for cnt_path in sorted(base_dir.glob('*.cnt')):
		info = parse_win32_cnt_filename(cnt_path)
		info_net = str(info.network_code).strip().upper()
		if info_net != str(network_code):
			raise ValueError(
				f'network_code mismatch in {cnt_path}: '
				f'expected={network_code}, parsed={info.network_code}'
			)
		start_jst = info.start_jst
		end_jst = start_jst + dt.timedelta(minutes=int(info.span_min))
		records.append(
			CntRecord(
				path=cnt_path,
				network_code=info_net,
				start_jst=start_jst,
				end_jst=end_jst,
				span_min=int(info.span_min),
			)
		)

	if not records:
		raise FileNotFoundError(
			f'no .cnt files found for network={network_code}: {base_dir}'
		)

	span_set = sorted({r.span_min for r in records})
	if len(span_set) != 1:
		raise ValueError(
			f'mixed span_min in network={network_code}: span_min_set={span_set}'
		)
	if int(span_set[0]) != int(SPAN_MIN_DEFAULT):
		raise ValueError(
			f'SPAN_MIN_DEFAULT mismatch for network={network_code}: '
			f'expected={int(SPAN_MIN_DEFAULT)}, found={int(span_set[0])}'
		)

	records.sort(key=lambda r: (r.start_jst, r.path.name))
	return records


def _build_cnt_index(
	cont_base_dir_by_network: dict[str, Path],
) -> dict[str, list[CntRecord]]:
	idx: dict[str, list[CntRecord]] = {}
	for net in sorted(cont_base_dir_by_network):
		code = str(net).strip().upper()
		if not code:
			raise ValueError('CONT_BASE_DIR_BY_NETWORK contains empty network_code')
		idx[code] = _build_cnt_index_for_network(
			code, Path(cont_base_dir_by_network[net])
		)
	return idx


def _select_cnt_covering_window(
	records: list[CntRecord],
	*,
	network_code: str,
	t_start_jst: dt.datetime,
	t_end_jst: dt.datetime,
) -> list[CntRecord]:
	if t_end_jst <= t_start_jst:
		raise ValueError('invalid time window: t_end_jst <= t_start_jst')

	overlap = [
		r for r in records if (r.end_jst > t_start_jst) and (r.start_jst < t_end_jst)
	]
	if not overlap:
		raise ValueError(
			f'no .cnt intersects window for network={network_code}: '
			f'{t_start_jst} - {t_end_jst}'
		)

	overlap.sort(key=lambda r: (r.start_jst, r.path.name))

	covered_until = t_start_jst
	selected: list[CntRecord] = []
	for rec in overlap:
		if rec.end_jst <= covered_until:
			continue
		if rec.start_jst > covered_until:
			raise ValueError(
				f'.cnt gap for network={network_code}: '
				f'gap_start={covered_until}, next_cnt_start={rec.start_jst}, '
				f'window={t_start_jst} - {t_end_jst}'
			)
		selected.append(rec)
		covered_until = rec.end_jst
		if covered_until >= t_end_jst:
			break

	if covered_until < t_end_jst:
		raise ValueError(
			f'window is not fully covered for network={network_code}: '
			f'covered_until={covered_until}, need_end={t_end_jst}'
		)

	return selected


def _materialize_file(src: Path, dst: Path, *, use_symlink: bool) -> None:
	if dst.exists() or dst.is_symlink():
		raise FileExistsError(f'destination already exists: {dst}')
	if use_symlink:
		dst.symlink_to(src.resolve())
	else:
		shutil.copy2(src, dst)


def _ch_path_for_cnt(network_code: str, cnt_path: Path) -> Path:
	if network_code not in CH47_BASE_DIR_BY_NETWORK:
		raise ValueError(
			f'CH47_BASE_DIR_BY_NETWORK missing network from picks: {network_code}'
		)
	ch_path = CH47_BASE_DIR_BY_NETWORK[network_code] / f'{cnt_path.stem}.ch'
	if not ch_path.is_file():
		raise FileNotFoundError(
			f'same-stem 47-station .ch not found for network={network_code}: {ch_path}'
		)
	return ch_path


def _norm_station_name(value: object) -> str:
	return str(value).strip().upper()


def _station_set_from_ch(ch_path: Path) -> set[str]:
	ch_df = read_hinet_channel_table(ch_path)
	validate_columns(ch_df, ['station'], f'channel table: {ch_path}')
	return set(ch_df['station'].map(_norm_station_name).tolist())


def _three_component_station_channels_from_ch(ch_path: Path) -> dict[str, set[int]]:
	ch_df = read_hinet_channel_table(ch_path)
	validate_columns(
		ch_df,
		['station', 'component', 'ch_int'],
		f'channel table: {ch_path}',
	)

	ch_df = ch_df.copy()
	ch_df['station_norm'] = ch_df['station'].map(_norm_station_name)
	ch_df['component_norm'] = ch_df['component'].astype(str).str.strip().str.upper()

	out: dict[str, set[int]] = {}
	for station_norm, sub in ch_df.groupby('station_norm', sort=True):
		component_rows = sub[sub['component_norm'].isin(['U', 'N', 'E'])]
		if component_rows['component_norm'].nunique() != 3:
			continue
		out[str(station_norm)] = set(component_rows['ch_int'].astype(int).tolist())

	return out


def _available_channel_ints_in_cnt(
	cnt_path: Path,
	channel_filter: set[int],
) -> set[int]:
	fs_by_ch = scan_channel_sampling_rate_map_win32(
		cnt_path,
		channel_filter=channel_filter,
		on_mixed='drop',
	)
	return set(int(ch) for ch in fs_by_ch.keys())


def _validate_stations_in_ch(
	*,
	stations: list[str],
	ch_paths: list[Path],
	network_code: str,
) -> None:
	required = set(stations)
	for ch_path in ch_paths:
		available = _station_set_from_ch(ch_path)
		missing = sorted(required - available)
		if missing:
			raise ValueError(
				f'event station missing from .ch for network={network_code}: '
				f'ch={ch_path}, missing={missing[:20]}'
			)


def _filter_stations_with_three_components(
	*,
	stations: list[str],
	cnt_paths: list[Path],
	ch_paths: list[Path],
	network_code: str,
) -> tuple[list[str], dict[str, str], dict[str, str]]:
	if not ch_paths:
		raise ValueError(f'ch_paths is empty for network={network_code}')
	if len(cnt_paths) != len(ch_paths):
		raise ValueError(
			f'cnt_paths/ch_paths length mismatch for network={network_code}: '
			f'cnt={len(cnt_paths)} ch={len(ch_paths)}'
		)

	station_by_norm = {_norm_station_name(sta): str(sta) for sta in stations}
	usable_station_norms = set(station_by_norm)
	drop_reasons_3c: dict[str, str] = {}
	drop_reasons_cnt_missing: dict[str, str] = {}

	for cnt_path, ch_path in zip(cnt_paths, ch_paths, strict=True):
		station_channels = _three_component_station_channels_from_ch(ch_path)

		needed_channel_ints: set[int] = set()
		for station_norm in usable_station_norms:
			channel_ints = station_channels.get(station_norm)
			if channel_ints is not None:
				needed_channel_ints.update(channel_ints)

		available_channel_ints = _available_channel_ints_in_cnt(
			cnt_path,
			needed_channel_ints,
		)

		next_usable_station_norms: set[str] = set()
		for station_norm in sorted(usable_station_norms):
			station_code = station_by_norm[station_norm]
			channel_ints = station_channels.get(station_norm)
			if channel_ints is None:
				drop_reasons_3c.setdefault(
					station_code,
					f'missing_3c_in_ch:{ch_path.name}',
				)
				continue

			missing_ch = sorted(channel_ints - available_channel_ints)
			if missing_ch:
				missing_text = ','.join(str(x) for x in missing_ch)
				drop_reasons_cnt_missing.setdefault(
					station_code,
					f'missing_channels_in_cnt:{cnt_path.name}:{missing_text}',
				)
				continue

			next_usable_station_norms.add(station_norm)

		usable_station_norms = next_usable_station_norms

	usable_stations = [
		sta for sta in stations if _norm_station_name(sta) in usable_station_norms
	]
	return usable_stations, drop_reasons_3c, drop_reasons_cnt_missing


def _resolve_event_id(order_idx: int, event_index: int) -> int:
	if EVENT_ID_MODE == 'sequential':
		return int(EVENT_ID_START) + int(order_idx)
	if EVENT_ID_MODE == 'event_index':
		if int(event_index) < 0:
			raise ValueError(f'event_index must be non-negative, got {event_index}')
		return int(event_index)
	raise ValueError(
		f"EVENT_ID_MODE must be 'sequential' or 'event_index', got {EVENT_ID_MODE}"
	)


def _event_time_window_from_row(
	event_row: pd.Series,
) -> tuple[pd.Timestamp, dt.datetime, dt.datetime, dt.datetime]:
	event_time_utc = pd.Timestamp(event_row['event_time_utc'])
	if event_time_utc.tzinfo is None:
		raise ValueError('internal error: event_time_utc must be timezone-aware')

	origin_time_jst = event_time_utc.tz_convert(_JST).to_pydatetime()
	origin_time_jst_naive = origin_time_jst.replace(tzinfo=None)
	t_start_jst, t_end_jst = compute_event_time_window(
		origin_time_jst_naive,
		pre_sec=int(PRE_SEC),
		post_sec=int(POST_SEC),
	)
	return event_time_utc, origin_time_jst, t_start_jst, t_end_jst


def _phase_count(ev_picks: pd.DataFrame, phase_type: str) -> int:
	return int(ev_picks['phase_type'].astype(str).str.upper().eq(phase_type).sum())


def _build_extra_gamma(
	*,
	event_index: int,
	event_row: pd.Series,
	qc_row: pd.Series,
	latlon_row: pd.Series,
	event_time_utc: pd.Timestamp,
) -> dict[str, object]:
	return {
		'gamma': {
			'event_index': int(event_index),
			'origin_time_utc': event_time_utc.isoformat().replace('+00:00', 'Z'),
			'latitude_deg': float(latlon_row['latitude_deg']),
			'longitude_deg': float(latlon_row['longitude_deg']),
			'depth_km': float(latlon_row['depth_km']),
			'x_km': float(event_row['x(km)']),
			'y_km': float(event_row['y(km)']),
			'z_km': float(event_row['z(km)']),
			'num_picks': int(qc_row['num_picks']),
			'num_p_picks': int(qc_row['num_p_picks']),
			'num_s_picks': int(qc_row['num_s_picks']),
			'sigma_time': float(qc_row['sigma_time']),
			'gamma_score': float(qc_row['gamma_score']),
			'near_any_boundary_1km': bool(qc_row['near_any_boundary_1km']),
		}
	}


def _validate_settings() -> None:  # noqa: C901
	required_files = [
		GAMMA_EVENTS_CSV,
		GAMMA_PICKS_CSV,
		GAMMA_EVENTS_QC_CSV,
		GAMMA_EVENTS_LATLON_CSV,
		STATIONS47_CSV,
	]
	for path in required_files:
		if not path.is_file():
			raise FileNotFoundError(f'required input file not found: {path}')

	if not CONT_BASE_DIR_BY_NETWORK:
		raise ValueError('CONT_BASE_DIR_BY_NETWORK is empty')
	if not CH47_BASE_DIR_BY_NETWORK:
		raise ValueError('CH47_BASE_DIR_BY_NETWORK is empty')

	for net, cont_dir in CONT_BASE_DIR_BY_NETWORK.items():
		code = str(net).strip().upper()
		if not code:
			raise ValueError('CONT_BASE_DIR_BY_NETWORK contains empty network_code')
		if not Path(cont_dir).is_dir():
			raise FileNotFoundError(
				f'continuous base dir not found for network={code}: {cont_dir}'
			)
		if code not in CH47_BASE_DIR_BY_NETWORK:
			raise ValueError(f'CH47_BASE_DIR_BY_NETWORK missing network={code}')

	for net, ch_dir in CH47_BASE_DIR_BY_NETWORK.items():
		code = str(net).strip().upper()
		if not code:
			raise ValueError('CH47_BASE_DIR_BY_NETWORK contains empty network_code')
		if not Path(ch_dir).is_dir():
			raise FileNotFoundError(
				f'47-station .ch subset dir not found for network={code}: {ch_dir}'
			)


def _validate_out_base_dir_is_safe() -> None:
	if not OUT_BASE_DIR.exists():
		return
	if not OUT_BASE_DIR.is_dir():
		raise NotADirectoryError(
			f'OUT_BASE_DIR exists but is not a directory: {OUT_BASE_DIR}'
		)
	event_dirs = sorted(p for p in OUT_BASE_DIR.iterdir() if p.is_dir())
	if event_dirs:
		raise FileExistsError(
			'OUT_BASE_DIR already contains event directories; remove it explicitly '
			f'before rebuilding: {OUT_BASE_DIR}'
		)


def _prepare_event_indices(
	events: pd.DataFrame,
	picks: pd.DataFrame,
	qc: pd.DataFrame,
	latlon: pd.DataFrame,
) -> tuple[list[int], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	pick_event_indices = sorted(picks['event_index'].astype(int).unique().tolist())
	events_by_index = events.set_index('event_index', drop=False)
	qc_by_index = qc.set_index('event_index', drop=False)
	latlon_by_index = latlon.set_index('event_index', drop=False)

	missing_event = [
		e for e in pick_event_indices if int(e) not in events_by_index.index
	]
	if missing_event:
		raise ValueError(
			'event_index exists in gamma_picks but not in gamma_events: '
			f'{missing_event[:20]}'
		)

	qc_mask = (
		(qc['num_picks'].astype(float) >= float(MIN_NUM_PICKS))
		& (qc['num_p_picks'].astype(float) >= float(MIN_NUM_P_PICKS))
		& (qc['num_s_picks'].astype(float) >= float(MIN_NUM_S_PICKS))
		& (qc['sigma_time'].astype(float) <= float(MAX_SIGMA_TIME))
	)
	if REQUIRE_NOT_NEAR_BOUNDARY_1KM:
		qc_mask &= ~qc['near_any_boundary_1km'].astype(bool)

	candidate_set = {int(e) for e in pick_event_indices}
	candidate_set &= {int(e) for e in events_by_index.index.tolist()}
	candidate_set &= {
		int(e) for e in qc.loc[qc_mask, 'event_index'].astype(int).tolist()
	}
	candidate_set &= {int(e) for e in latlon_by_index.index.tolist()}

	if not candidate_set:
		raise ValueError(
			'created event count would be 0 after QC and input intersection'
		)

	order_df = events_by_index.loc[sorted(candidate_set)].reset_index(drop=True).copy()
	if EVENT_ID_MODE == 'sequential':
		order_df = order_df.sort_values(
			['event_time_utc', 'event_index'], kind='mergesort'
		)
		ordered_event_indices = order_df['event_index'].astype(int).tolist()
	elif EVENT_ID_MODE == 'event_index':
		ordered_event_indices = sorted(candidate_set)
	else:
		raise ValueError(
			f"EVENT_ID_MODE must be 'sequential' or 'event_index', got {EVENT_ID_MODE}"
		)

	if MAX_EVENTS is not None:
		if int(MAX_EVENTS) <= 0:
			raise ValueError(f'MAX_EVENTS must be positive or None, got {MAX_EVENTS}')
		ordered_event_indices = ordered_event_indices[: int(MAX_EVENTS)]

	if not ordered_event_indices:
		raise ValueError('created event count would be 0 after MAX_EVENTS')

	return ordered_event_indices, events_by_index, qc_by_index, latlon_by_index


def _validate_event_stations_in_stations47(
	ev_picks: pd.DataFrame,
	stations47: pd.DataFrame,
	*,
	event_index: int,
) -> None:
	known = set(zip(stations47['network_code'], stations47['station'], strict=False))
	required = set(
		zip(ev_picks['network_code'], ev_picks['station_code'], strict=False)
	)
	missing = sorted(required - known)
	if missing:
		raise ValueError(
			'event stations are not present in stations_47.csv for '
			f'event_index={event_index}: '
			f'{missing[:20]}'
		)


def _plan_win32_groups(
	ev_picks: pd.DataFrame,
	cnt_index: dict[str, list[CntRecord]],
	t_start_jst: dt.datetime,
	t_end_jst: dt.datetime,
) -> tuple[list[dict[str, object]], dict[str, object]]:
	network_codes = sorted(ev_picks['network_code'].drop_duplicates().tolist())
	for net in network_codes:
		if net not in CONT_BASE_DIR_BY_NETWORK:
			raise ValueError(
				f'CONT_BASE_DIR_BY_NETWORK missing network from picks: {net}'
			)
		if net not in CH47_BASE_DIR_BY_NETWORK:
			raise ValueError(
				f'CH47_BASE_DIR_BY_NETWORK missing network from picks: {net}'
			)

	group_plans: list[dict[str, object]] = []
	dropped_stations_3c: dict[str, str] = {}
	dropped_stations_cnt_missing: dict[str, str] = {}
	for net in network_codes:
		p_net = ev_picks[ev_picks['network_code'] == net]
		stations = sorted(p_net['station_code'].drop_duplicates().tolist())
		selected_cnt = _select_cnt_covering_window(
			cnt_index[net],
			network_code=net,
			t_start_jst=t_start_jst,
			t_end_jst=t_end_jst,
		)

		ch_paths = [_ch_path_for_cnt(net, rec.path) for rec in selected_cnt]
		usable_stations, dropped_3c, dropped_cnt_missing = (
			_filter_stations_with_three_components(
				stations=stations,
				cnt_paths=[rec.path for rec in selected_cnt],
				ch_paths=ch_paths,
				network_code=net,
			)
		)
		dropped_stations_3c.update(dropped_3c)
		dropped_stations_cnt_missing.update(dropped_cnt_missing)
		if not usable_stations:
			continue

		group_plans.append(
			{
				'network_code': str(net),
				'stations': usable_stations,
				'cnt_paths': [rec.path for rec in selected_cnt],
				'ch_paths': ch_paths,
			}
		)

	dropped_stations_3c_list = [
		f'{sta}:{reason}' for sta, reason in sorted(dropped_stations_3c.items())
	]
	dropped_stations_cnt_missing_list = [
		f'{sta}:{reason}'
		for sta, reason in sorted(dropped_stations_cnt_missing.items())
	]
	n_station_gamma = int(ev_picks['station_code'].drop_duplicates().shape[0])
	n_station_loki = int(sum(len(plan['stations']) for plan in group_plans))
	n_network_loki = int(len(group_plans))
	loki_input = {
		'n_station_gamma': n_station_gamma,
		'n_station_loki': n_station_loki,
		'n_network_loki': n_network_loki,
		'dropped_station_count_3c': int(len(dropped_stations_3c_list)),
		'dropped_stations_3c': dropped_stations_3c_list,
		'dropped_station_count_cnt_missing': int(
			len(dropped_stations_cnt_missing_list)
		),
		'dropped_stations_cnt_missing': dropped_stations_cnt_missing_list,
	}
	return group_plans, loki_input


def _materialize_win32_groups(
	event_dir: Path,
	group_plans: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[str]]:
	win32_groups: list[dict[str, object]] = []
	network_codes: list[str] = []
	for plan in group_plans:
		net = str(plan['network_code'])
		stations = list(plan['stations'])
		cnt_paths = [Path(p) for p in plan['cnt_paths']]
		ch_paths = [Path(p) for p in plan['ch_paths']]
		network_codes.append(net)

		cnt_names: list[str] = []
		for cnt_path in cnt_paths:
			dst = event_dir / cnt_path.name
			_materialize_file(cnt_path, dst, use_symlink=bool(USE_SYMLINK))
			cnt_names.append(dst.name)

		ch_names: list[str] = []
		for ch_src in ch_paths:
			ch_dst = event_dir / ch_src.name
			_materialize_file(ch_src, ch_dst, use_symlink=bool(USE_SYMLINK))
			ch_names.append(ch_dst.name)

		win32_groups.append(
			{
				'network_code': str(net),
				'select_used': True,
				'stations': stations,
				'cnt_files': cnt_names,
				'ch_file': ch_names[0],
				'ch_files': ch_names,
			}
		)

	if not win32_groups:
		raise ValueError('no win32_groups created')

	return win32_groups, network_codes


def _build_one_event(  # noqa: PLR0913
	order_idx: int,
	event_index: int,
	events_by_index: pd.DataFrame,
	qc_by_index: pd.DataFrame,
	latlon_by_index: pd.DataFrame,
	picks: pd.DataFrame,
	stations47: pd.DataFrame,
	cnt_index: dict[str, list[CntRecord]],
) -> EventBuildResult | EventDropResult:
	ev_picks = picks[picks['event_index'] == int(event_index)].copy()
	if ev_picks.empty:
		raise ValueError(f'internal error: no picks for event_index={event_index}')
	_validate_event_stations_in_stations47(
		ev_picks, stations47, event_index=int(event_index)
	)

	event_row = events_by_index.loc[int(event_index)]
	qc_row = qc_by_index.loc[int(event_index)]
	latlon_row = latlon_by_index.loc[int(event_index)]
	event_time_utc, origin_time_jst, t_start_jst, t_end_jst = (
		_event_time_window_from_row(event_row)
	)

	group_plans, loki_input = _plan_win32_groups(
		ev_picks,
		cnt_index,
		t_start_jst,
		t_end_jst,
	)
	if (
		int(loki_input['n_station_loki']) < int(MIN_LOKI_STATIONS_PER_EVENT)
		or int(loki_input['n_network_loki']) < int(MIN_LOKI_NETWORKS_PER_EVENT)
	):
		dropped_stations_3c = list(loki_input['dropped_stations_3c'])
		dropped_stations_cnt_missing = list(
			loki_input['dropped_stations_cnt_missing']
		)
		return EventDropResult(
			summary_row={
				'event_index': int(event_index),
				'origin_time_utc': event_time_utc.isoformat().replace('+00:00', 'Z'),
				'reason': 'too_few_usable_stations',
				'n_station_gamma': int(loki_input['n_station_gamma']),
				'n_station_loki': int(loki_input['n_station_loki']),
				'n_network_loki': int(loki_input['n_network_loki']),
				'dropped_station_count_3c': int(
					loki_input['dropped_station_count_3c']
				),
				'dropped_stations_3c': ';'.join(dropped_stations_3c),
				'dropped_station_count_cnt_missing': int(
					loki_input['dropped_station_count_cnt_missing']
				),
				'dropped_stations_cnt_missing': ';'.join(
					dropped_stations_cnt_missing
				),
			}
		)

	event_id = _resolve_event_id(order_idx, int(event_index))
	event_dir = OUT_BASE_DIR / f'{event_id:06d}'
	event_dir.mkdir(parents=False, exist_ok=False)

	win32_groups, network_codes = _materialize_win32_groups(
		event_dir,
		group_plans,
	)
	extra = _build_extra_gamma(
		event_index=int(event_index),
		event_row=event_row,
		qc_row=qc_row,
		latlon_row=latlon_row,
		event_time_utc=event_time_utc,
	)
	extra['loki_input'] = loki_input

	write_event_json_win32_groups(
		event_dir=event_dir,
		event_id=int(event_id),
		origin_time_jst=origin_time_jst,
		pre_sec=int(PRE_SEC),
		post_sec=int(POST_SEC),
		span_min=int(SPAN_MIN_DEFAULT),
		threads=int(THREADS),
		win32_groups=win32_groups,
		extra=extra,
	)

	summary_row = {
		'event_id': int(event_id),
		'event_index': int(event_index),
		'origin_time_utc': event_time_utc.isoformat().replace('+00:00', 'Z'),
		'origin_time_jst': origin_time_jst.isoformat(),
		'window_start_jst': t_start_jst.isoformat(),
		'window_end_jst': t_end_jst.isoformat(),
		'n_picks': int(ev_picks.shape[0]),
		'n_p_picks': _phase_count(ev_picks, 'P'),
		'n_s_picks': _phase_count(ev_picks, 'S'),
		'n_station': int(loki_input['n_station_loki']),
		'n_network': len(network_codes),
		'n_station_gamma': int(loki_input['n_station_gamma']),
		'n_station_loki': int(loki_input['n_station_loki']),
		'n_network_loki': int(loki_input['n_network_loki']),
		'dropped_station_count_3c': int(loki_input['dropped_station_count_3c']),
		'dropped_stations_3c': ';'.join(list(loki_input['dropped_stations_3c'])),
		'dropped_station_count_cnt_missing': int(
			loki_input['dropped_station_count_cnt_missing']
		),
		'dropped_stations_cnt_missing': ';'.join(
			list(loki_input['dropped_stations_cnt_missing'])
		),
		'networks': '|'.join(network_codes),
		'latitude_deg': float(latlon_row['latitude_deg']),
		'longitude_deg': float(latlon_row['longitude_deg']),
		'depth_km': float(latlon_row['depth_km']),
		'sigma_time': float(qc_row['sigma_time']),
		'gamma_score': float(qc_row['gamma_score']),
		'near_any_boundary_1km': bool(qc_row['near_any_boundary_1km']),
		'event_dir': _repo_rel(event_dir),
	}
	return EventBuildResult(event_dir=event_dir, summary_row=summary_row)


def _write_event_index_map(summary_rows: list[dict[str, object]]) -> Path:
	map_df = pd.DataFrame(summary_rows).sort_values('event_id').reset_index(drop=True)
	map_csv = OUT_BASE_DIR / '_event_index_map.csv'
	map_df.to_csv(map_csv, index=False)
	return map_csv


def _write_dropped_events(drop_rows: list[dict[str, object]]) -> Path:
	drop_df = pd.DataFrame(drop_rows, columns=_DROPPED_EVENTS_COLUMNS)
	drop_csv = OUT_BASE_DIR / '_dropped_events.csv'
	drop_df.to_csv(drop_csv, index=False)
	return drop_csv


def _write_build_config(event_count: int) -> Path:
	generated_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
	config: dict[str, Any] = {
		'generated_at_utc': generated_at.isoformat().replace('+00:00', 'Z'),
		'inputs': {
			'gamma_events_csv': _repo_rel(GAMMA_EVENTS_CSV),
			'gamma_picks_csv': _repo_rel(GAMMA_PICKS_CSV),
			'gamma_events_qc_csv': _repo_rel(GAMMA_EVENTS_QC_CSV),
			'gamma_events_latlon_csv': _repo_rel(GAMMA_EVENTS_LATLON_CSV),
			'stations47_csv': _repo_rel(STATIONS47_CSV),
		},
		'output': {
			'out_base_dir': _repo_rel(OUT_BASE_DIR),
			'event_count': int(event_count),
		},
		'window': {
			'pre_sec': int(PRE_SEC),
			'post_sec': int(POST_SEC),
			'span_min': int(SPAN_MIN_DEFAULT),
		},
		'filters': {
			'min_num_picks': int(MIN_NUM_PICKS),
			'min_num_p_picks': int(MIN_NUM_P_PICKS),
			'min_num_s_picks': int(MIN_NUM_S_PICKS),
			'max_sigma_time': float(MAX_SIGMA_TIME),
			'require_not_near_boundary_1km': bool(REQUIRE_NOT_NEAR_BOUNDARY_1KM),
			'max_events': None if MAX_EVENTS is None else int(MAX_EVENTS),
			'min_loki_stations_per_event': int(MIN_LOKI_STATIONS_PER_EVENT),
			'min_loki_networks_per_event': int(MIN_LOKI_NETWORKS_PER_EVENT),
			'three_component_station_filter_enabled': True,
			'cnt_channel_presence_filter_enabled': True,
		},
	}
	config_json = OUT_BASE_DIR / '_build_config.json'
	write_json(config_json, config, ensure_ascii=False, indent=2)
	return config_json


def _print_first_event_qc(created_event_dirs: list[Path]) -> None:
	first = created_event_dirs[0]
	ev = load_event_json(first)

	win = ev.get('win32')
	if not isinstance(win, dict):
		raise TypeError(f'event.json has invalid win32 block: {first / "event.json"}')
	groups = win.get('groups')
	if not isinstance(groups, list):
		raise TypeError(f'event.json has invalid groups block: {first / "event.json"}')

	n_group = len(groups)
	n_cnt = int(sum(len(g['cnt_files']) for g in groups))
	n_sta = int(sum(len(g['stations']) for g in groups))
	print(
		f'[QC] first_event={first.name} groups={n_group} '
		f'cnt_files={n_cnt} stations={n_sta}'
	)


def main() -> None:
	"""Build Izu2009 event_dir groups from gamma_events/gamma_picks."""
	cont_map = _normalize_network_mapping_keys(
		CONT_BASE_DIR_BY_NETWORK,
		label='CONT_BASE_DIR_BY_NETWORK',
	)
	ch47_map = _normalize_network_mapping_keys(
		CH47_BASE_DIR_BY_NETWORK,
		label='CH47_BASE_DIR_BY_NETWORK',
	)
	CONT_BASE_DIR_BY_NETWORK.clear()
	CONT_BASE_DIR_BY_NETWORK.update(cont_map)
	CH47_BASE_DIR_BY_NETWORK.clear()
	CH47_BASE_DIR_BY_NETWORK.update(ch47_map)

	_validate_settings()
	_validate_out_base_dir_is_safe()

	events = _load_gamma_events(GAMMA_EVENTS_CSV)
	picks = _load_gamma_picks(GAMMA_PICKS_CSV)
	qc = _load_gamma_qc(GAMMA_EVENTS_QC_CSV)
	latlon = _load_gamma_latlon(GAMMA_EVENTS_LATLON_CSV)
	stations47 = _load_stations47(STATIONS47_CSV)

	pick_event_indices, events_by_index, qc_by_index, latlon_by_index = (
		_prepare_event_indices(events, picks, qc, latlon)
	)
	cnt_index = _build_cnt_index(CONT_BASE_DIR_BY_NETWORK)

	OUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

	results: list[EventBuildResult] = []
	drop_rows: list[dict[str, object]] = []
	for event_index in pick_event_indices:
		result = _build_one_event(
			len(results),
			int(event_index),
			events_by_index,
			qc_by_index,
			latlon_by_index,
			picks,
			stations47,
			cnt_index,
		)
		if isinstance(result, EventDropResult):
			drop_rows.append(result.summary_row)
			continue
		results.append(result)

	drop_csv = _write_dropped_events(drop_rows)
	if not results:
		raise ValueError(f'no event directories created; wrote dropped events: {drop_csv}')

	summary_rows = [r.summary_row for r in results]
	created_event_dirs = [r.event_dir for r in results]
	map_csv = _write_event_index_map(summary_rows)
	config_json = _write_build_config(len(created_event_dirs))

	print(f'wrote event_dirs: {len(created_event_dirs)} under {OUT_BASE_DIR}')
	print(f'dropped events: {len(drop_rows)}')
	print(f'wrote: {map_csv}')
	print(f'wrote: {drop_csv}')
	print(f'wrote: {config_json}')
	_print_first_event_qc(created_event_dirs)


if __name__ == '__main__':
	main()

# 実行例:
# python proc/izu2009/association/qc_gamma_izu2009.py
# python proc/izu2009/prepare_data/make_subset_ch_47.py
# python proc/izu2009/loki/build_event_dirs_from_gamma_izu2009.py
