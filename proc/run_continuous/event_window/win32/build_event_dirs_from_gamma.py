# %%
"""Build Loki-ready WIN32 event directories from GaMMA outputs."""

# proc/run_continuous/event_window/win32/build_event_dirs_from_gamma.py
#
# Purpose:
# - Convert GaMMA association result CSVs (gamma_events.csv / gamma_picks.csv)
#   into per-event WIN32 directories for Loki input.
# - Each event_dir contains:
#   - event.json (groups format via write_event_json_win32_groups)
#   - required .cnt files
#   - required .ch file(s)

from __future__ import annotations

import datetime as dt
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common.core import (
	load_event_json,
	validate_columns,
	write_event_json_win32_groups,
)
from jma.win32_reader import compute_event_time_window
from pipelines.win32_eqt_continuous_pipelines import parse_win32_cnt_filename

# =========================
# 設定（ここだけ触ればOK）
# =========================
GAMMA_EVENTS_CSV = Path(
	'/workspace/proc/run_continuous/association/win32/out/gamma_events.csv'
)
GAMMA_PICKS_CSV = Path(
	'/workspace/proc/run_continuous/association/win32/out/gamma_picks.csv'
)

# 典型設定例（0301単一ネット）:
# CONT_BASE_DIR_BY_NETWORK = {
# 	'0301': Path('/workspace/data/izu2009/continuous/0301'),
# }
# CH_TABLE_BY_NETWORK = {
# 	'0301': Path(
# 		'/workspace/proc/prepare_data/izu2009/download_continuous/continuous_ch47/0301/'
# 		'win_0301_200912170000_10m_4dd999af.ch'
# 	),
# }
CONT_BASE_DIR_BY_NETWORK: dict[str, Path] = {
	'0101': Path('/workspace/data/izu2009/continuous/0101'),
	'0203': Path('/workspace/data/izu2009/continuous/0203'),
	'0207': Path('/workspace/data/izu2009/continuous/0207'),
	'0301': Path('/workspace/data/izu2009/continuous/0301'),
}
CH_TABLE_BY_NETWORK: dict[str, Path] = {
	'0101': Path(
		'/workspace/proc/prepare_data/izu2009/download_continuous/continuous_ch47/0101/'
		'win_0101_200912170000_10m_aa3c27a4.ch'
	),
	'0203': Path(
		'/workspace/proc/prepare_data/izu2009/download_continuous/continuous_ch47/0203/'
		'win_0203_200912170000_10m_9a3c463f.ch'
	),
	'0207': Path(
		'/workspace/proc/prepare_data/izu2009/download_continuous/continuous_ch47/0207/'
		'win_0207_200912170000_10m_1c7df708.ch'
	),
	'0301': Path(
		'/workspace/proc/prepare_data/izu2009/download_continuous/continuous_ch47/0301/'
		'win_0301_200912170000_10m_4dd999af.ch'
	),
}

OUT_BASE_DIR = Path('/workspace/proc/run_continuous/event_window/win32/out/gamma')

PRE_SEC = 10
POST_SEC = 50
SPAN_MIN_DEFAULT = 10
THREADS = 8
USE_SYMLINK = True

# event_id mode:
# - 'sequential': 000001, 000002, ...
# - 'event_index': event_index をそのまま使う（ゼロ埋め表示のみ）
EVENT_ID_MODE = 'sequential'
EVENT_ID_START = 1
# =========================

_JST = dt.timezone(dt.timedelta(hours=9))
_STATION_ID_SPLIT = '.'
_EVENT_TIME_COL_CANDIDATES = ('time', 'timestamp', 'event_time', 'origin_time')
_EVENT_INDEX_COL_CANDIDATES = ('event_index', 'event_idx', 'idx_eve')


@dataclass(frozen=True)
class CntRecord:
	"""One WIN32 continuous tile."""

	path: Path
	network_code: str
	start_jst: dt.datetime
	end_jst: dt.datetime
	span_min: int


@dataclass(frozen=True)
class EventBuildResult:
	"""One built event directory and its summary row."""

	event_dir: Path
	summary_row: dict[str, object]


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


def _normalize_event_index(events_df: pd.DataFrame) -> pd.DataFrame:
	for col in _EVENT_INDEX_COL_CANDIDATES:
		if col in events_df.columns:
			ev = events_df.copy()
			ev['event_index'] = _to_int_series_strict(ev[col], col)
			break
	else:
		ev = events_df.copy()
		ev['event_index'] = ev.index.astype('int64')

	if ev['event_index'].duplicated().any():
		dup = (
			ev.loc[ev['event_index'].duplicated(), 'event_index']
			.astype(int)
			.tolist()
		)
		raise ValueError(f'duplicated event_index in gamma_events.csv: {dup[:20]}')

	return ev


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

	events = _normalize_event_index(events)
	time_col = _resolve_event_time_column(events)
	events['event_time_utc'] = pd.to_datetime(
		events[time_col], utc=True, errors='raise'
	)
	return events


def _split_station_id(station_id: str) -> tuple[str, str]:
	s = str(station_id).strip()
	if s.count(_STATION_ID_SPLIT) != 1:
		raise ValueError(f'invalid station_id format (need <net>.<station>): {s!r}')

	net, sta = s.split(_STATION_ID_SPLIT, 1)
	net = net.strip().upper()
	sta = sta.strip().upper()
	if not net or not sta:
		raise ValueError(f'invalid station_id format (empty token): {s!r}')
	return net, sta


def _load_gamma_picks(picks_csv: Path) -> pd.DataFrame:
	picks = pd.read_csv(picks_csv)
	validate_columns(
		picks, ['station_id', 'event_index'], f'gamma_picks.csv: {picks_csv}'
	)

	picks = picks.copy()
	picks['event_index'] = _to_int_series_strict(picks['event_index'], 'event_index')
	picks = picks[picks['event_index'] >= 0].reset_index(drop=True)
	if picks.empty:
		raise ValueError('gamma_picks.csv has no assigned picks (event_index >= 0)')

	parts = picks['station_id'].astype(str).map(_split_station_id)
	picks['network_code'] = [t[0] for t in parts]
	picks['station_code'] = [t[1] for t in parts]
	return picks


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
		base_dir = Path(cont_base_dir_by_network[net])
		idx[code] = _build_cnt_index_for_network(code, base_dir)
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
		r
		for r in records
		if (r.end_jst > t_start_jst) and (r.start_jst < t_end_jst)
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


def _build_extra_gamma(
	event_index: int,
	event_row: pd.Series,
	event_time_utc: pd.Timestamp,
	ev_picks: pd.DataFrame,
	networks: list[str],
) -> dict[str, object]:
	extra_gamma: dict[str, object] = {
		'event_index': int(event_index),
		'n_picks': int(ev_picks.shape[0]),
		'n_station': int(ev_picks['station_id'].nunique()),
		'n_network': len(networks),
		'networks': list(networks),
		'origin_time_utc': event_time_utc.isoformat().replace('+00:00', 'Z'),
	}

	for c in ['x(km)', 'y(km)', 'z(km)', 'sigma_time', 'sigma_amp']:
		if c not in event_row.index:
			continue
		v = event_row[c]
		if pd.isna(v):
			continue
		extra_gamma[c] = float(v)

	return {'gamma': extra_gamma}


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


def _validate_settings() -> None:
	if not GAMMA_EVENTS_CSV.is_file():
		raise FileNotFoundError(f'GAMMA_EVENTS_CSV not found: {GAMMA_EVENTS_CSV}')
	if not GAMMA_PICKS_CSV.is_file():
		raise FileNotFoundError(f'GAMMA_PICKS_CSV not found: {GAMMA_PICKS_CSV}')
	if not CONT_BASE_DIR_BY_NETWORK:
		raise ValueError('CONT_BASE_DIR_BY_NETWORK is empty')
	if not CH_TABLE_BY_NETWORK:
		raise ValueError('CH_TABLE_BY_NETWORK is empty')

	for net, ch_path in CH_TABLE_BY_NETWORK.items():
		code = str(net).strip().upper()
		if not code:
			raise ValueError('CH_TABLE_BY_NETWORK contains empty network_code')
		if not Path(ch_path).is_file():
			raise FileNotFoundError(f'.ch not found for network={code}: {ch_path}')
		if code not in CONT_BASE_DIR_BY_NETWORK:
			raise ValueError(f'CONT_BASE_DIR_BY_NETWORK missing network={code}')


def _prepare_event_indices(
	events: pd.DataFrame,
	picks: pd.DataFrame,
) -> tuple[list[int], pd.DataFrame]:
	pick_event_indices = sorted(picks['event_index'].astype(int).unique().tolist())
	events_by_index = events.set_index('event_index', drop=False)

	missing_event = [
		e for e in pick_event_indices if int(e) not in events_by_index.index
	]
	if missing_event:
		raise ValueError(
			'event_index exists in gamma_picks but not in gamma_events: '
			f'{missing_event[:20]}'
		)

	if EVENT_ID_MODE == 'sequential':
		order_df = events_by_index.loc[pick_event_indices]
		order_df = order_df.sort_values(
			['event_time_utc', 'event_index'], kind='mergesort'
		)
		ordered_event_indices = order_df['event_index'].astype(int).tolist()
		return ordered_event_indices, events_by_index

	return pick_event_indices, events_by_index


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


def _build_win32_groups(
	event_dir: Path,
	ev_picks: pd.DataFrame,
	cnt_index: dict[str, list[CntRecord]],
	t_start_jst: dt.datetime,
	t_end_jst: dt.datetime,
) -> tuple[list[dict[str, object]], list[str]]:
	network_codes = sorted(ev_picks['network_code'].drop_duplicates().tolist())
	for net in network_codes:
		if net not in CONT_BASE_DIR_BY_NETWORK:
			raise ValueError(
				f'CONT_BASE_DIR_BY_NETWORK missing network from picks: {net}'
			)
		if net not in CH_TABLE_BY_NETWORK:
			raise ValueError(f'CH_TABLE_BY_NETWORK missing network from picks: {net}')

	win32_groups: list[dict[str, object]] = []
	for net in network_codes:
		p_net = ev_picks[ev_picks['network_code'] == net]
		stations = sorted(p_net['station_code'].drop_duplicates().tolist())
		selected_cnt = _select_cnt_covering_window(
			cnt_index[net],
			network_code=net,
			t_start_jst=t_start_jst,
			t_end_jst=t_end_jst,
		)

		cnt_names: list[str] = []
		for rec in selected_cnt:
			dst = event_dir / rec.path.name
			_materialize_file(rec.path, dst, use_symlink=bool(USE_SYMLINK))
			cnt_names.append(dst.name)

		ch_src = Path(CH_TABLE_BY_NETWORK[net])
		ch_dst = event_dir / ch_src.name
		_materialize_file(ch_src, ch_dst, use_symlink=bool(USE_SYMLINK))

		win32_groups.append(
			{
				'network_code': str(net),
				'select_used': False,
				'stations': stations,
				'cnt_files': cnt_names,
				'ch_file': ch_dst.name,
			}
		)

	if not win32_groups:
		raise ValueError('no win32_groups created')

	return win32_groups, network_codes


def _build_one_event(
	order_idx: int,
	event_index: int,
	events_by_index: pd.DataFrame,
	picks: pd.DataFrame,
	cnt_index: dict[str, list[CntRecord]],
) -> EventBuildResult:
	ev_picks = picks[picks['event_index'] == int(event_index)].copy()
	if ev_picks.empty:
		raise ValueError(f'internal error: no picks for event_index={event_index}')

	event_row = events_by_index.loc[int(event_index)]
	event_time_utc, origin_time_jst, t_start_jst, t_end_jst = (
		_event_time_window_from_row(event_row)
	)

	event_id = _resolve_event_id(order_idx, int(event_index))
	event_dir = OUT_BASE_DIR / f'{event_id:06d}'
	event_dir.mkdir(parents=False, exist_ok=False)

	win32_groups, network_codes = _build_win32_groups(
		event_dir,
		ev_picks,
		cnt_index,
		t_start_jst,
		t_end_jst,
	)
	extra = _build_extra_gamma(
		int(event_index),
		event_row,
		event_time_utc,
		ev_picks,
		network_codes,
	)

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
		'origin_time_jst': origin_time_jst.isoformat(),
		'window_start_jst': t_start_jst.isoformat(),
		'window_end_jst': t_end_jst.isoformat(),
		'n_picks': int(ev_picks.shape[0]),
		'n_station': int(ev_picks['station_id'].nunique()),
		'n_network': len(network_codes),
		'networks': '|'.join(network_codes),
		'event_dir': str(event_dir),
	}
	return EventBuildResult(event_dir=event_dir, summary_row=summary_row)


def _write_event_index_map(summary_rows: list[dict[str, object]]) -> Path:
	map_df = pd.DataFrame(summary_rows).sort_values('event_id').reset_index(drop=True)
	map_csv = OUT_BASE_DIR / '_event_index_map.csv'
	map_df.to_csv(map_csv, index=False)
	return map_csv


def main() -> None:
	"""Build event_dir群 from gamma_events.csv/gamma_picks.csv."""
	cont_map = _normalize_network_mapping_keys(
		CONT_BASE_DIR_BY_NETWORK,
		label='CONT_BASE_DIR_BY_NETWORK',
	)
	ch_map = _normalize_network_mapping_keys(
		CH_TABLE_BY_NETWORK,
		label='CH_TABLE_BY_NETWORK',
	)
	CONT_BASE_DIR_BY_NETWORK.clear()
	CONT_BASE_DIR_BY_NETWORK.update(cont_map)
	CH_TABLE_BY_NETWORK.clear()
	CH_TABLE_BY_NETWORK.update(ch_map)

	_validate_settings()
	events = _load_gamma_events(GAMMA_EVENTS_CSV)
	picks = _load_gamma_picks(GAMMA_PICKS_CSV)
	pick_event_indices, events_by_index = _prepare_event_indices(events, picks)
	cnt_index = _build_cnt_index(CONT_BASE_DIR_BY_NETWORK)

	OUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

	results = [
		_build_one_event(i, int(event_index), events_by_index, picks, cnt_index)
		for i, event_index in enumerate(pick_event_indices)
	]
	if not results:
		raise ValueError('no event directories created')

	summary_rows = [r.summary_row for r in results]
	created_event_dirs = [r.event_dir for r in results]
	map_csv = _write_event_index_map(summary_rows)

	print(f'wrote event_dirs: {len(created_event_dirs)} under {OUT_BASE_DIR}')
	print(f'wrote: {map_csv}')
	_print_first_event_qc(created_event_dirs)


if __name__ == '__main__':
	main()

# 実行例:
# export PYTHONPATH="$PWD/src"
# python proc/run_continuous/event_window/win32/build_event_dirs_from_gamma.py
