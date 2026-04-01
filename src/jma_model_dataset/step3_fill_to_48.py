from __future__ import annotations

import datetime as dt
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from common.csv_util import open_dict_writer
from common.done_marker import read_done_json, should_skip_done, write_done_json
from common.geo import haversine_distance_km
from common.time_util import ceil_minutes, floor_minute
from jma.download import _name_stem, download_win_for_stations
from jma.prepare.event_paths import resolve_evt_and_ch, resolve_txt_for_evt
from jma.prepare.event_txt import read_event_txt_meta
from jma.station_reader import read_hinet_channel_table
from jma.stationcode_common import normalize_code, normalize_network_code
from jma.stationcode_presence import PresenceDB
from jma.win32_reader import get_evt_info, scan_channel_sampling_rate_map_win32
from jma_model_dataset.paths import (
	active_ch_path,
	continuous_dir,
	fill_to_48_done_path,
	logs_dir,
	missing_txt_path,
	raw_root,
)

__all__ = [
	'DEFAULT_CONT_SCAN_SECOND_BLOCKS',
	'DEFAULT_FILL_NETWORK_CODE',
	'DEFAULT_MAX_SPAN_MIN',
	'DEFAULT_MIN_STATIONS',
	'FillTo48Paths',
	'FillTo48Result',
	'download_fill_to_48_for_event',
	'load_fill_to_48_station_geo',
	'resolve_fill_to_48_paths',
]

DEFAULT_MIN_STATIONS = 48
DEFAULT_FILL_NETWORK_CODE = '0101'
DEFAULT_MAX_SPAN_MIN = 3
DEFAULT_CONT_SCAN_SECOND_BLOCKS = 3

_LOG_FIELDS = [
	'event_dir',
	'evt_file',
	'missing_file',
	'event_month',
	'event_lat',
	'event_lon',
	't0_jst',
	'span_min',
	'n_active',
	'n_continuous',
	'n_before',
	'n_need',
	'n_selected',
	'n_after',
	'network_code',
	'select_used',
	'status',
	'cnt_file',
	'ch_file',
	'message',
]


@dataclass(frozen=True)
class FillTo48Paths:
	event_dir: Path
	raw_dir: Path
	evt_path: Path
	txt_path: Path
	ch_path: Path
	active_path: Path
	missing_path: Path
	outdir: Path
	log_path: Path
	done_path: Path


@dataclass(frozen=True)
class FillTo48Result:
	event_dir: Path
	evt_path: Path
	txt_path: Path
	active_path: Path
	missing_path: Path
	outdir: Path
	log_path: Path
	done_path: Path
	run_tag: str
	status: str
	t0: dt.datetime | None
	span_min: int | None
	n_active: int | None
	n_continuous_before: int | None
	n_before: int | None
	n_selected: int | None
	n_after: int | None
	cnt_path: Path | None
	ch_path: Path | None


def _fill_to_48_log_path(event_dir: Path, stem: str) -> Path:
	return logs_dir(event_dir) / f'{stem}_fill_to_48_log.csv'


def resolve_fill_to_48_paths(event_dir: Path, *, run_tag: str) -> FillTo48Paths:
	event_dir = Path(event_dir).resolve()
	if not event_dir.is_dir():
		raise NotADirectoryError(f'event directory not found: {event_dir}')

	run_tag2 = str(run_tag).strip()
	if run_tag2 == '':
		raise ValueError('run_tag must be non-empty')

	raw_dir = raw_root(event_dir)
	if not raw_dir.is_dir():
		raise NotADirectoryError(f'raw directory not found: {raw_dir}')

	evt_path, ch_path = resolve_evt_and_ch(raw_dir)
	txt_path = resolve_txt_for_evt(evt_path)
	active_path = active_ch_path(event_dir, evt_path.stem)
	if not active_path.is_file():
		raise FileNotFoundError(f'flow active .ch not found: {active_path}')

	return FillTo48Paths(
		event_dir=event_dir,
		raw_dir=raw_dir,
		evt_path=evt_path,
		txt_path=txt_path,
		ch_path=ch_path,
		active_path=active_path,
		missing_path=missing_txt_path(event_dir, evt_path.stem),
		outdir=continuous_dir(event_dir),
		log_path=_fill_to_48_log_path(event_dir, evt_path.stem),
		done_path=fill_to_48_done_path(event_dir, evt_path.stem, run_tag2),
	)


def load_fill_to_48_station_geo(
	channel_table_path: Path,
) -> dict[str, tuple[float, float]]:
	df = read_hinet_channel_table(channel_table_path)
	if df.empty:
		raise ValueError(f'empty channel table: {channel_table_path}')

	df2 = (
		df[['station', 'lat', 'lon']]
		.groupby('station', as_index=False)
		.agg({'lat': 'first', 'lon': 'first'})
	)

	out: dict[str, tuple[float, float]] = {}
	for _, row in df2.iterrows():
		station = normalize_code(row['station'])
		if station == '':
			continue
		out[station] = (float(row['lat']), float(row['lon']))

	if not out:
		raise ValueError(f'no station coordinates loaded from {channel_table_path}')
	return out


def _stations_in_channel_table(ch_path: Path) -> set[str]:
	df = read_hinet_channel_table(ch_path)
	stations = df['station'].astype(str).map(normalize_code).tolist()
	return set([station for station in stations if station])


def _stations_present_in_cnt(
	cnt_path: Path,
	ch_path: Path,
	*,
	max_second_blocks: int,
) -> set[str]:
	fs_by_ch = scan_channel_sampling_rate_map_win32(
		cnt_path,
		max_second_blocks=max_second_blocks,
	)
	if not fs_by_ch:
		return set()

	present_ch = set(int(ch_no) for ch_no in fs_by_ch.keys())
	df = read_hinet_channel_table(ch_path)
	df2 = df[df['ch_int'].isin(present_ch)]
	if df2.empty:
		return set()

	stations = df2['station'].astype(str).map(normalize_code).tolist()
	return set([station for station in stations if station])


def _scan_continuous_present_stations(
	cont_dir: Path,
	*,
	max_second_blocks: int,
) -> set[str]:
	if not cont_dir.is_dir():
		return set()

	out: set[str] = set()
	for cnt_path in sorted(cont_dir.glob('*.cnt')):
		ch_path = cnt_path.with_suffix('.ch')
		if not ch_path.is_file():
			raise FileNotFoundError(f'missing .ch for .cnt: {cnt_path} -> {ch_path}')
		out.update(
			_stations_present_in_cnt(
				cnt_path,
				ch_path,
				max_second_blocks=max_second_blocks,
			)
		)
	return out


def _presence_station_candidates_0101(
	pdb: PresenceDB,
	*,
	event_month: str,
) -> list[str]:
	if event_month not in pdb.month_cols:
		raise ValueError(f'event_month={event_month} not in presence columns')

	net = normalize_network_code(DEFAULT_FILL_NETWORK_CODE)
	hit = pdb.pres[
		(pdb.pres['network_code'] == net) & (pdb.pres[event_month].fillna(0) == 1)
	]
	if hit.empty:
		return []

	cands = hit['ch_key'].astype(str).map(normalize_code).tolist()
	return sorted(set([cand for cand in cands if cand]))


def _select_nearest_0101(
	*,
	event_lat: float,
	event_lon: float,
	candidate_stations: list[str],
	station_geo_0101: dict[str, tuple[float, float]],
	n_need: int,
) -> tuple[list[str], list[tuple[str, float]]]:
	if n_need <= 0:
		return [], []

	cands = sorted(set([normalize_code(sta) for sta in candidate_stations]))
	cands = [sta for sta in cands if sta in station_geo_0101]
	if not cands:
		return [], []

	lat_arr = np.asarray([station_geo_0101[sta][0] for sta in cands], dtype=float)
	lon_arr = np.asarray([station_geo_0101[sta][1] for sta in cands], dtype=float)
	dist_km = haversine_distance_km(
		lat0_deg=float(event_lat),
		lon0_deg=float(event_lon),
		lat_deg=lat_arr,
		lon_deg=lon_arr,
	)

	order = np.argsort(dist_km)
	k = min(int(n_need), len(cands))
	selected = [cands[int(idx)] for idx in order[:k]]
	selected_with_dist = [
		(cands[int(idx)], float(dist_km[int(idx)])) for idx in order[:k]
	]
	return selected, selected_with_dist


def _event_window(evt_path: Path, *, max_span_min: int) -> tuple[dt.datetime, int]:
	evt_info = get_evt_info(evt_path, scan_rate_blocks=1)
	t0 = floor_minute(evt_info.start_time)
	span_min = min(
		ceil_minutes((evt_info.end_time_exclusive - t0).total_seconds()),
		int(max_span_min),
	)
	return t0, span_min


def _build_log_row(
	*,
	paths: FillTo48Paths,
	event_month: str,
	event_lat: float,
	event_lon: float,
	t0: dt.datetime,
	span_min: int,
	n_active: int,
	n_continuous: int,
	n_before: int,
	n_need: int,
	n_selected: int,
	n_after: int,
	status: str,
	cnt_file: str,
	ch_file: str,
	message: str,
) -> dict[str, Any]:
	return {
		'event_dir': str(paths.event_dir),
		'evt_file': paths.evt_path.name,
		'missing_file': paths.missing_path.name if paths.missing_path.is_file() else '',
		'event_month': str(event_month),
		'event_lat': float(event_lat),
		'event_lon': float(event_lon),
		't0_jst': f'{t0:%Y-%m-%d %H:%M:%S}',
		'span_min': int(span_min),
		'n_active': int(n_active),
		'n_continuous': int(n_continuous),
		'n_before': int(n_before),
		'n_need': int(n_need),
		'n_selected': int(n_selected),
		'n_after': int(n_after),
		'network_code': DEFAULT_FILL_NETWORK_CODE,
		'select_used': True,
		'status': str(status),
		'cnt_file': str(cnt_file),
		'ch_file': str(ch_file),
		'message': str(message),
	}


def _write_log_row(log_path: Path, row: dict[str, Any]) -> None:
	log_f, writer = open_dict_writer(log_path, fieldnames=_LOG_FIELDS)
	with closing(log_f):
		writer.writerow(row)


def _write_done(
	done_path: Path,
	*,
	paths: FillTo48Paths,
	run_tag: str,
	event_month: str,
	t0: dt.datetime,
	span_min: int,
	n_active: int,
	n_continuous: int,
	n_before: int,
	n_need: int,
	n_selected: int,
	n_after: int,
	status: str,
	cnt_file: str,
	ch_file: str,
	selected_stations: list[str],
	selected_with_dist: list[tuple[str, float]],
	message: str,
) -> None:
	write_done_json(
		done_path,
		{
			'evt_file': paths.evt_path.name,
			'evt_stem': paths.evt_path.stem,
			'run_tag': str(run_tag),
			'status': str(status),
			'event_month': str(event_month),
			'network_code': DEFAULT_FILL_NETWORK_CODE,
			't0_jst': f'{t0:%Y-%m-%d %H:%M:%S}',
			'span_min': int(span_min),
			'n_active': int(n_active),
			'n_continuous': int(n_continuous),
			'n_before': int(n_before),
			'n_need': int(n_need),
			'n_selected': int(n_selected),
			'n_after': int(n_after),
			'cnt_file': str(cnt_file),
			'ch_file': str(ch_file),
			'selected_stations': list(selected_stations),
			'selected_station_distances_km': [
				{'station': sta, 'distance_km': dist_km}
				for sta, dist_km in selected_with_dist
			],
			'message': str(message),
		},
	)


def download_fill_to_48_for_event(
	event_dir: Path,
	client: Any,
	*,
	pdb: PresenceDB,
	station_geo_0101: dict[str, tuple[float, float]],
	run_tag: str = 'v1',
	threads: int = 8,
	cleanup: bool = True,
	skip_if_exists: bool = True,
	skip_if_done: bool = True,
	min_stations: int = DEFAULT_MIN_STATIONS,
	max_span_min: int = DEFAULT_MAX_SPAN_MIN,
	cont_scan_second_blocks: int = DEFAULT_CONT_SCAN_SECOND_BLOCKS,
) -> FillTo48Result:
	run_tag2 = str(run_tag).strip()
	if run_tag2 == '':
		raise ValueError('run_tag must be non-empty')

	threads2 = int(threads)
	if threads2 <= 0:
		raise ValueError('threads must be >= 1')

	min_stations2 = int(min_stations)
	if min_stations2 <= 0:
		raise ValueError('min_stations must be >= 1')

	max_span_min2 = int(max_span_min)
	if max_span_min2 <= 0:
		raise ValueError('max_span_min must be >= 1')

	cont_scan_second_blocks2 = int(cont_scan_second_blocks)
	if cont_scan_second_blocks2 <= 0:
		raise ValueError('cont_scan_second_blocks must be >= 1')

	paths = resolve_fill_to_48_paths(event_dir, run_tag=run_tag2)
	done_obj = read_done_json(paths.done_path, on_missing='empty', on_error='empty')
	if skip_if_done and should_skip_done(done_obj, run_tag=run_tag2, ok_statuses=None):
		return FillTo48Result(
			event_dir=paths.event_dir,
			evt_path=paths.evt_path,
			txt_path=paths.txt_path,
			active_path=paths.active_path,
			missing_path=paths.missing_path,
			outdir=paths.outdir,
			log_path=paths.log_path,
			done_path=paths.done_path,
			run_tag=run_tag2,
			status='skipped_done',
			t0=None,
			span_min=None,
			n_active=None,
			n_continuous_before=None,
			n_before=None,
			n_selected=None,
			n_after=None,
			cnt_path=None,
			ch_path=None,
		)

	meta = read_event_txt_meta(paths.txt_path)
	t0, span_min = _event_window(paths.evt_path, max_span_min=max_span_min2)
	active_stations = _stations_in_channel_table(paths.active_path)
	continuous_stations = _scan_continuous_present_stations(
		paths.outdir,
		max_second_blocks=cont_scan_second_blocks2,
	)
	existing_stations = set(active_stations) | set(continuous_stations)
	n_active = len(active_stations)
	n_continuous_before = len(continuous_stations)
	n_before = len(existing_stations)
	n_need = max(0, min_stations2 - n_before)

	if n_need == 0:
		status = 'already_satisfied'
		message = f'existing_stations={n_before}'
		_write_log_row(
			paths.log_path,
			_build_log_row(
				paths=paths,
				event_month=meta.event_month,
				event_lat=meta.lat,
				event_lon=meta.lon,
				t0=t0,
				span_min=span_min,
				n_active=n_active,
				n_continuous=n_continuous_before,
				n_before=n_before,
				n_need=0,
				n_selected=0,
				n_after=n_before,
				status=status,
				cnt_file='',
				ch_file='',
				message=message,
			),
		)
		_write_done(
			paths.done_path,
			paths=paths,
			run_tag=run_tag2,
			event_month=meta.event_month,
			t0=t0,
			span_min=span_min,
			n_active=n_active,
			n_continuous=n_continuous_before,
			n_before=n_before,
			n_need=0,
			n_selected=0,
			n_after=n_before,
			status=status,
			cnt_file='',
			ch_file='',
			selected_stations=[],
			selected_with_dist=[],
			message=message,
		)
		return FillTo48Result(
			event_dir=paths.event_dir,
			evt_path=paths.evt_path,
			txt_path=paths.txt_path,
			active_path=paths.active_path,
			missing_path=paths.missing_path,
			outdir=paths.outdir,
			log_path=paths.log_path,
			done_path=paths.done_path,
			run_tag=run_tag2,
			status=status,
			t0=t0,
			span_min=span_min,
			n_active=n_active,
			n_continuous_before=n_continuous_before,
			n_before=n_before,
			n_selected=0,
			n_after=n_before,
			cnt_path=None,
			ch_path=None,
		)

	candidate_stations = _presence_station_candidates_0101(
		pdb,
		event_month=meta.event_month,
	)
	candidate_stations = [
		station for station in candidate_stations if station not in existing_stations
	]
	selected, selected_with_dist = _select_nearest_0101(
		event_lat=meta.lat,
		event_lon=meta.lon,
		candidate_stations=candidate_stations,
		station_geo_0101=station_geo_0101,
		n_need=n_need,
	)

	if not selected:
		status = 'shortage'
		message = 'no 0101 candidates remain after excluding existing flow stations'
		_write_log_row(
			paths.log_path,
			_build_log_row(
				paths=paths,
				event_month=meta.event_month,
				event_lat=meta.lat,
				event_lon=meta.lon,
				t0=t0,
				span_min=span_min,
				n_active=n_active,
				n_continuous=n_continuous_before,
				n_before=n_before,
				n_need=n_need,
				n_selected=0,
				n_after=n_before,
				status=status,
				cnt_file='',
				ch_file='',
				message=message,
			),
		)
		_write_done(
			paths.done_path,
			paths=paths,
			run_tag=run_tag2,
			event_month=meta.event_month,
			t0=t0,
			span_min=span_min,
			n_active=n_active,
			n_continuous=n_continuous_before,
			n_before=n_before,
			n_need=n_need,
			n_selected=0,
			n_after=n_before,
			status=status,
			cnt_file='',
			ch_file='',
			selected_stations=[],
			selected_with_dist=[],
			message=message,
		)
		return FillTo48Result(
			event_dir=paths.event_dir,
			evt_path=paths.evt_path,
			txt_path=paths.txt_path,
			active_path=paths.active_path,
			missing_path=paths.missing_path,
			outdir=paths.outdir,
			log_path=paths.log_path,
			done_path=paths.done_path,
			run_tag=run_tag2,
			status=status,
			t0=t0,
			span_min=span_min,
			n_active=n_active,
			n_continuous_before=n_continuous_before,
			n_before=n_before,
			n_selected=0,
			n_after=n_before,
			cnt_path=None,
			ch_path=None,
		)

	stem = _name_stem(DEFAULT_FILL_NETWORK_CODE, t0, sorted(selected), span_min)
	cnt_path = paths.outdir / f'{stem}.cnt'
	ch_path = paths.outdir / f'{stem}.ch'

	if skip_if_exists and cnt_path.is_file() and ch_path.is_file():
		continuous_stations_after = _scan_continuous_present_stations(
			paths.outdir,
			max_second_blocks=cont_scan_second_blocks2,
		)
		n_after = len(set(active_stations) | set(continuous_stations_after))
		status = 'exists' if n_after >= min_stations2 else 'partial'
		message = f'reused existing output for {len(selected)} selected station(s)'
		_write_log_row(
			paths.log_path,
			_build_log_row(
				paths=paths,
				event_month=meta.event_month,
				event_lat=meta.lat,
				event_lon=meta.lon,
				t0=t0,
				span_min=span_min,
				n_active=n_active,
				n_continuous=n_continuous_before,
				n_before=n_before,
				n_need=n_need,
				n_selected=len(selected),
				n_after=n_after,
				status=status,
				cnt_file=cnt_path.name,
				ch_file=ch_path.name,
				message=message,
			),
		)
		_write_done(
			paths.done_path,
			paths=paths,
			run_tag=run_tag2,
			event_month=meta.event_month,
			t0=t0,
			span_min=span_min,
			n_active=n_active,
			n_continuous=n_continuous_before,
			n_before=n_before,
			n_need=n_need,
			n_selected=len(selected),
			n_after=n_after,
			status=status,
			cnt_file=cnt_path.name,
			ch_file=ch_path.name,
			selected_stations=selected,
			selected_with_dist=selected_with_dist,
			message=message,
		)
		return FillTo48Result(
			event_dir=paths.event_dir,
			evt_path=paths.evt_path,
			txt_path=paths.txt_path,
			active_path=paths.active_path,
			missing_path=paths.missing_path,
			outdir=paths.outdir,
			log_path=paths.log_path,
			done_path=paths.done_path,
			run_tag=run_tag2,
			status=status,
			t0=t0,
			span_min=span_min,
			n_active=n_active,
			n_continuous_before=n_continuous_before,
			n_before=n_before,
			n_selected=len(selected),
			n_after=n_after,
			cnt_path=cnt_path,
			ch_path=ch_path,
		)

	cnt_out, ch_out, _select_used = download_win_for_stations(
		client,
		stations=selected,
		when=t0,
		network_code=DEFAULT_FILL_NETWORK_CODE,
		span_min=span_min,
		outdir=paths.outdir,
		threads=threads2,
		cleanup=cleanup,
		clear_selection=True,
		skip_if_exists=False,
		use_select=True,
		data_name=cnt_path.name,
		ctable_name=ch_path.name,
	)
	continuous_stations_after = _scan_continuous_present_stations(
		paths.outdir,
		max_second_blocks=cont_scan_second_blocks2,
	)
	n_after = len(set(active_stations) | set(continuous_stations_after))
	status = 'downloaded' if n_after >= min_stations2 else 'partial'
	message = f'downloaded {len(selected)} selected 0101 station(s)'
	_write_log_row(
		paths.log_path,
		_build_log_row(
			paths=paths,
			event_month=meta.event_month,
			event_lat=meta.lat,
			event_lon=meta.lon,
			t0=t0,
			span_min=span_min,
			n_active=n_active,
			n_continuous=n_continuous_before,
			n_before=n_before,
			n_need=n_need,
			n_selected=len(selected),
			n_after=n_after,
			status=status,
			cnt_file=Path(cnt_out).name,
			ch_file=Path(ch_out).name,
			message=message,
		),
	)
	_write_done(
		paths.done_path,
		paths=paths,
		run_tag=run_tag2,
		event_month=meta.event_month,
		t0=t0,
		span_min=span_min,
		n_active=n_active,
		n_continuous=n_continuous_before,
		n_before=n_before,
		n_need=n_need,
		n_selected=len(selected),
		n_after=n_after,
		status=status,
		cnt_file=Path(cnt_out).name,
		ch_file=Path(ch_out).name,
		selected_stations=selected,
		selected_with_dist=selected_with_dist,
		message=message,
	)
	return FillTo48Result(
		event_dir=paths.event_dir,
		evt_path=paths.evt_path,
		txt_path=paths.txt_path,
		active_path=paths.active_path,
		missing_path=paths.missing_path,
		outdir=paths.outdir,
		log_path=paths.log_path,
		done_path=paths.done_path,
		run_tag=run_tag2,
		status=status,
		t0=t0,
		span_min=span_min,
		n_active=n_active,
		n_continuous_before=n_continuous_before,
		n_before=n_before,
		n_selected=len(selected),
		n_after=n_after,
		cnt_path=Path(cnt_out),
		ch_path=Path(ch_out),
	)
