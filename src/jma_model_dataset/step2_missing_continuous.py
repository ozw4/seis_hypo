from __future__ import annotations

import datetime as dt
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.csv_util import open_dict_writer
from common.done_marker import read_done_json, should_skip_done, write_done_json
from common.time_util import ceil_minutes, floor_minute
from jma.download import (
	_name_stem,
	_supports_station_selection,
	download_win_for_stations,
)
from jma.prepare.event_paths import resolve_single_evt
from jma.prepare.missing_io import read_missing_by_network
from jma.win32_reader import get_evt_info
from jma_model_dataset.paths import (
	continuous_dir,
	continuous_done_path,
	logs_dir,
	missing_txt_path,
	raw_root,
)

__all__ = [
	'MissingContinuousPaths',
	'EventDownloadResult',
	'resolve_missing_continuous_paths',
	'download_missing_continuous_for_event',
]

_LOG_FIELDS = [
	'event_dir',
	'evt_file',
	't0_jst',
	'span_min',
	'network_code',
	'n_stations_request',
	'select_used',
	'full_download',
	'threads_used',
	'try_idx',
	'status',
	'cnt_file',
	'ch_file',
	'message',
]


@dataclass(frozen=True)
class MissingContinuousPaths:
	event_dir: Path
	raw_dir: Path
	evt_path: Path
	missing_path: Path
	outdir: Path
	log_path: Path


@dataclass(frozen=True)
class EventDownloadResult:
	event_dir: Path
	evt_path: Path
	missing_path: Path
	outdir: Path
	log_path: Path
	run_tag: str
	t0: dt.datetime
	span_min: int
	n_networks_total: int
	n_downloaded: int
	n_exists: int
	n_skipped_done: int


def _continuous_log_path(event_dir: Path, stem: str) -> Path:
	return logs_dir(event_dir) / f'{stem}_continuous_download_log.csv'


def resolve_missing_continuous_paths(event_dir: Path) -> MissingContinuousPaths:
	event_dir = Path(event_dir).resolve()
	if not event_dir.is_dir():
		raise NotADirectoryError(f'event directory not found: {event_dir}')

	raw_dir = raw_root(event_dir)
	if not raw_dir.is_dir():
		raise NotADirectoryError(f'raw directory not found: {raw_dir}')

	evt_path = resolve_single_evt(raw_dir, allow_none=False)
	missing_path = missing_txt_path(event_dir, evt_path.stem)
	if not missing_path.is_file():
		raise FileNotFoundError(f'flow missing target not found: {missing_path}')

	return MissingContinuousPaths(
		event_dir=event_dir,
		raw_dir=raw_dir,
		evt_path=evt_path,
		missing_path=missing_path,
		outdir=continuous_dir(event_dir),
		log_path=_continuous_log_path(event_dir, evt_path.stem),
	)


def _event_window(evt_path: Path) -> tuple[dt.datetime, int]:
	evt_info = get_evt_info(evt_path, scan_rate_blocks=1)
	t0 = floor_minute(evt_info.start_time)
	span_min = min(
		ceil_minutes((evt_info.end_time_exclusive - t0).total_seconds()),
		3,
	)
	return t0, span_min


def _should_skip_net_done(done_path: Path, *, run_tag: str) -> bool:
	obj = read_done_json(done_path, on_missing='empty', on_error='empty')
	return should_skip_done(
		obj,
		run_tag=run_tag,
		ok_statuses={'done', 'exists'},
	)


def _write_net_done(
	done_path: Path,
	*,
	evt_file: str,
	run_tag: str,
	network_code: str,
	status: str,
	cnt_file: str,
	ch_file: str,
	message: str,
	n_stations_request: int,
	threads_used: int,
	try_idx: int,
) -> None:
	write_done_json(
		done_path,
		{
			'evt_file': str(evt_file),
			'run_tag': str(run_tag),
			'network_code': str(network_code),
			'status': str(status),
			'cnt_file': str(cnt_file),
			'ch_file': str(ch_file),
			'message': str(message),
			'n_stations_request': int(n_stations_request),
			'threads_used': int(threads_used),
			'try_idx': int(try_idx),
		},
	)


def _build_log_row(
	*,
	event_dir: Path,
	evt_file: str,
	t0: dt.datetime,
	span_min: int,
	network_code: str,
	n_stations_request: int,
	select_used: bool,
	full_download: bool,
	threads_used: int,
	try_idx: int,
	status: str,
	cnt_file: str,
	ch_file: str,
	message: str,
) -> dict[str, Any]:
	return {
		'event_dir': str(event_dir),
		'evt_file': str(evt_file),
		't0_jst': f'{t0:%Y-%m-%d %H:%M:%S}',
		'span_min': int(span_min),
		'network_code': str(network_code),
		'n_stations_request': int(n_stations_request),
		'select_used': bool(select_used),
		'full_download': bool(full_download),
		'threads_used': int(threads_used),
		'try_idx': int(try_idx),
		'status': str(status),
		'cnt_file': str(cnt_file),
		'ch_file': str(ch_file),
		'message': str(message),
	}


def download_missing_continuous_for_event(
	event_dir: Path,
	client: Any,
	*,
	run_tag: str = 'v1',
	threads: int = 8,
	cleanup: bool = True,
	skip_if_exists: bool = True,
	skip_if_done: bool = True,
) -> EventDownloadResult:
	paths = resolve_missing_continuous_paths(event_dir)

	run_tag2 = str(run_tag).strip()
	if run_tag2 == '':
		raise ValueError('run_tag must be non-empty')

	threads2 = int(threads)
	if threads2 <= 0:
		raise ValueError('threads must be >= 1')

	t0, span_min = _event_window(paths.evt_path)
	stations_by_network = read_missing_by_network(paths.missing_path)
	if not stations_by_network:
		raise ValueError(f'no missing station pairs found in {paths.missing_path}')

	log_f, writer = open_dict_writer(paths.log_path, fieldnames=_LOG_FIELDS)
	n_downloaded = 0
	n_exists = 0
	n_skipped_done = 0

	with closing(log_f):
		for network_code, stations in stations_by_network.items():
			network_code2 = str(network_code)
			done_path = continuous_done_path(
				paths.event_dir,
				paths.evt_path.stem,
				run_tag2,
				network_code2,
			)
			if skip_if_done and _should_skip_net_done(done_path, run_tag=run_tag2):
				n_skipped_done += 1
				print(
					f'[info] skip network (done exists): '
					f'code={network_code2} start={t0} span_min={span_min} '
					f'n_stations={len(stations)}',
					flush=True,
				)
				continue

			select_supported = _supports_station_selection(network_code2)
			full_download = not select_supported
			stations_for_name = stations if select_supported else ['ALL']
			stem = _name_stem(network_code2, t0, stations_for_name, span_min)
			cnt_path = paths.outdir / f'{stem}.cnt'
			ch_path = paths.outdir / f'{stem}.ch'

			if skip_if_exists and cnt_path.is_file() and ch_path.is_file():
				writer.writerow(
					_build_log_row(
						event_dir=paths.event_dir,
						evt_file=paths.evt_path.name,
						t0=t0,
						span_min=span_min,
						network_code=network_code2,
						n_stations_request=len(stations),
						select_used=select_supported,
						full_download=full_download,
						threads_used=0,
						try_idx=0,
						status='exists',
						cnt_file=cnt_path.name,
						ch_file=ch_path.name,
						message='',
					)
				)
				_write_net_done(
					done_path,
					evt_file=paths.evt_path.name,
					run_tag=run_tag2,
					network_code=network_code2,
					status='exists',
					cnt_file=cnt_path.name,
					ch_file=ch_path.name,
					message='',
					n_stations_request=len(stations),
					threads_used=0,
					try_idx=0,
				)
				n_exists += 1
				continue

			cnt_out, ch_out, select_used = download_win_for_stations(
				client,
				stations=stations,
				when=t0,
				network_code=network_code2,
				span_min=span_min,
				outdir=paths.outdir,
				threads=threads2,
				cleanup=cleanup,
				clear_selection=True,
				skip_if_exists=False,
				use_select=select_supported,
				data_name=cnt_path.name,
				ctable_name=ch_path.name,
			)
			writer.writerow(
				_build_log_row(
					event_dir=paths.event_dir,
					evt_file=paths.evt_path.name,
					t0=t0,
					span_min=span_min,
					network_code=network_code2,
					n_stations_request=len(stations),
					select_used=bool(select_used),
					full_download=full_download,
					threads_used=threads2,
					try_idx=1,
					status='downloaded',
					cnt_file=Path(cnt_out).name,
					ch_file=Path(ch_out).name,
					message='',
				)
			)
			_write_net_done(
				done_path,
				evt_file=paths.evt_path.name,
				run_tag=run_tag2,
				network_code=network_code2,
				status='done',
				cnt_file=Path(cnt_out).name,
				ch_file=Path(ch_out).name,
				message='',
				n_stations_request=len(stations),
				threads_used=threads2,
				try_idx=1,
			)
			n_downloaded += 1

	return EventDownloadResult(
		event_dir=paths.event_dir,
		evt_path=paths.evt_path,
		missing_path=paths.missing_path,
		outdir=paths.outdir,
		log_path=paths.log_path,
		run_tag=run_tag2,
		t0=t0,
		span_min=span_min,
		n_networks_total=len(stations_by_network),
		n_downloaded=n_downloaded,
		n_exists=n_exists,
		n_skipped_done=n_skipped_done,
	)
