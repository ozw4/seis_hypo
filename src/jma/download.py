from __future__ import annotations

import datetime as dt
import hashlib
import logging
import os
import time
from collections.abc import Sequence
from contextlib import contextmanager
from netrc import netrc
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from HinetPy import Client

from common.core import write_event_json, write_event_json_win32_groups
from common.time_util import floor_minute, minute_range
from jma.win32_reader import compute_event_time_window


def load_hinet_credentials_from_netrc(machine: str) -> tuple[str, str]:
	"""Load and validate Hi-net credentials from .netrc."""
	auth = netrc().authenticators(machine)
	if auth is None:
		raise ValueError(f'no .netrc entry found for machine={machine!r}')

	login, _, password = auth
	if not login:
		raise ValueError(f'.netrc entry for machine={machine!r} has no login')
	if not password:
		raise ValueError(f'.netrc entry for machine={machine!r} has no password')
	if len(password) > 12:
		raise ValueError(
			'Hi-net password is longer than 12 characters; '
			'set the exact usable password in .netrc'
		)

	return login, password


def create_hinet_client(*, machine: str = 'hinet') -> Client:
	"""Create a Hi-net client using credentials from .netrc."""
	login, password = load_hinet_credentials_from_netrc(machine)
	return Client(
		login, password, timeout=150, sleep_time_in_seconds=10, max_sleep_count=60
	)


@contextmanager
def _pushd(path: str | Path):
	cwd = Path.cwd()
	os.chdir(path)
	try:
		yield
	finally:
		os.chdir(cwd)


def download_event_waveform_directories(
	client: Client,
	starttime: dt.datetime | str,
	endtime: dt.datetime | str,
	outdir: str | Path,
	*,
	region: str = '00',
	minmagnitude: float = 3.0,
	maxmagnitude: float = 9.9,
) -> list[Path]:
	outdir = Path(outdir)
	outdir.mkdir(parents=True, exist_ok=True)

	with TemporaryDirectory(
		dir=outdir,
		prefix='tmp_get_event_waveform_',
	) as tmpdir:
		staging_dir = Path(tmpdir)
		with _pushd(staging_dir):
			client.get_event_waveform(
				starttime,
				endtime,
				region=region,
				minmagnitude=minmagnitude,
				maxmagnitude=maxmagnitude,
			)

		downloaded_paths = sorted(staging_dir.iterdir())
		if not downloaded_paths:
			raise RuntimeError(
				'new event directories were not created by get_event_waveform'
			)

		event_dirs: list[Path] = []
		for downloaded_path in downloaded_paths:
			if not downloaded_path.is_dir():
				raise ValueError(
					f'unexpected artifact from get_event_waveform: {downloaded_path}'
				)

			target_dir = outdir / downloaded_path.name
			if target_dir.exists():
				raise FileExistsError(f'event directory already exists: {target_dir}')

			downloaded_path.rename(target_dir)
			event_dirs.append(target_dir)

	return event_dirs


def _name_stem(
	network_code: str, t0: dt.datetime, stations: Sequence[str], span_min: int
) -> str:
	# 安定名: ネット+分時刻+span+stations の短いハッシュ
	key = f'{network_code}|{t0:%Y%m%d%H%M}|{span_min}|{",".join(sorted(stations))}'
	digest = hashlib.sha1(key.encode()).hexdigest()[:8]
	return f'win_{network_code}_{t0:%Y%m%d%H%M}_{span_min}m_{digest}'


_JST = ZoneInfo('Asia/Tokyo')


def _supports_station_selection(network_code: str) -> bool:
	"""HinetPy の select_stations が効くネットワーク群（Hi-net/F-net/S-net/MeSO-net）。"""
	c = str(network_code).strip()
	if c in {'0101', '0103', '0103A', '0131'}:
		return True
	# S-net 派生コードもまとめて許容（0120, 0120A, 0120B, 0120C...）
	return c.startswith('0120')


def _event_id_from_row(event_row: pd.Series) -> int:
	if 'event_id' not in event_row:
		raise KeyError("event_row must have 'event_id'")
	return int(event_row['event_id'])


def _request_win32_with_retry(
	client: Client,
	*,
	network_code: str,
	start: dt.datetime,
	span_min: int,
	outdir: str,
	data_name: str,
	ctable_name: str,
	threads: int,
	cleanup: bool,
	n_stations: int,
	max_attempts: int = 5,
) -> tuple[str, str]:
	for attempt in range(1, max_attempts + 1):
		data, ctable = client.get_continuous_waveform(
			network_code,
			start,
			span_min,
			outdir=outdir,
			data=data_name,
			ctable=ctable_name,
			threads=threads,
			cleanup=cleanup,
		)

		if data is not None and ctable is not None:
			return data, ctable

		if attempt == max_attempts:
			raise ValueError(
				'Fail to request WIN32 (returned None). '
				f'code={network_code}, start={start}, span_min={span_min}, '
				f'n_stations={n_stations}, attempts={attempt}'
			)

		wait_sec = attempt * 2
		logging.warning(
			'WIN32 request returned None; retrying. '
			f'code={network_code}, start={start}, span_min={span_min}, '
			f'n_stations={n_stations}, attempts={attempt}/{max_attempts}, '
			f'wait_sec={wait_sec}'
		)
		time.sleep(wait_sec)

	raise RuntimeError('unreachable')


def _origin_time_jst_from_row(event_row: pd.Series) -> tuple[dt.datetime, dt.datetime]:
	"""戻り値:
	- origin_time_naive_jst: HinetPy要求用（tz無し、JST相当として扱う）
	- origin_time_aware_jst: event.json保存用（Asia/Tokyo tz付き）
	"""
	if 'origin_time' not in event_row:
		raise KeyError("event_row must have 'origin_time'")

	ts = pd.to_datetime(event_row['origin_time'])
	if ts.tz is None:
		aware = ts.tz_localize(_JST)
	else:
		aware = ts.tz_convert(_JST)
	naive = aware.tz_localize(None)
	return naive.to_pydatetime(), aware.to_pydatetime()


def download_win_for_event(
	client: Client,
	station_list: list[str],
	event_row: pd.Series,
	base_input_dir: str | Path,
	*,
	network_code: str = '0101',
	pre_sec: int,
	post_sec: int,
	span_min_default: int = 1,
	threads: int = 8,
	save_catalog_fields: bool = True,
) -> tuple[Path, list[Path], Path]:
	base_input_dir = Path(base_input_dir)
	base_input_dir.mkdir(parents=True, exist_ok=True)

	event_id = _event_id_from_row(event_row)
	event_dir = base_input_dir / f'{event_id:06d}'
	event_dir.mkdir(parents=True, exist_ok=True)

	origin_time_naive_jst, origin_time_aware_jst = _origin_time_jst_from_row(event_row)

	t_start, t_end = compute_event_time_window(
		origin_time_naive_jst,
		pre_sec=pre_sec,
		post_sec=post_sec,
	)

	cnt_path_list: list[Path] = []
	ch_path: Path | None = None

	for m in minute_range(t_start, t_end):
		cnt_path, ch_tmp, _select_used = download_win_for_stations(
			client,
			stations=station_list,
			when=m,
			network_code=network_code,
			span_min=span_min_default,
			outdir=event_dir,
			threads=threads,
			cleanup=True,
			clear_selection=False,
			skip_if_exists=True,
			use_select=None,
		)
		cnt_path_list.append(cnt_path)
		if ch_path is None:
			ch_path = ch_tmp

	if ch_path is None:
		raise ValueError('failed to obtain channel table (.ch)')

	extra: dict[str, Any] | None = None
	if save_catalog_fields:
		extra = {}
		for k in [
			'latitude_deg',
			'longitude_deg',
			'depth_km',
			'mag1',
			'record_type',
			'hypocenter_flag',
		]:
			if k in event_row.index and not pd.isna(event_row[k]):
				v = event_row[k]
				extra[k] = (
					float(v) if isinstance(v, (int, float, np.number)) else str(v)
				)

	write_event_json(
		event_dir=event_dir,
		event_id=event_id,
		origin_time_jst=origin_time_aware_jst,
		pre_sec=pre_sec,
		post_sec=post_sec,
		network_code=network_code,
		span_min=span_min_default,
		threads=threads,
		stations=station_list,
		cnt_files=[p.name for p in cnt_path_list],
		ch_file=ch_path.name,
		extra=extra,
	)

	return event_dir, cnt_path_list, ch_path


def download_win_for_stations(
	client: Client,
	stations: str | Sequence[str],
	when: dt.datetime,
	*,
	network_code: str = '0101',
	span_min: int = 1,
	outdir: str | Path = '.',
	threads: int = 8,
	cleanup: bool = True,
	clear_selection: bool = False,
	skip_if_exists: bool = True,
	use_select: bool | None = None,
	data_name: str | None = None,
	ctable_name: str | None = None,
) -> tuple[Path, Path, bool]:
	"""指定ステーション群・指定時刻（JST）の Win32 をダウンロード。select無しネットも対応。"""
	t0 = floor_minute(when)
	outdir = Path(outdir)
	outdir.mkdir(parents=True, exist_ok=True)

	station_list = [stations] if isinstance(stations, str) else list(stations)
	if not station_list:
		raise ValueError('stations が空です。')

	select_used = (
		_supports_station_selection(network_code)
		if use_select is None
		else bool(use_select)
	)

	if data_name is None or ctable_name is None:
		# select を使わない場合、stations はサーバ側に反映されないので ALL として安定化
		stations_for_name = station_list if select_used else None
		stem = _name_stem(network_code, t0, stations_for_name, span_min)
		data_name = data_name or f'{stem}.cnt'
		ctable_name = ctable_name or f'{stem}.ch'

	cnt_path = outdir / data_name
	ch_path = outdir / ctable_name

	if skip_if_exists and cnt_path.exists() and ch_path.exists():
		return cnt_path, ch_path, select_used

	if select_used:
		client.select_stations(network_code, station_list)

	data, ctable = _request_win32_with_retry(
		client,
		network_code=network_code,
		start=t0,
		span_min=span_min,
		outdir=str(outdir),
		data_name=data_name,
		ctable_name=ctable_name,
		threads=threads,
		cleanup=cleanup,
		n_stations=len(station_list),
	)

	if clear_selection and select_used:
		client.select_stations(network_code)

	return outdir / data, outdir / ctable, select_used


def download_win_for_event_multi_network(
	client: Client,
	stations_by_network: dict[str, list[str]],
	event_row: pd.Series,
	base_input_dir: str | Path,
	*,
	pre_sec: int,
	post_sec: int,
	span_min_default: int = 1,
	threads: int = 8,
	save_catalog_fields: bool = True,
) -> Path:
	"""複数 network_code を使って event_dir を作り、WIN32 を保存し、event.json を groups 形式で書く。

	groups の各要素に select_used を保存する。
	"""
	base_input_dir = Path(base_input_dir)
	base_input_dir.mkdir(parents=True, exist_ok=True)

	event_id = _event_id_from_row(event_row)
	event_dir = base_input_dir / f'{event_id:06d}'
	event_dir.mkdir(parents=True, exist_ok=True)

	origin_time_naive_jst, origin_time_aware_jst = _origin_time_jst_from_row(event_row)
	t_start, t_end = compute_event_time_window(
		origin_time_naive_jst,
		pre_sec=pre_sec,
		post_sec=post_sec,
	)

	if not stations_by_network:
		raise ValueError('stations_by_network is empty')

	win32_groups: list[dict[str, Any]] = []

	for network_code, station_list in stations_by_network.items():
		code = str(network_code).strip()
		station_list = list(station_list)

		if not code:
			raise ValueError('network_code is empty')
		if not station_list:
			continue

		cnt_path_list: list[Path] = []
		ch_path: Path | None = None
		select_used_group: bool | None = None

		for m in minute_range(t_start, t_end):
			cnt_path, ch_tmp, select_used = download_win_for_stations(
				client,
				stations=station_list,
				when=m,
				network_code=code,
				span_min=span_min_default,
				outdir=event_dir,
				threads=threads,
				cleanup=True,
				clear_selection=False,
				skip_if_exists=True,
				use_select=None,
			)
			cnt_path_list.append(cnt_path)
			if ch_path is None:
				ch_path = ch_tmp
			if select_used_group is None:
				select_used_group = select_used
			elif select_used_group != select_used:
				raise ValueError(f'inconsistent select_used within network={code}')

		if ch_path is None or select_used_group is None:
			raise ValueError(f'failed to obtain channel table (.ch) for code={code}')

		# 選択解除（select が効くネットだけ）
		if select_used_group:
			client.select_stations(code)

		win32_groups.append(
			{
				'network_code': code,
				'select_used': bool(select_used_group),
				'stations': station_list,
				'cnt_files': [p.name for p in cnt_path_list],
				'ch_file': ch_path.name,
			}
		)

	if not win32_groups:
		raise ValueError('no win32_groups created (all station lists empty?)')

	extra: dict[str, Any] | None = None
	if save_catalog_fields:
		extra = {}
		for k in [
			'latitude_deg',
			'longitude_deg',
			'depth_km',
			'mag1',
			'record_type',
			'hypocenter_flag',
		]:
			if k in event_row.index and not pd.isna(event_row[k]):
				v = event_row[k]
				extra[k] = (
					float(v) if isinstance(v, (int, float, np.number)) else str(v)
				)

	write_event_json_win32_groups(
		event_dir=event_dir,
		event_id=event_id,
		origin_time_jst=origin_time_aware_jst,
		pre_sec=pre_sec,
		post_sec=post_sec,
		span_min=span_min_default,
		threads=threads,
		win32_groups=win32_groups,
		extra=extra,
	)

	return event_dir
