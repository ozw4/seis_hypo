from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from HinetPy import Client

from common.time_util import minute_range
from jma.win32_reader import compute_event_time_window


def _name_stem(
	network_code: str, t0: dt.datetime, stations: Sequence[str], span_min: int
) -> str:
	# 安定名: ネット+分時刻+span+stations の短いハッシュ
	key = f'{network_code}|{t0:%Y%m%d%H%M}|{span_min}|{",".join(sorted(stations))}'
	digest = hashlib.sha1(key.encode()).hexdigest()[:8]
	return f'win_{network_code}_{t0:%Y%m%d%H%M}_{span_min}m_{digest}'


def download_win_for_event(
	client: Client,
	station_list: Sequence[str],
	event_row: pd.Series,
	event_dir: Path,
	*,
	network_code: str = '0101',
	pre_sec: int,
	post_sec: int,
	span_min_default: int = 1,
	threads: int = 8,
) -> tuple[list[Path], Path]:
	"""イベント1件に対して、必要な1分Win32をすべてダウンロードする。

	戻り値:
	- cnt_path_list: ダウンロードした .cnt の Path のリスト（時間順）
	- ch_path: 最初に取得した .ch の Path
	"""
	waveform_dir = event_dir / 'waveforms' / 'win32'
	waveform_dir.mkdir(parents=True, exist_ok=True)

	# ★ JMAカタログの origin_time を pandas で一貫して解釈
	origin_time = pd.to_datetime(event_row['origin_time'])

	# pre/post に基づく時刻窓（JST 相当のローカル時刻）
	t_start, t_end = compute_event_time_window(
		origin_time,
		pre_sec=pre_sec,
		post_sec=post_sec,
	)

	cnt_path_list: list[Path] = []
	ch_path: Path | None = None

	for m in minute_range(t_start, t_end):
		cnt_path, ch_tmp = download_win_for_stations(
			client,
			stations=station_list,
			when=m,  # JST 相当
			network_code=network_code,
			span_min=span_min_default,
			outdir=waveform_dir,
			threads=threads,
			cleanup=True,
			clear_selection=False,
			skip_if_exists=True,
		)
		cnt_path_list.append(cnt_path)
		if ch_path is None:
			ch_path = ch_tmp

	if ch_path is None:
		msg = 'no WIN32 files downloaded for this event'
		raise RuntimeError(msg)

	return cnt_path_list, ch_path


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
	data_name: str | None = None,
	ctable_name: str | None = None,
) -> tuple[Path, Path]:
	"""指定ステーション群・指定時刻（JST）の1分 Win32 をダウンロード。既存ならスキップ。"""
	t0 = when.replace(second=0, microsecond=0)
	outdir = Path(outdir)
	outdir.mkdir(parents=True, exist_ok=True)
	station_list = [stations] if isinstance(stations, str) else list(stations)
	if not station_list:
		raise ValueError('stations が空です。')

	if data_name is None or ctable_name is None:
		stem = _name_stem(network_code, t0, station_list, span_min)
		data_name = data_name or f'{stem}.cnt'
		ctable_name = ctable_name or f'{stem}.ch'

	cnt_path = outdir / data_name
	ch_path = outdir / ctable_name

	if skip_if_exists and cnt_path.exists() and ch_path.exists():
		return cnt_path, ch_path

	client.select_stations(network_code, station_list)

	data, ctable = client.get_continuous_waveform(
		network_code,
		t0,
		span_min,
		outdir=str(outdir),
		data=data_name,
		ctable=ctable_name,
		threads=threads,
		cleanup=cleanup,
	)

	# ★ HinetPy が失敗時に (None, None) を返すケースがあるので即時失敗
	if data is None or ctable is None:
		msg = (
			f'Fail to request WIN32 (returned None). '
			f'code={network_code}, start={t0}, span_min={span_min}, '
			f'n_stations={len(station_list)}'
		)
		raise ValueError(msg)

	if clear_selection:
		client.select_stations(network_code)

	return outdir / data, outdir / ctable
