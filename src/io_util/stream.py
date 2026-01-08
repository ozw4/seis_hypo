import datetime as dt
import json
import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import Stream, Trace, UTCDateTime

from common.core import load_event_json, slice_with_pad
from common.time_util import floor_minute
from jma.station_reader import read_hinet_channel_table
from jma.win32_reader import read_win32


def infer_net_sta_loc(station_field: str) -> tuple[str, str, str]:
	"""channel_table の station 表記から network/station/location を推定。"""
	if '.' in station_field:
		net, sta = station_field.split('.', 1)
		return net, sta, ''
	return '', station_field, ''


logger = logging.getLogger(__name__)


def _force_components_une_by_order(
	ch_df: pd.DataFrame,
	components_order: tuple[str, str, str] = ('U', 'N', 'E'),
	*,
	log_limit: int = 50,
) -> pd.DataFrame:
	"""方針: .ch の component 表記は信用しない。
	各 station について「出現順の先頭3本」を U,N,E として再ラベルする。
	（download_win_for_event の WIN32 が U,N,E の順で入っているという決め打ちに合わせる）

	追加仕様:
	- 元のcomponentが U,N,E 以外だった局について、変換したことをログに残す
	"""
	if 'station' not in ch_df.columns:
		raise ValueError("channel table must have 'station' column")

	n_need = len(components_order)
	out_parts: list[pd.DataFrame] = []
	bad: list[str] = []

	changed: list[str] = []
	total_changed = 0

	def _as_tuple3(x: Iterable[str]) -> tuple[str, str, str]:
		lst = list(x)
		if len(lst) != 3:
			raise ValueError('internal: expected 3 components')
		return (str(lst[0]), str(lst[1]), str(lst[2]))

	# sort=False で入力順を保持（= .ch の順を尊重）
	for sta, g in ch_df.groupby('station', sort=False):
		if len(g) < n_need:
			bad.append(f'{sta}(n={len(g)})')
			continue

		g2 = g.iloc[:n_need].copy()

		if 'component' in g2.columns:
			orig = _as_tuple3(g2['component'].tolist())
			target = components_order
			if orig != target:
				total_changed += 1
				if len(changed) < log_limit:
					changed.append(f'{sta}:{orig}->{target}')
		else:
			# component列が無い場合も「変換が発生した」とみなして記録
			total_changed += 1
			if len(changed) < log_limit:
				changed.append(f'{sta}:(no component)->{components_order}')

		g2['component'] = list(components_order)
		out_parts.append(g2)

	if bad:
		raise ValueError(f'stations with <{n_need} channels in .ch: {bad[:20]}')

	if not out_parts:
		raise ValueError('no stations remained after forcing U/N/E mapping')
	"""
	if total_changed > 0:
		logger.warning(
			'forced components to %s for %d stations (showing up to %d): %s',
			components_order,
			total_changed,
			log_limit,
			', '.join(changed),
		)
	"""
	return pd.concat(out_parts, ignore_index=True)


def build_stream_from_downloaded_win32(
	event_dir: str | Path,
	*,
	base_sampling_rate_hz: int,
	channel_prefix: str = 'HH',
	components_order: tuple[str, str, str] = ('U', 'N', 'E'),
) -> Stream:
	event_dir = Path(event_dir)
	if not event_dir.is_dir():
		raise FileNotFoundError(f'event_dir not found: {event_dir}')

	meta = load_event_json(event_dir)

	origin_time_jst = pd.to_datetime(meta['origin_time_jst']).to_pydatetime()
	if origin_time_jst.tzinfo is None:
		raise ValueError('origin_time_jst must be timezone-aware (e.g. +09:00)')

	pre_sec = int(meta['window']['pre_sec'])
	post_sec = int(meta['window']['post_sec'])
	span_min = int(meta['win32']['span_min'])

	win32 = meta['win32']
	if isinstance(win32, dict) and 'groups' in win32:
		groups = list(win32['groups'])
		if not groups:
			raise ValueError(f'win32.groups is empty in event.json: {event_dir}')
	else:
		groups = [{'ch_file': win32['ch_file'], 'cnt_files': win32['cnt_files']}]

	t_start_jst = origin_time_jst - dt.timedelta(seconds=pre_sec)
	t_end_jst = origin_time_jst + dt.timedelta(seconds=post_sec)
	if t_end_jst <= t_start_jst:
		raise ValueError('t_end must be later than t_start')

	first_minute = floor_minute(t_start_jst)

	offset_start_sec = (t_start_jst - first_minute).total_seconds()
	offset_end_sec = (t_end_jst - first_minute).total_seconds()
	start_idx = int(round(offset_start_sec * base_sampling_rate_hz))
	end_idx = int(round(offset_end_sec * base_sampling_rate_hz))

	starttime_utc = UTCDateTime(t_start_jst.astimezone(dt.timezone.utc))
	delta = 1.0 / float(base_sampling_rate_hz)

	st = Stream()
	duration_sec = span_min * 60

	for g in groups:
		ch_path = event_dir / str(g['ch_file'])
		if not ch_path.is_file():
			raise FileNotFoundError(f'.ch not found: {ch_path}')

		cnt_path_list = [event_dir / str(n) for n in g['cnt_files']]
		if not cnt_path_list:
			raise ValueError(f'cnt_files is empty in event.json group: {event_dir}')
		for p in cnt_path_list:
			if not p.is_file():
				raise FileNotFoundError(f'.cnt not found: {p}')

		ch_df_raw = read_hinet_channel_table(ch_path)
		ch_df = _force_components_une_by_order(
			ch_df_raw, components_order=components_order
		)

		arr_list: list[np.ndarray] = []
		for cnt in cnt_path_list:
			arr_min = read_win32(
				cnt,
				ch_df,
				base_sampling_rate_HZ=int(base_sampling_rate_hz),
				duration_SECOND=int(duration_sec),
			)
			arr_list.append(arr_min)

		arr_concat = (
			arr_list[0] if len(arr_list) == 1 else np.concatenate(arr_list, axis=1)
		)
		arr_event = slice_with_pad(arr_concat, start_idx, end_idx)

		stations = sorted(ch_df['station'].unique().tolist())
		for sta in stations:
			df_sta = ch_df[ch_df['station'] == sta]
			for comp in components_order:
				rows = df_sta.index[df_sta['component'] == comp].to_list()
				if len(rows) != 1:
					raise ValueError(
						f'station={sta} comp={comp} must exist exactly once, got {len(rows)}'
					)
				i = rows[0]

				tr = Trace(data=arr_event[i].astype(np.float32, copy=False))
				tr.stats.starttime = starttime_utc
				tr.stats.delta = delta
				tr.stats.station = sta
				tr.stats.channel = f'{channel_prefix}{comp}'
				st += tr

	return st


def _validate_index_0_to_n_minus_1(idx: pd.Series, *, label: str) -> None:
	idx_i = idx.astype(int)
	if idx_i.min() != 0:
		raise ValueError(f'{label} index must start at 0. min={idx_i.min()}')
	if idx_i.nunique() != len(idx_i):
		raise ValueError(f'{label} index must be unique per row')
	if idx_i.max() != len(idx_i) - 1:
		raise ValueError(
			f'{label} index must end at N-1. max={idx_i.max()} N={len(idx_i)}'
		)


def build_stream_from_forge_event_npy(
	event_dir: str | Path,
	*,
	channel_code: str = 'DASZ',
) -> Stream:
	"""Forge (DAS) のイベントdirから waveform.npy を読み、ObsPy Stream に変換する。

	期待するファイル:
	- waveform.npy: shape (C, T)
	- meta.json: fs_hz, window_start_utc を含む
	- stations.csv: station_id, index を含む（index昇順がC方向の並び）

	注意:
	- comp=('Z',) を前提とし、Trace.stats.channel の末尾が 'Z' になるようにする。
	"""
	event_dir = Path(event_dir)
	if not event_dir.is_dir():
		raise FileNotFoundError(f'event_dir not found: {event_dir}')

	npy_path = event_dir / 'waveform.npy'
	meta_path = event_dir / 'meta.json'
	stations_path = event_dir / 'stations.csv'

	if not npy_path.is_file():
		raise FileNotFoundError(f'waveform.npy not found: {npy_path}')
	if not meta_path.is_file():
		raise FileNotFoundError(f'meta.json not found: {meta_path}')
	if not stations_path.is_file():
		raise FileNotFoundError(f'stations.csv not found: {stations_path}')

	meta = json.loads(meta_path.read_text(encoding='utf-8'))
	fs_hz = float(meta['fs_hz'])
	if fs_hz <= 0.0:
		raise ValueError(f'fs_hz must be > 0. got {fs_hz}')

	t0 = pd.to_datetime(meta['window_start_utc'], utc=True)
	if pd.isna(t0):
		raise ValueError(
			f'failed to parse window_start_utc: {meta.get("window_start_utc")}'
		)
	starttime_utc = UTCDateTime(t0.to_pydatetime())
	delta = 1.0 / float(fs_hz)

	sta = pd.read_csv(stations_path)
	required = ['station_id', 'index']
	missing = [c for c in required if c not in sta.columns]
	if missing:
		raise ValueError(
			f'stations.csv missing columns: {missing}. cols={list(sta.columns)}'
		)

	sta['station_id'] = sta['station_id'].astype(str)
	sta['index'] = sta['index'].astype(int)
	sta = sta.sort_values('index').reset_index(drop=True)
	_validate_index_0_to_n_minus_1(sta['index'], label='stations.csv')

	x = np.load(npy_path, allow_pickle=False)
	if x.ndim != 2:
		raise ValueError(f'waveform.npy must be 2D (C,T). got shape={x.shape}')
	c, t = int(x.shape[0]), int(x.shape[1])
	if c != len(sta):
		raise ValueError(
			f'channel count mismatch: waveform C={c} vs stations rows={len(sta)}'
		)
	if t <= 0:
		raise ValueError(f'waveform T must be > 0. got T={t}')

	if str(channel_code)[-1] != 'Z':
		raise ValueError(
			f"channel_code must end with 'Z' for comp=('Z',). got: {channel_code}"
		)

	st = Stream()
	for i, station_id in enumerate(sta['station_id'].tolist()):
		tr = Trace(data=np.asarray(x[i], dtype=np.float32))
		tr.stats.starttime = starttime_utc
		tr.stats.delta = delta
		tr.stats.station = station_id
		tr.stats.channel = str(channel_code)
		st += tr

	return st
