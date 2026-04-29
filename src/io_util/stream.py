import datetime as dt
import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import Stream, Trace, UTCDateTime

from common.core import load_event_json, slice_with_pad
from common.json_io import read_json
from jma.station_reader import read_hinet_channel_table
from jma.win32_reader import read_win32
from pipelines.win32_eqt_continuous_pipelines import parse_win32_cnt_filename


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
	（download_win_for_event の WIN32 が U,N,E の順で入っている
	という決め打ちに合わせる）

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


def _channel_keys(ch_df: pd.DataFrame) -> list[tuple[str, str]]:
	return list(
		zip(
			ch_df['station'].astype(str).str.strip().str.upper(),
			ch_df['component'].astype(str).str.strip().str.upper(),
			strict=False,
		)
	)


def _validate_unique_channel_keys(
	keys: list[tuple[str, str]],
	*,
	ch_path: Path,
) -> None:
	if len(set(keys)) == len(keys):
		return

	seen: set[tuple[str, str]] = set()
	duplicates: list[tuple[str, str]] = []
	for key in keys:
		if key in seen:
			duplicates.append(key)
		seen.add(key)
	raise ValueError(
		f'duplicate station/component rows in .ch: {ch_path} {duplicates[:20]}'
	)


def _ch_paths_for_group(
	*,
	event_dir: Path,
	group: dict,
	cnt_path_list: list[Path],
) -> list[Path]:
	if 'ch_files' in group:
		ch_files = group['ch_files']
		if not isinstance(ch_files, list):
			raise TypeError(f'ch_files must be a list in event.json group: {event_dir}')
		ch_names = [str(n) for n in ch_files]
		if len(ch_names) != len(cnt_path_list):
			raise ValueError(
				'ch_files length must match cnt_files length in event.json group: '
				f'{event_dir} ch_files={len(ch_names)} cnt_files={len(cnt_path_list)}'
			)
	else:
		if 'ch_file' not in group:
			raise ValueError(
				f'event.json group must contain ch_file or ch_files: {event_dir}'
			)
		ch_names = [str(group['ch_file']) for _ in cnt_path_list]

	ch_path_list = [event_dir / name for name in ch_names]
	for ch_path in ch_path_list:
		if not ch_path.is_file():
			raise FileNotFoundError(f'.ch not found: {ch_path}')
	return ch_path_list


def _load_group_ch_df(
	*,
	ch_path: Path,
	group: dict,
	event_dir: Path,
	components_order: tuple[str, str, str],
) -> pd.DataFrame:
	ch_df_raw = read_hinet_channel_table(ch_path)
	ch_df = _force_components_une_by_order(ch_df_raw, components_order=components_order)
	if bool(group.get('select_used', False)):
		selected_stations = [
			str(s).strip().upper() for s in group.get('stations', []) if str(s).strip()
		]
		if not selected_stations:
			raise ValueError(f'select_used is true but stations is empty: {event_dir}')
		station_norm = ch_df['station'].astype(str).str.strip().str.upper()
		missing = sorted(set(selected_stations) - set(station_norm.tolist()))
		if missing:
			raise ValueError(
				f'selected stations missing from .ch: {ch_path} missing={missing[:20]}'
			)
		ch_df = ch_df[station_norm.isin(selected_stations)].reset_index(drop=True)

	keys = _channel_keys(ch_df)
	_validate_unique_channel_keys(keys, ch_path=ch_path)
	return ch_df


def _align_arr_to_reference_keys(
	arr: np.ndarray,
	*,
	keys: list[tuple[str, str]],
	reference_keys: list[tuple[str, str]],
	ch_path: Path,
) -> np.ndarray:
	if keys == reference_keys:
		return arr

	key_to_row = {key: i for i, key in enumerate(keys)}
	reference_key_set = set(reference_keys)
	missing = [key for key in reference_keys if key not in key_to_row]
	extra = [key for key in keys if key not in reference_key_set]
	if missing or extra:
		raise ValueError(
			'channel table station/component set differs across cnt files: '
			f'ch={ch_path} missing={missing[:20]} extra={extra[:20]}'
		)

	return arr[[key_to_row[key] for key in reference_keys], :]


def _cnt_paths_for_group(*, event_dir: Path, group: dict) -> list[Path]:
	cnt_path_list = [event_dir / str(n) for n in group['cnt_files']]
	if not cnt_path_list:
		raise ValueError(f'cnt_files is empty in event.json group: {event_dir}')
	for p in cnt_path_list:
		if not p.is_file():
			raise FileNotFoundError(f'.cnt not found: {p}')
	return cnt_path_list


def _slice_indices_from_first_cnt(
	*,
	first_cnt: Path,
	t_start_jst_naive: dt.datetime,
	t_end_jst_naive: dt.datetime,
	base_sampling_rate_hz: int,
) -> tuple[int, int]:
	first_cnt_info = parse_win32_cnt_filename(first_cnt)
	first_cnt_start_jst = first_cnt_info.start_jst
	offset_start_sec = (t_start_jst_naive - first_cnt_start_jst).total_seconds()
	offset_end_sec = (t_end_jst_naive - first_cnt_start_jst).total_seconds()
	start_idx = round(offset_start_sec * base_sampling_rate_hz)
	end_idx = round(offset_end_sec * base_sampling_rate_hz)
	return start_idx, end_idx


def _read_group_event_array(  # noqa: PLR0913
	*,
	event_dir: Path,
	group: dict,
	t_start_jst_naive: dt.datetime,
	t_end_jst_naive: dt.datetime,
	base_sampling_rate_hz: int,
	duration_sec: int,
	components_order: tuple[str, str, str],
) -> tuple[np.ndarray, pd.DataFrame]:
	cnt_path_list = _cnt_paths_for_group(event_dir=event_dir, group=group)
	ch_path_list = _ch_paths_for_group(
		event_dir=event_dir,
		group=group,
		cnt_path_list=cnt_path_list,
	)
	start_idx, end_idx = _slice_indices_from_first_cnt(
		first_cnt=cnt_path_list[0],
		t_start_jst_naive=t_start_jst_naive,
		t_end_jst_naive=t_end_jst_naive,
		base_sampling_rate_hz=base_sampling_rate_hz,
	)

	arr_list: list[np.ndarray] = []
	reference_ch_df: pd.DataFrame | None = None
	reference_keys: list[tuple[str, str]] | None = None
	for cnt, ch_path in zip(cnt_path_list, ch_path_list, strict=True):
		ch_df = _load_group_ch_df(
			ch_path=ch_path,
			group=group,
			event_dir=event_dir,
			components_order=components_order,
		)
		keys = _channel_keys(ch_df)
		arr_min = read_win32(
			cnt,
			ch_df,
			base_sampling_rate_HZ=int(base_sampling_rate_hz),
			duration_SECOND=int(duration_sec),
		)
		if reference_ch_df is None:
			reference_ch_df = ch_df
			reference_keys = keys
		else:
			if reference_keys is None:
				raise ValueError('internal error: reference_keys is None')
			arr_min = _align_arr_to_reference_keys(
				arr_min,
				keys=keys,
				reference_keys=reference_keys,
				ch_path=ch_path,
			)
		arr_list.append(arr_min)

	if reference_ch_df is None:
		raise ValueError(f'no channel table loaded for event.json group: {event_dir}')

	arr_concat = arr_list[0] if len(arr_list) == 1 else np.concatenate(arr_list, axis=1)
	arr_event = slice_with_pad(arr_concat, start_idx, end_idx)
	return arr_event, reference_ch_df


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

	starttime_utc = UTCDateTime(t_start_jst.astimezone(dt.timezone.utc))
	delta = 1.0 / float(base_sampling_rate_hz)

	st = Stream()
	duration_sec = span_min * 60
	t_start_jst_naive = t_start_jst.replace(tzinfo=None)
	t_end_jst_naive = t_end_jst.replace(tzinfo=None)

	for g in groups:
		arr_event, ch_df = _read_group_event_array(
			event_dir=event_dir,
			group=g,
			t_start_jst_naive=t_start_jst_naive,
			t_end_jst_naive=t_end_jst_naive,
			base_sampling_rate_hz=base_sampling_rate_hz,
			duration_sec=int(duration_sec),
			components_order=components_order,
		)

		stations = sorted(ch_df['station'].unique().tolist())
		for sta in stations:
			df_sta = ch_df[ch_df['station'] == sta]
			for comp in components_order:
				rows = df_sta.index[df_sta['component'] == comp].to_list()
				if len(rows) != 1:
					raise ValueError(
						f'station={sta} comp={comp} must exist exactly once, '
						f'got {len(rows)}'
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

	meta = read_json(meta_path, encoding='utf-8', errors='strict')
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
