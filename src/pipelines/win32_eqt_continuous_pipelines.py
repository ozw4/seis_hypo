from __future__ import annotations

import csv
import datetime as dt
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common.core import as_int_rate
from jma.ch_table_util import normalize_ch_table_components_to_une
from jma.station_reader import read_hinet_channel_table
from jma.win32_reader import (
	read_win32,
	read_win32_resampled,
	scan_channel_sampling_rate_map_win32,
)
from pipelines.das_pick_csv_accumulator import PickAccumulator

_JST = dt.timezone(dt.timedelta(hours=9))
_WIN32_CNT_RE = re.compile(
	r'^win_([0-9A-Za-z]+)_(\d{12})_([1-9][0-9]*)m_[0-9a-fA-F]+\.cnt$'
)


@dataclass(frozen=True, slots=True)
class Win32CntFileInfo:
	network_code: str
	start_jst: dt.datetime
	span_min: int


@dataclass(frozen=True, slots=True)
class Win32WindowMeta:
	network_code: str
	window_start_jst: dt.datetime
	window_start_epoch_ms: int


@dataclass(frozen=True, slots=True)
class Win32EqtPickStats:
	windows_processed: int
	picks_written: int
	stations_processed: int


def _build_eqt_runner_3c(*, weights: str, in_samples: int, batch_stations: int):
	from pick.eqt_runner_3c import EqTWindowRunner3C

	return EqTWindowRunner3C(
		weights=str(weights),
		in_samples=int(in_samples),
		batch_stations=int(batch_stations),
	)


def _ensure_parent(p: Path) -> None:
	p.parent.mkdir(parents=True, exist_ok=True)


def _jst_to_epoch_ms(ts: dt.datetime) -> int:
	if ts.tzinfo is None:
		t_aware = ts.replace(tzinfo=_JST)
	else:
		t_aware = ts.astimezone(_JST)
	return int(round(float(t_aware.timestamp()) * 1000.0))


def _epoch_ms_to_jst_iso(ms: int) -> str:
	return dt.datetime.fromtimestamp(float(ms) / 1000.0, tz=_JST).isoformat(
		timespec='milliseconds'
	)


def parse_win32_cnt_filename(path: str | Path) -> Win32CntFileInfo:
	p = Path(path)
	m = _WIN32_CNT_RE.fullmatch(p.name)
	if m is None:
		raise ValueError(f'invalid WIN32 .cnt filename format: {p.name}')

	network_code = str(m.group(1))
	start_jst = dt.datetime.strptime(str(m.group(2)), '%Y%m%d%H%M')
	span_min = int(m.group(3))
	if span_min <= 0:
		raise ValueError(f'invalid span_min in filename: {p.name}')

	return Win32CntFileInfo(
		network_code=network_code,
		start_jst=start_jst,
		span_min=span_min,
	)


def _station_axis_layout_from_ch_une(
	ch_une: pd.DataFrame,
) -> tuple[list[str], np.ndarray, np.ndarray]:
	if ch_une.empty:
		raise ValueError('no station rows after U/N/E normalization')

	axis_to_idx = {'U': 0, 'N': 1, 'E': 2}
	station_codes: list[str] = []
	station_to_idx: dict[str, int] = {}
	row_station_idx: list[int] = []
	row_axis_idx: list[int] = []
	seen_pairs: set[tuple[int, int]] = set()

	for i in range(len(ch_une)):
		r = ch_une.iloc[int(i)]
		sta = str(r['station'])
		if not sta:
			raise ValueError(f'empty station at normalized row index={i}')

		comp = str(r['component']).upper()
		axis_idx = axis_to_idx.get(comp)
		if axis_idx is None:
			raise ValueError(
				f'invalid component at normalized row index={i}: {comp}'
			)

		sta_idx = station_to_idx.get(sta)
		if sta_idx is None:
			sta_idx = len(station_codes)
			station_to_idx[sta] = sta_idx
			station_codes.append(sta)

		pair = (int(sta_idx), int(axis_idx))
		if pair in seen_pairs:
			raise ValueError(
				f'duplicate station/axis rows after normalization: station={sta} axis={comp}'
			)
		seen_pairs.add(pair)

		row_station_idx.append(int(sta_idx))
		row_axis_idx.append(int(axis_idx))

	return (
		station_codes,
		np.asarray(row_station_idx, dtype=np.int32),
		np.asarray(row_axis_idx, dtype=np.int32),
	)


def prepare_win32_ch_table_une(
	ch_table: pd.DataFrame | str | Path,
) -> tuple[pd.DataFrame, list[str]]:
	if isinstance(ch_table, (str, Path)):
		ch_df = read_hinet_channel_table(ch_table)
	else:
		ch_df = ch_table.copy()

	ch_une = normalize_ch_table_components_to_une(
		ch_df,
		require_full_une=False,
	).reset_index(drop=True)
	station_codes, _, _ = _station_axis_layout_from_ch_une(ch_une)
	return ch_une, station_codes


def _scatter_to_station_3c(
	arr_2d: np.ndarray,
	*,
	row_station_idx: np.ndarray,
	row_axis_idx: np.ndarray,
	station_count: int,
) -> np.ndarray:
	if arr_2d.ndim != 2:
		raise ValueError(f'arr_2d must be 2D (n_ch, T), got {arr_2d.shape}')
	n_ch, n_t = arr_2d.shape
	if int(n_ch) != len(row_station_idx) or int(n_ch) != len(row_axis_idx):
		raise ValueError(
			'n_ch mismatch: '
			f'got {n_ch}, expected {len(row_station_idx)}'
		)
	out = np.zeros((int(station_count), 3, int(n_t)), dtype=np.float32)
	out[row_station_idx, row_axis_idx, :] = np.asarray(arr_2d, dtype=np.float32)
	return out


def _validate_cnt_channels_for_fixed_fs(
	*,
	cnt_path: Path,
	ch_une: pd.DataFrame,
	target_fs_hz: int,
) -> None:
	ch_ints = ch_une['ch_int'].to_numpy(dtype=np.int32)
	req_set = set(int(x) for x in ch_ints.tolist())
	fs_by_ch = scan_channel_sampling_rate_map_win32(
		cnt_path,
		channel_filter=req_set,
		on_mixed='raise',
	)

	found_set = set(int(x) for x in fs_by_ch.keys())
	missing = sorted(req_set - found_set)
	bad = sorted(
		(int(ch), int(fs))
		for ch, fs in fs_by_ch.items()
		if int(fs) != int(target_fs_hz)
	)

	if not missing and not bad:
		return

	lines = [
		f'WIN32 fs precheck failed: {cnt_path.name}',
		f'target_fs_hz={int(target_fs_hz)}',
	]
	if missing:
		lines.append(
			f'missing channel_no in WIN32: n={len(missing)} examples={missing[:20]}'
		)
	if bad:
		lines.append(
			'channel_no with fs != target: '
			f'n={len(bad)} examples={bad[:20]}'
		)
	lines.append(
		'use_resampled=True to decode by per-channel fs and resample to target fs'
	)
	raise ValueError('\n'.join(lines))


def iter_win32_station_windows(
	*,
	cnt_paths: Sequence[str | Path],
	ch_table: pd.DataFrame | str | Path,
	target_fs_hz: float,
	eqt_in_samples: int,
	eqt_overlap: int,
	use_resampled: bool = False,
	resampled_missing_channel_policy: str = 'zero',
) -> Iterator[tuple[np.ndarray, Win32WindowMeta]]:
	if not cnt_paths:
		raise ValueError('cnt_paths is empty')

	fs_i = as_int_rate(float(target_fs_hz), 'target_fs_hz')
	L = int(eqt_in_samples)
	O = int(eqt_overlap)
	H = int(L - O)
	if L <= 0:
		raise ValueError('eqt_in_samples must be positive')
	if O < 0 or O >= L:
		raise ValueError('eqt_overlap must satisfy 0 <= overlap < eqt_in_samples')
	if H <= 0:
		raise ValueError('hop length must be positive')

	ch_une, station_codes = prepare_win32_ch_table_une(ch_table)
	station_codes_chk, row_station_idx, row_axis_idx = _station_axis_layout_from_ch_une(
		ch_une
	)
	if station_codes_chk != station_codes:
		raise ValueError('internal error: station order mismatch in ch_une layout')
	station_count = len(station_codes)

	read_fn = read_win32_resampled if bool(use_resampled) else read_win32

	network_code: str | None = None
	prev_end_jst: dt.datetime | None = None

	buf_3c: np.ndarray | None = None
	buf_start_jst: dt.datetime | None = None

	for raw_path in cnt_paths:
		cnt_path = Path(raw_path)
		info = parse_win32_cnt_filename(cnt_path)

		if network_code is None:
			network_code = str(info.network_code)
		elif str(info.network_code) != str(network_code):
			raise ValueError(
				f'mixed network codes in one run: {network_code} vs {info.network_code}'
			)

		if prev_end_jst is not None and info.start_jst != prev_end_jst:
			raise ValueError(
				f'.cnt timeline is not contiguous: expected {prev_end_jst} got {info.start_jst}'
			)

		duration_sec = int(info.span_min) * 60
		if bool(use_resampled):
			arr_2d = read_fn(
				cnt_path,
				ch_une,
				target_sampling_rate_HZ=int(fs_i),
				duration_SECOND=int(duration_sec),
				missing_channel_policy=str(resampled_missing_channel_policy),
			)
		else:
			_validate_cnt_channels_for_fixed_fs(
				cnt_path=cnt_path,
				ch_une=ch_une,
				target_fs_hz=int(fs_i),
			)
			arr_2d = read_fn(
				cnt_path,
				ch_une,
				base_sampling_rate_HZ=int(fs_i),
				duration_SECOND=int(duration_sec),
			)

		arr_3c = _scatter_to_station_3c(
			arr_2d,
			row_station_idx=row_station_idx,
			row_axis_idx=row_axis_idx,
			station_count=station_count,
		)

		if buf_3c is None:
			buf_3c = arr_3c
			buf_start_jst = info.start_jst
		else:
			buf_3c = np.concatenate([buf_3c, arr_3c], axis=2)

		if buf_start_jst is None:
			raise ValueError('internal error: buffer start time is missing')

		step = dt.timedelta(seconds=float(H) / float(fs_i))
		while int(buf_3c.shape[2]) >= int(L):
			window = np.asarray(buf_3c[:, :, 0:L], dtype=np.float32)
			meta = Win32WindowMeta(
				network_code=str(network_code),
				window_start_jst=buf_start_jst,
				window_start_epoch_ms=_jst_to_epoch_ms(buf_start_jst),
			)
			yield window, meta

			buf_3c = np.asarray(buf_3c[:, :, H:], dtype=np.float32)
			buf_start_jst = buf_start_jst + step

		prev_end_jst = info.start_jst + dt.timedelta(minutes=int(info.span_min))


class _Win32PickWriterAdapter:
	def __init__(
		self,
		*,
		base_writer: csv.writer,
		station_codes: list[str],
		network_code: str,
		include_network_code: bool,
	):
		self.base_writer = base_writer
		self.station_codes = station_codes
		self.network_code = str(network_code)
		self.include_network_code = bool(include_network_code)

	def writerow(self, row: list[object]) -> None:
		c_idx = int(row[2])
		if c_idx < 0 or c_idx >= len(self.station_codes):
			raise ValueError(
				f'channel index out of station range: idx={c_idx}, n={len(self.station_codes)}'
			)
		phase = str(row[3]).upper()
		if phase not in ['P', 'S']:
			raise ValueError(f'invalid phase from accumulator: {phase}')
		t_ms = int(row[4])
		w_conf = float(row[6])

		out = [
			str(self.station_codes[c_idx]),
			phase,
			_epoch_ms_to_jst_iso(t_ms),
			float(w_conf),
		]
		if bool(self.include_network_code):
			out.append(str(self.network_code))
		self.base_writer.writerow(out)


def pipeline_win32_eqt_pick_to_csv(
	*,
	cnt_paths: Sequence[str | Path],
	ch_table: pd.DataFrame | str | Path,
	out_csv: str | Path,
	eqt_weights: str = 'original',
	eqt_in_samples: int = 6000,
	eqt_overlap: int = 3000,
	eqt_batch_stations: int = 64,
	use_resampled: bool = False,
	resampled_missing_channel_policy: str = 'zero',
	target_fs_hz: float = 100.0,
	det_gate_enable: bool = True,
	det_threshold: float = 0.30,
	p_threshold: float = 0.10,
	s_threshold: float = 0.10,
	min_pick_sep_samples: int = 50,
	overlap_merge: str = 'max',
	include_network_code: bool = True,
	print_every_windows: int = 50,
) -> Win32EqtPickStats:
	if not cnt_paths:
		raise ValueError('cnt_paths is empty')
	if overlap_merge not in ['max', 'mean']:
		raise ValueError(f"overlap_merge must be 'max' or 'mean', got {overlap_merge}")

	fs_i = as_int_rate(float(target_fs_hz), 'target_fs_hz')
	L = int(eqt_in_samples)
	O = int(eqt_overlap)
	H = int(L - O)
	if H <= 0:
		raise ValueError('eqt_overlap must be smaller than eqt_in_samples')

	ch_une, station_codes = prepare_win32_ch_table_une(ch_table)
	network_code = parse_win32_cnt_filename(Path(cnt_paths[0])).network_code

	runner = _build_eqt_runner_3c(
		weights=str(eqt_weights),
		in_samples=int(eqt_in_samples),
		batch_stations=int(eqt_batch_stations),
	)

	accumulator = PickAccumulator(
		channel_range=None,
		channel_ids=np.arange(len(station_codes), dtype=np.int32),
		min_pick_sep_samples=int(min_pick_sep_samples),
		p_threshold=float(p_threshold),
		s_threshold=float(s_threshold),
		det_gate_enable=bool(det_gate_enable),
		det_threshold=float(det_threshold),
	)

	pending_d: np.ndarray | None = None
	pending_p: np.ndarray | None = None
	pending_s: np.ndarray | None = None
	pending_start_ms: int | None = None

	windows_processed = 0
	picks_written = 0

	out_csv = Path(out_csv)
	_ensure_parent(out_csv)

	with out_csv.open('w', newline='', encoding='utf-8') as f:
		wcsv = csv.writer(f)
		head = ['station_code', 'Phase', 'pick_time', 'w_conf']
		if bool(include_network_code):
			head.append('network_code')
		wcsv.writerow(head)

		writer_adapter = _Win32PickWriterAdapter(
			base_writer=wcsv,
			station_codes=station_codes,
			network_code=network_code,
			include_network_code=bool(include_network_code),
		)

		for wave_3c, meta in iter_win32_station_windows(
			cnt_paths=cnt_paths,
			ch_table=ch_une,
			target_fs_hz=float(fs_i),
			eqt_in_samples=int(eqt_in_samples),
			eqt_overlap=int(eqt_overlap),
			use_resampled=bool(use_resampled),
			resampled_missing_channel_policy=str(resampled_missing_channel_policy),
		):
			if str(meta.network_code) != str(network_code):
				raise ValueError(
					f'mixed network codes in one run: {network_code} vs {meta.network_code}'
				)
			det_w, p_w, s_w = runner.predict_window(wave_3c)

			if pending_p is None:
				pending_d = det_w
				pending_p = p_w
				pending_s = s_w
				pending_start_ms = int(meta.window_start_epoch_ms)
			else:
				if overlap_merge == 'max':
					pending_d[:, H:L] = np.maximum(pending_d[:, H:L], det_w[:, 0:O])
					pending_p[:, H:L] = np.maximum(pending_p[:, H:L], p_w[:, 0:O])
					pending_s[:, H:L] = np.maximum(pending_s[:, H:L], s_w[:, 0:O])
				else:
					pending_d[:, H:L] = (pending_d[:, H:L] + det_w[:, 0:O]) * 0.5
					pending_p[:, H:L] = (pending_p[:, H:L] + p_w[:, 0:O]) * 0.5
					pending_s[:, H:L] = (pending_s[:, H:L] + s_w[:, 0:O]) * 0.5

				picks_written += accumulator.accumulate_chunk(
					writer_adapter,
					seg_id=0,
					block_start=0,
					chunk_d=pending_d[:, 0:H] if bool(det_gate_enable) else None,
					chunk_p=pending_p[:, 0:H],
					chunk_s=pending_s[:, 0:H],
					chunk_start_ms=int(pending_start_ms),
					chunk_fs_hz=float(fs_i),
				)

				pending_d = np.concatenate([pending_d[:, H:L], det_w[:, O:L]], axis=1)
				pending_p = np.concatenate([pending_p[:, H:L], p_w[:, O:L]], axis=1)
				pending_s = np.concatenate([pending_s[:, H:L], s_w[:, O:L]], axis=1)
				pending_start_ms = int(meta.window_start_epoch_ms)

			windows_processed += 1
			if (
				int(print_every_windows) > 0
				and (int(windows_processed) % int(print_every_windows)) == 0
			):
				print(f'[INFO] windows={windows_processed} picks={picks_written}')

		if pending_p is not None:
			picks_written += accumulator.accumulate_chunk(
				writer_adapter,
				seg_id=0,
				block_start=0,
				chunk_d=pending_d if bool(det_gate_enable) else None,
				chunk_p=pending_p,
				chunk_s=pending_s,
				chunk_start_ms=int(pending_start_ms),
				chunk_fs_hz=float(fs_i),
			)
			picks_written += accumulator.flush(writer_adapter, seg_id=0, block_start=0)

	print(
		f'[DONE] wrote CSV: {out_csv} windows={windows_processed} picks={picks_written}'
	)
	return Win32EqtPickStats(
		windows_processed=int(windows_processed),
		picks_written=int(picks_written),
		stations_processed=len(station_codes),
	)
