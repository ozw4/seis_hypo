from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from obspy import Stream, Trace, UTCDateTime
from scipy.signal import detrend as sp_detrend

from common.config import JmaDtPickErrorPreprocessConfig
from jma.prepare.inventory import InventoryResult, build_inventory
from jma.station_reader import read_hinet_channel_table
from jma.win32_reader import (
	read_win32,
	scan_channel_sampling_rate_map_win32,
	select_hinet_channels,
)
from waveform.filters import bandpass_iir_filtfilt
from waveform.preprocess import resample_window_poly

MIN_INPUT_FS_HZ = 30
ZERO_FRAC_MAX = 0.98
CLIP_FRAC_MAX = 0.02


def _qc_trace_raw(
	x: np.ndarray, *, zero_frac_max: float, clip_frac_max: float
) -> tuple[bool, str]:
	if x.ndim != 1:
		return False, f'bad_shape:{x.shape}'
	if not np.isfinite(x).all():
		return False, 'nan_or_inf'
	if float(np.max(np.abs(x))) <= 0.0:
		return False, 'all_zero'

	zero_frac = float(np.mean(x == 0.0))
	if zero_frac > float(zero_frac_max):
		return False, f'too_many_zeros:{zero_frac:.3f}'

	mx = float(np.max(x))
	mn = float(np.min(x))
	if mx == mn:
		return False, 'constant'

	frac_max = float(np.mean(x == mx))
	frac_min = float(np.mean(x == mn))
	if max(frac_max, frac_min) > float(clip_frac_max):
		return False, f'clipped_suspect:{max(frac_max, frac_min):.3f}'

	return True, ''


@dataclass(frozen=True)
class UStreamLoadResult:
	t0: datetime
	fs_hz: float
	stream: Stream
	stations_used: list[str]
	skips: list[dict[str, object]]


def load_u_stream_for_event(
	event_dir: str | Path,
	*,
	stations: list[str] | None,
	preprocess_cfg: JmaDtPickErrorPreprocessConfig,
	inventory: InventoryResult | None = None,
) -> UStreamLoadResult:
	event_dir = Path(event_dir)
	if not event_dir.is_dir():
		raise FileNotFoundError(f'event_dir not found: {event_dir}')

	inv = build_inventory(event_dir) if inventory is None else inventory
	source_by_id = {s.source_id: s for s in inv.sources}

	if stations is None:
		station_list = list(dict.fromkeys(sorted(inv.station_meta.keys())))
	else:
		station_list = list(dict.fromkeys([str(s) for s in stations]))

	required_u_by_source: dict[str, set[int]] = {}
	u_meta_by_station: dict[str, dict[str, object]] = {}
	skips: list[dict[str, object]] = []

	for sta in station_list:
		st_meta = inv.station_meta.get(sta)
		if st_meta is None:
			u_meta_by_station[sta] = {'_skip_reason': 'station_not_in_inventory'}
			continue

		u = st_meta.get('U')
		if not u:
			u_meta_by_station[sta] = {'_skip_reason': 'no_U_component'}
			continue

		source_id = str(u['source_id'])
		ch_int = int(u['ch_int'])

		u_meta_by_station[sta] = {
			'_skip_reason': '',
			'u': u,
			'st_meta': st_meta,
			'source_id': source_id,
			'ch_int': ch_int,
		}
		required_u_by_source.setdefault(source_id, set()).add(ch_int)

	fs_map_by_source: dict[str, dict[int, int]] = {}
	for source_id, ch_set in required_u_by_source.items():
		if source_id not in source_by_id:
			continue
		data_path = Path(source_by_id[source_id].data_path)
		fs_map = scan_channel_sampling_rate_map_win32(
			data_path,
			channel_filter=set(ch_set),
			on_mixed='drop',
		)
		fs_map_by_source[source_id] = {int(k): int(v) for k, v in fs_map.items()}

	candidates: list[dict[str, object]] = []
	for sta in station_list:
		meta0 = u_meta_by_station.get(sta)
		if meta0 is None:
			skips.append(
				{
					'event_dir': str(event_dir),
					'station': str(sta),
					'reason': 'station_meta_missing_internal',
				}
			)
			continue

		skip_reason = str(meta0.get('_skip_reason', ''))
		if skip_reason:
			skips.append(
				{
					'event_dir': str(event_dir),
					'station': str(sta),
					'reason': skip_reason,
				}
			)
			continue

		source_id = str(meta0['source_id'])
		ch_int = int(meta0['ch_int'])

		if source_id not in source_by_id:
			skips.append(
				{
					'event_dir': str(event_dir),
					'station': str(sta),
					'reason': f'source_not_found:{source_id}',
				}
			)
			continue

		fs_map = fs_map_by_source.get(source_id, {})
		fs_in = fs_map.get(ch_int, None)
		if fs_in is None:
			skips.append(
				{
					'event_dir': str(event_dir),
					'station': str(sta),
					'reason': 'mixed_sampling_rate_or_not_found',
				}
			)
			continue
		if int(fs_in) <= int(MIN_INPUT_FS_HZ):
			skips.append(
				{
					'event_dir': str(event_dir),
					'station': str(sta),
					'reason': f'low_sampling_rate:{int(fs_in)}',
				}
			)
			continue

		candidates.append(
			{
				'station': str(sta),
				'source_id': source_id,
				'ch_int': int(ch_int),
				'fs_in': int(fs_in),
			}
		)

	if not candidates:
		raise ValueError(f'no usable stations found in {event_dir}')

	candidates_by_source_fs: dict[tuple[str, int], list[dict[str, object]]] = {}
	for item in candidates:
		key = (str(item['source_id']), int(item['fs_in']))
		candidates_by_source_fs.setdefault(key, []).append(item)

	ch_table_cache: dict[str, object] = {}
	traces: list[Trace] = []
	stations_used: list[str] = []
	ref_start_time: datetime | None = None
	ref_span_seconds: int | None = None

	for (source_id, fs_in), items in candidates_by_source_fs.items():
		src = source_by_id[source_id]
		meta = inv.sources_meta[source_id]
		# inventory start_time is treated as JST (naive)
		start_time = datetime.fromisoformat(str(meta['start_time']))
		span_seconds = int(meta['span_seconds'])

		if ref_start_time is None:
			ref_start_time = start_time
			ref_span_seconds = span_seconds
		elif ref_start_time != start_time or ref_span_seconds != span_seconds:
			for item in items:
				skips.append(
					{
						'event_dir': str(event_dir),
						'station': str(item['station']),
						'reason': 'window_mismatch',
					}
				)
			continue

		data_path = Path(src.data_path)
		ch_path = Path(src.ch_path)
		if source_id not in ch_table_cache:
			ch_table_cache[source_id] = read_hinet_channel_table(ch_path)
		ch_table = ch_table_cache[source_id]

		ch_hex_set = set(str(x) for x in ch_table['ch_hex'].tolist())
		channels_hex_use: list[str] = []
		items_use: list[dict[str, object]] = []
		for item in items:
			ch_hex = f'{int(item["ch_int"]):04X}'
			if ch_hex not in ch_hex_set:
				skips.append(
					{
						'event_dir': str(event_dir),
						'station': str(item['station']),
						'reason': 'channel_not_in_table',
					}
				)
				continue
			channels_hex_use.append(ch_hex)
			items_use.append(item)

		if not channels_hex_use:
			continue

		df_sel = select_hinet_channels(ch_table, channels_hex=channels_hex_use)
		row_by_hex = {str(ch): int(i) for i, ch in enumerate(df_sel['ch_hex'].tolist())}

		raw2d = read_win32(
			data_path,
			df_sel,
			base_sampling_rate_HZ=int(fs_in),
			duration_SECOND=int(span_seconds),
			channels_hex=None,
		)

		out_len = int(span_seconds) * int(preprocess_cfg.fs_target_hz)
		for item in items_use:
			ch_hex = f'{int(item["ch_int"]):04X}'
			row = row_by_hex.get(ch_hex)
			if row is None:
				skips.append(
					{
						'event_dir': str(event_dir),
						'station': str(item['station']),
						'reason': 'channel_not_in_table',
					}
				)
				continue
			raw = np.asarray(raw2d[int(row)], dtype=np.float32)
			ok, why = _qc_trace_raw(
				raw,
				zero_frac_max=float(ZERO_FRAC_MAX),
				clip_frac_max=float(CLIP_FRAC_MAX),
			)
			if not ok:
				skips.append(
					{
						'event_dir': str(event_dir),
						'station': str(item['station']),
						'reason': f'qc_fail:{why}',
					}
				)
				continue
			raw_rs = resample_window_poly(
				raw[None, :],
				fs_in=float(fs_in),
				fs_out=float(preprocess_cfg.fs_target_hz),
				out_len=int(out_len),
			)[0]

			x_dt = np.asarray(raw_rs, dtype=float)
			if preprocess_cfg.detrend is not None:
				if str(preprocess_cfg.detrend) != 'linear':
					raise ValueError(f'unsupported detrend: {preprocess_cfg.detrend!r}')
				x_dt = sp_detrend(x_dt, type='linear')

			bp = preprocess_cfg.bandpass
			x_proc = bandpass_iir_filtfilt(
				x_dt,
				fs=float(preprocess_cfg.fs_target_hz),
				fstop_lo=float(bp.fstop_lo),
				fpass_lo=float(bp.fpass_lo),
				fpass_hi=float(bp.fpass_hi),
				fstop_hi=float(bp.fstop_hi),
				gpass=float(bp.gpass),
				gstop=float(bp.gstop),
			)

			tr = Trace(data=np.asarray(x_proc, dtype=np.float32))
			tr.stats.station = str(item['station'])
			tr.stats.channel = 'HHU'
			tr.stats.starttime = UTCDateTime(start_time - timedelta(hours=9))
			tr.stats.delta = 1.0 / float(preprocess_cfg.fs_target_hz)
			traces.append(tr)
			stations_used.append(str(item['station']))

	if ref_start_time is None or ref_span_seconds is None or not traces:
		raise ValueError(f'no usable traces built for {event_dir}')

	st = Stream(traces=traces)
	return UStreamLoadResult(
		t0=ref_start_time,
		fs_hz=float(preprocess_cfg.fs_target_hz),
		stream=st,
		stations_used=stations_used,
		skips=skips,
	)
