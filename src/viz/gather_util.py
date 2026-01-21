from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from obspy import Stream

from common.time_util import as_utc_aware
from io_util.trace_util import trace_station_comp


def build_gather_matrix(
	st: Stream,
	comp: str,
	*,
	start_time_mode: str = 'min',
	length_mode: str = 'max',
	align_on_start: bool = True,
) -> tuple[np.ndarray, list[str], float, dt.datetime]:
	trs: list[tuple[str, object]] = []
	for tr in st:
		sta_full, c = trace_station_comp(tr)
		if c == comp:
			trs.append((sta_full, tr))

	if not trs:
		raise ValueError(f'no traces for comp={comp}')

	fs = float(trs[0][1].stats.sampling_rate)
	start_times = [t.stats.starttime for _, t in trs]
	if start_time_mode == 'min':
		t0 = min(start_times)
	elif start_time_mode == 'first':
		t0 = start_times[0]
	else:
		raise ValueError(f'unknown start_time_mode: {start_time_mode}')
	t_start_utc = as_utc_aware(t0.datetime)

	if length_mode == 'max':
		# determine length to cover all traces
		end_times = [t.stats.endtime for _, t in trs]
		t1 = max(end_times)
		total_samples = int(round((t1 - t0) * fs)) + 1
	elif length_mode == 'min':
		total_samples = min(int(t.stats.npts) for _, t in trs)
	else:
		raise ValueError(f'unknown length_mode: {length_mode}')

	stations = [sta for sta, _ in trs]
	data = np.zeros((len(trs), total_samples), dtype=np.float32)

	for i, (_, tr) in enumerate(trs):
		if align_on_start:
			offset_samples = int(round((tr.stats.starttime - t0) * fs))
		else:
			offset_samples = 0
		if offset_samples < 0:
			raise ValueError(
				'negative offset detected; check start_time_mode or align_on_start'
			)
		n = min(len(tr.data), total_samples - offset_samples)
		if n <= 0:
			continue
		data[i, offset_samples : offset_samples + n] = tr.data[:n].astype(
			np.float32, copy=False
		)

	return data, stations, fs, t_start_utc


def picks_to_sample_idx(
	stations: list[str],
	phs_df: pd.DataFrame,
	*,
	fs: float,
	t_start_utc: dt.datetime,
	dropna: bool = True,
	deduplicate: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
	if dropna:
		phs_df = phs_df.dropna(subset=['tp', 'ts']).copy()
	if deduplicate:
		phs_df = phs_df.sort_values('station').drop_duplicates(
			subset=['station'], keep='first'
		)
	phs_df = phs_df.set_index('station')
	p_idx = np.full(len(stations), np.nan, dtype=float)
	s_idx = np.full(len(stations), np.nan, dtype=float)

	t_start_utc = as_utc_aware(t_start_utc)

	for i, sta in enumerate(stations):
		if sta not in phs_df.index:
			continue
		tp = phs_df.loc[sta, 'tp']
		ts = phs_df.loc[sta, 'ts']

		tp_dt = as_utc_aware(tp.to_pydatetime())
		ts_dt = as_utc_aware(ts.to_pydatetime())

		p_idx[i] = (tp_dt - t_start_utc).total_seconds() * fs
		s_idx[i] = (ts_dt - t_start_utc).total_seconds() * fs

	return p_idx, s_idx
