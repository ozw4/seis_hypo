from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from obspy import Stream

from common.time_util import as_utc_aware
from io_util.trace_util import trace_station_comp


def build_gather_matrix(
	st: Stream, comp: str
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
	t0 = min(start_times)
	t_start_utc = as_utc_aware(t0.datetime)

	# determine length to cover all traces
	end_times = [t.stats.endtime for _, t in trs]
	t1 = max(end_times)
	total_samples = int(round((t1 - t0) * fs)) + 1

	stations = [sta for sta, _ in trs]
	data = np.zeros((len(trs), total_samples), dtype=np.float32)

	for i, (_, tr) in enumerate(trs):
		offset_samples = int(round((tr.stats.starttime - t0) * fs))
		n = min(len(tr.data), total_samples - offset_samples)
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
) -> tuple[np.ndarray, np.ndarray]:
	phs_df = phs_df.dropna(subset=['tp', 'ts']).copy()
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
