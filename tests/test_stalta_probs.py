from __future__ import annotations

import numpy as np
from obspy import Stream, Trace, UTCDateTime

from pick.stalta_probs import StaltaProbSpec, build_probs_by_station_stalta


def test_build_probs_by_station_stalta_flat_cf_returns_zeros() -> None:
	fs = 100.0
	npts = 1000
	x = np.zeros(npts, dtype=np.float32)
	tr = Trace(data=x)
	tr.stats.station = 'STA1'
	tr.stats.channel = 'HHU'
	tr.stats.delta = 1.0 / fs
	tr.stats.starttime = UTCDateTime(0)
	st = Stream(traces=[tr])

	spec = StaltaProbSpec(transform='raw', sta_sec=0.2, lta_sec=2.0)
	probs = build_probs_by_station_stalta(
		st,
		fs=fs,
		component='U',
		phase='P',
		spec=spec,
	)

	assert 'STA1' in probs
	assert 'P' in probs['STA1']
	arr = probs['STA1']['P']
	assert arr.dtype == np.float32
	assert arr.shape == (npts,)
	assert np.all(arr == 0.0)
