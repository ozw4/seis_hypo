from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from obspy import Stream


def station_zne_from_stream(
	st: Stream,
	*,
	vertical_alias: Iterable[str] = ('U', 'Z'),
	fill_missing_with_zeros: bool = True,
	log_label: str = 'Input',
) -> dict[str, np.ndarray]:
	"""event窓の波形Stream から station->(3,N) の ZNE配列（順序: U/Z, N, E）を作る。"""
	if not st:
		raise ValueError('empty stream')

	npts_ref = int(st[0].stats.npts)
	delta_ref = float(st[0].stats.delta)
	start_ref = st[0].stats.starttime

	vertical_alias_set = {str(x) for x in vertical_alias}
	label = str(log_label) if log_label else 'Input'

	by_sta: dict[str, dict[str, np.ndarray]] = {}
	for tr in st:
		if int(tr.stats.npts) != npts_ref:
			raise ValueError('inconsistent npts in stream')
		if float(tr.stats.delta) != delta_ref:
			raise ValueError('inconsistent delta in stream')
		if tr.stats.starttime != start_ref:
			raise ValueError('inconsistent starttime in stream')

		sta = getattr(tr.stats, 'station', None)
		cha = getattr(tr.stats, 'channel', None)
		if sta is None or cha is None:
			raise ValueError('trace.stats.station/channel missing')

		comp = str(cha)[-1]
		if comp in vertical_alias_set:
			comp = 'U'
		if comp not in ('U', 'N', 'E'):
			continue

		d = by_sta.setdefault(str(sta), {})
		if comp in d:
			raise ValueError(f'duplicate component: station={sta} comp={comp}')

		d[comp] = np.asarray(tr.data, dtype=np.float32)

	out: dict[str, np.ndarray] = {}
	n_fill = 0
	for sta, d in by_sta.items():
		missing = [c for c in ('U', 'N', 'E') if c not in d]
		if missing and not fill_missing_with_zeros:
			continue

		if 'U' not in d and 'N' not in d and 'E' not in d:
			continue

		if missing and fill_missing_with_zeros:
			n_fill += 1
			print(f'[WARN] {label} input: station={sta} missing={missing} -> fill zeros')

		u = d.get('U', np.zeros(npts_ref, dtype=np.float32))
		n = d.get('N', np.zeros(npts_ref, dtype=np.float32))
		e = d.get('E', np.zeros(npts_ref, dtype=np.float32))

		zne = np.vstack([u, n, e])
		out[sta] = zne

	if not out:
		raise ValueError(
			f'no station usable for {label} input (need at least 1 of U/Z/N/E)'
		)

	if n_fill > 0:
		print(f'[INFO] {label} input: filled missing components for {n_fill} stations')

	return out
