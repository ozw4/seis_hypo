# EqT backend（あなたが src/pick などに追加した想定）
# backend_eqt_probs(x_3cn, fs, weights=..., in_samples=..., overlap=..., batch_size=...)
from __future__ import annotations

import numpy as np
from obspy import Stream


def station_zne_from_stream(
	st: Stream,
	*,
	vertical_alias: Iterable[str] = ('U', 'Z'),
	fill_missing_with_zeros: bool = True,
) -> dict[str, np.ndarray]:
	"""event窓の波形Stream から station->(3,N) の ZNE配列（順序: U/Z, N, E）を作る。

	- channel末尾が U/N/E または Z/N/E を想定（UとZは鉛直として同一扱い）
	- 3成分が欠ける局がある場合:
	  - fill_missing_with_zeros=True なら不足成分を0埋めしつつ WARN を出す
	  - False なら不足局は捨てる（従来挙動に近い）
	"""
	if not st:
		raise ValueError('empty stream')

	npts_ref = int(st[0].stats.npts)
	delta_ref = float(st[0].stats.delta)
	start_ref = st[0].stats.starttime

	vertical_alias_set = {str(x) for x in vertical_alias}

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
			print(f'[WARN] EqT input: station={sta} missing={missing} -> fill zeros')

		u = d.get('U', np.zeros(npts_ref, dtype=np.float32))
		n = d.get('N', np.zeros(npts_ref, dtype=np.float32))
		e = d.get('E', np.zeros(npts_ref, dtype=np.float32))

		zne = np.vstack([u, n, e])
		out[sta] = zne

	if not out:
		# ここで落ちる場合は、そもそも stream に U/Z/N/E の末尾を持つ trace が無いか、
		# station が全部欠損で fill_missing_with_zeros=False の可能性が高い
		raise ValueError('no station usable for EqT (need at least 1 of U/Z/N/E)')

	if n_fill > 0:
		print(f'[INFO] EqT input: filled missing components for {n_fill} stations')

	return out
