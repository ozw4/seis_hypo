# file: src/loki_tools/prob_stream.py
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from obspy import Stream, Trace


def build_loki_ps_prob_stream(
	ref_stream: Stream,
	*,
	probs_by_station: Mapping[str, Mapping[str, np.ndarray]],
	channel_prefix: str = 'HH',
	require_both_ps: bool = True,
) -> Stream:
	"""EqTransformer等の確率系列を、LOKIが direct_input で読める P/S 成分の Stream に変換する。

	LOKI(loki/waveforms.py) は comp=['P','S'] のとき、
	tr.stats.channel[-1] が 'P' / 'S' のトレースを拾うため、
	channel は 'HHP' / 'HHS' のように末尾が P/S になる必要がある。

	Parameters
	----------
	ref_stream:
		時刻基準（starttime, delta, npts）を取るための参照 Stream。
		通常は build_stream_from_downloaded_win32() から得たイベント窓の Stream。
	probs_by_station:
		station -> {'P': 1D array, 'S': 1D array, ...} の辞書。
		ここでの 'P' が LOKI の obs_dataV（vertical->P）に相当、
		'S' が obs_dataH（horizontal->S）に相当する。
	channel_prefix:
		出力 channel を f"{channel_prefix}P", f"{channel_prefix}S" にする。
		末尾が必ず P/S になるようにする（例: "HH" -> "HHP"/"HHS"）。
	require_both_ps:
		True のとき、各 station に P/S の両方が無いとエラー。

	Returns
	-------
	Stream:
		stationごとに 2本（P, S）を持つ Stream。starttime/delta は ref_stream に合わせる。

	"""
	if not ref_stream:
		raise ValueError('ref_stream must be non-empty')

	ref0 = ref_stream[0]
	starttime = ref0.stats.starttime
	delta = float(ref0.stats.delta)
	npts = int(ref0.stats.npts)

	# 参照Streamの整合性チェック（落ちるなら入力窓の作り方が壊れてる）
	for tr in ref_stream:
		if tr.stats.starttime != starttime:
			raise ValueError('ref_stream has inconsistent starttime across traces')
		if float(tr.stats.delta) != delta:
			raise ValueError('ref_stream has inconsistent delta across traces')
		if int(tr.stats.npts) != npts:
			raise ValueError('ref_stream has inconsistent npts across traces')

	out = Stream()
	net = getattr(ref0.stats, 'network', '')

	for sta, probs in probs_by_station.items():
		if not isinstance(probs, Mapping):
			raise ValueError(
				f'probs_by_station[{sta}] must be a mapping, got {type(probs)}'
			)

		p = probs.get('P')
		s = probs.get('S')

		if require_both_ps and (p is None or s is None):
			raise ValueError(f"station={sta} must have both 'P' and 'S' probs")

		for phase, arr in (('P', p), ('S', s)):
			if arr is None:
				continue

			a = np.asarray(arr)
			if a.ndim != 1:
				raise ValueError(
					f'station={sta} phase={phase} probs must be 1D, got shape={a.shape}'
				)
			if int(a.shape[0]) != npts:
				raise ValueError(
					f'station={sta} phase={phase} length mismatch: got={a.shape[0]} expected={npts}'
				)

			tr = Trace(data=a.astype(np.float32, copy=False))
			tr.stats.starttime = starttime
			tr.stats.delta = delta
			tr.stats.station = str(sta)
			tr.stats.channel = f'{channel_prefix}{phase}'  # 末尾が 'P' / 'S'
			if net:
				tr.stats.network = net
			out += tr

	return out
