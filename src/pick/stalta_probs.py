# src/picks/stalta_probs.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from obspy import Stream, Trace
from seisai_pick import stalta as seisai_stalta

from common.array_util import as_1d_float
from waveform.filters import percentile_clip, smooth_ma_same
from waveform.transforms import abs_1d, envelope_1d

Phase = Literal['P', 'S']
TransformName = Literal['raw', 'abs', 'envelope']


def _sec_to_samples(fs: float, sec: float) -> int:
	if fs <= 0:
		raise ValueError('fs must be > 0')
	if sec <= 0:
		raise ValueError('sec must be > 0')
	n = int(round(sec * fs))
	n = max(n, 1)
	return n


def _sec_to_odd_samples(fs: float, sec: float) -> int:
	n = _sec_to_samples(fs, sec)
	if n % 2 == 0:
		n += 1
	return n


def _apply_transform(x: np.ndarray, *, name: TransformName) -> np.ndarray:
	a = as_1d_float(x)
	if name == 'raw':
		return a
	if name == 'abs':
		return abs_1d(a)
	if name == 'envelope':
		return envelope_1d(a)
	raise ValueError(f'unsupported transform={name!r}')


def _select_one_trace_per_station(st: Stream, *, component: str) -> dict[str, Trace]:
	if not st:
		raise ValueError('stream must be non-empty')

	comp = str(component)
	if len(comp) != 1:
		raise ValueError(f'component must be 1 char like U/N/E, got {component!r}')

	out: dict[str, Trace] = {}
	for tr in st:
		sta = getattr(tr.stats, 'station', None)
		ch = getattr(tr.stats, 'channel', None)
		if sta is None or ch is None:
			continue
		ch_s = str(ch)
		if not ch_s:
			continue
		if ch_s[-1] != comp:
			continue

		sta_s = str(sta)
		if sta_s in out:
			raise ValueError(f'duplicate trace for station={sta_s} component={comp}')
		out[sta_s] = tr

	if not out:
		raise ValueError(f'no traces matched component={comp}')
	return out


def _scale_0_1_strict(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
	a = as_1d_float(x)
	mn = float(np.min(a))
	mx = float(np.max(a))
	den = mx - mn
	if den <= eps:
		raise ValueError(f'cannot scale to [0,1]: range too small (min={mn}, max={mx})')
	return (a - mn) / den


@dataclass(frozen=True)
class StaltaProbSpec:
	"""1成分から LOKI direct_input 用の重み系列（0..1）を作る設定。

	注意:
	  seisai.pick.stalta は内部で x^2 を使う（逐次和）ので、
	  transform="square" みたいな事前二乗は絶対に入れない（x^4 になる）。
	"""

	transform: TransformName = 'raw'
	sta_sec: float = 0.2
	lta_sec: float = 2.0
	smooth_sec: float | None = None  # 入力（transform後）を平滑化してからSTALTA
	clip_p: float | None = 99.5  # STALTA出力を上側クリップ
	log1p: bool = False  # STALTA出力（非負）に log1p
	eps: float = 1e-12


def build_probs_by_station_stalta(
	ref_stream: Stream,
	*,
	fs: float,
	component: str = 'U',
	phase: Phase = 'P',
	spec: StaltaProbSpec | None = None,
) -> dict[str, dict[str, np.ndarray]]:
	"""EqTの代わりに、seisai.pick.stalta ベースで probs_by_station を作る（1フェーズだけ埋める）。

	Returns:
	  station -> {phase: np.ndarray} 形式。配列長は npts と一致。

	"""
	if spec is None:
		spec = StaltaProbSpec()

	if fs <= 0:
		raise ValueError('fs must be > 0')

	phase_s = str(phase)
	if phase_s not in ('P', 'S'):
		raise ValueError(f'phase must be P or S, got {phase!r}')

	tr_by_sta = _select_one_trace_per_station(ref_stream, component=component)

	any_tr = next(iter(tr_by_sta.values()))
	npts = int(any_tr.stats.npts)
	delta = float(any_tr.stats.delta)
	starttime = any_tr.stats.starttime

	fs_from_delta = 1.0 / delta
	if abs(fs_from_delta - float(fs)) > 1e-6:
		raise ValueError(f'fs mismatch: fs={fs} but 1/delta={fs_from_delta}')

	for tr in tr_by_sta.values():
		if int(tr.stats.npts) != npts:
			raise ValueError('inconsistent npts across selected traces')
		if float(tr.stats.delta) != delta:
			raise ValueError('inconsistent delta across selected traces')
		if tr.stats.starttime != starttime:
			raise ValueError('inconsistent starttime across selected traces')

	ns = _sec_to_samples(float(fs), float(spec.sta_sec))
	nl = _sec_to_samples(float(fs), float(spec.lta_sec))
	if nl < ns:
		raise ValueError(f'require lta_sec >= sta_sec (ns={ns}, nl={nl})')
	if nl >= npts:
		raise ValueError(f'lta window too long for trace length: nl={nl} npts={npts}')

	out: dict[str, dict[str, np.ndarray]] = {}

	for sta, tr in tr_by_sta.items():
		x = as_1d_float(np.asarray(tr.data), name=f'{sta}.data')
		y = _apply_transform(x, name=spec.transform)

		if spec.smooth_sec is not None:
			win = _sec_to_odd_samples(float(fs), float(spec.smooth_sec))
			y = smooth_ma_same(y, win)

		cf = seisai_stalta.stalta(
			y, ns=ns, nl=nl, eps=float(spec.eps), axis=-1, out_dtype=np.float64
		)

		# LTAが安定する前は、検出としては邪魔になりやすいのでゼロ
		if nl > 1:
			cf[: nl - 1] = 0.0

		if spec.clip_p is not None:
			cf = percentile_clip(cf, p=float(spec.clip_p))

		if spec.log1p:
			if np.any(cf < 0):
				mn = float(np.min(cf))
				raise ValueError(
					f'log1p requires nonnegative cf, found min={mn} at station={sta}'
				)
			cf = np.log1p(cf)

		cf01 = _scale_0_1_strict(cf, eps=float(spec.eps))
		out[sta] = {phase_s: cf01.astype(np.float32, copy=False)}

	return out


def build_probs_by_station_stalta_ps(
	ref_stream: Stream,
	*,
	fs: float,
	component: str = 'U',
	p_spec: StaltaProbSpec | None = None,
	s_spec: StaltaProbSpec | None = None,
) -> dict[str, dict[str, np.ndarray]]:
	"""P/S両方をまとめて作る。

	例（おすすめの初期値）:
	  P: transform="raw" or "abs"
	  S: transform="envelope"
	"""
	if p_spec is None:
		p_spec = StaltaProbSpec(transform='raw')
	if s_spec is None:
		s_spec = StaltaProbSpec(transform='envelope')

	p = build_probs_by_station_stalta(
		ref_stream,
		fs=fs,
		component=component,
		phase='P',
		spec=p_spec,
	)
	s = build_probs_by_station_stalta(
		ref_stream,
		fs=fs,
		component=component,
		phase='S',
		spec=s_spec,
	)

	out: dict[str, dict[str, np.ndarray]] = {}
	for sta in sorted(set(p.keys()) | set(s.keys())):
		d: dict[str, np.ndarray] = {}
		if sta in p and 'P' in p[sta]:
			d['P'] = p[sta]['P']
		if sta in s and 'S' in s[sta]:
			d['S'] = s[sta]['S']
		out[sta] = d

	return out


__all__ = [
	'StaltaProbSpec',
	'build_probs_by_station_stalta',
	'build_probs_by_station_stalta_ps',
]
