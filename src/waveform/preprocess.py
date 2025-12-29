# file: src/waveform/preprocess.py
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from obspy import Stream
from scipy.signal import detrend as sp_detrend
from scipy.signal import resample_poly

from common.config import LokiWaveformStackingInputs
from common.core import as_int_rate
from waveform.filters import bandpass_iir_filtfilt, mad_scale_1d


def spec_from_inputs(inputs: LokiWaveformStackingInputs) -> DetrendBandpassSpec:
	base = DetrendBandpassSpec()
	return DetrendBandpassSpec(
		detrend=getattr(inputs, 'pre_detrend', base.detrend),
		fstop_lo=float(getattr(inputs, 'pre_fstop_lo', base.fstop_lo)),
		fpass_lo=float(getattr(inputs, 'pre_fpass_lo', base.fpass_lo)),
		fpass_hi=float(getattr(inputs, 'pre_fpass_hi', base.fpass_hi)),
		fstop_hi=float(getattr(inputs, 'pre_fstop_hi', base.fstop_hi)),
		gpass=float(getattr(inputs, 'pre_gpass', base.gpass)),
		gstop=float(getattr(inputs, 'pre_gstop', base.gstop)),
		mad_scale=bool(getattr(inputs, 'pre_mad_scale', base.mad_scale)),
		mad_eps=float(getattr(inputs, 'pre_mad_eps', base.mad_eps)),
		mad_c=float(getattr(inputs, 'pre_mad_c', base.mad_c)),
	)


@dataclass(frozen=True)
class DetrendBandpassSpec:
	# detrend: 'constant' | 'linear' | None
	detrend: str | None = 'linear'

	# bandpass params（LokiWaveformStackingInputs の pre_* と一致）
	fstop_lo: float = 0.5
	fpass_lo: float = 1.0
	fpass_hi: float = 23.0
	fstop_hi: float = 25.0
	gpass: float = 1.0
	gstop: float = 40.0

	# robust scaling（LokiWaveformStackingInputs の pre_mad_* と一致）
	mad_scale: bool = True
	mad_eps: float = 1e-6
	mad_c: float = 1.4826

	# dtype
	out_dtype: np.dtype = np.float32


def _require_sampling_rate(tr, fs_expected: float | None) -> float:
	fs = float(tr.stats.sampling_rate)
	if fs_expected is not None and abs(fs - float(fs_expected)) > 1e-6:
		raise ValueError(
			f'sampling_rate mismatch: trace={tr.id} fs={fs} expected={fs_expected}'
		)
	return fs


def preprocess_stream_detrend_bandpass(
	st: Stream,
	*,
	spec: DetrendBandpassSpec = DetrendBandpassSpec(),
	fs_expected: float | None = None,
) -> Stream:
	"""LOKI投入前の前処理（detrend + bandpass + (optional) MAD scaling）を Stream に in-place 適用する。

	- build_stream_from_downloaded_win32() で揃えた sampling_rate を前提にする
	- fs_expected を渡すと全trace一致を厳密チェック
	"""
	if not isinstance(st, Stream):
		raise TypeError(f'st must be obspy.Stream, got {type(st)}')

	if len(st) == 0:
		raise ValueError('empty Stream')

	for tr in st:
		fs = _require_sampling_rate(tr, fs_expected)

		x = np.asarray(tr.data, dtype=float)

		# detrend
		if spec.detrend in ('constant', 'linear'):
			x = sp_detrend(x, type=spec.detrend)
		elif spec.detrend is None:
			pass
		else:
			raise ValueError("spec.detrend must be 'constant'|'linear'|None")

		# bandpass
		x = bandpass_iir_filtfilt(
			x,
			fs=float(fs),
			fstop_lo=spec.fstop_lo,
			fpass_lo=spec.fpass_lo,
			fpass_hi=spec.fpass_hi,
			fstop_hi=spec.fstop_hi,
			gpass=spec.gpass,
			gstop=spec.gstop,
		)

		# robust normalize (MAD)
		if spec.mad_scale:
			x = mad_scale_1d(x, eps=spec.mad_eps, c=spec.mad_c)

		tr.data = np.asarray(x, dtype=spec.out_dtype)

	return st


def strainrate_to_pseudovel(
	wave: np.ndarray,
	*,
	fs_in: float,
	pseudovel_scale: float = 1.0,
) -> np.ndarray:
	"""strain-rate -> pseudo-velocity（時間積分）。

	- サンプリング周波数と長さは保持する（resampleしない）
	- 正規化（zscore）はここでは行わない（EqT直前で統一）
	"""
	x = np.asarray(wave, dtype=float)
	if x.ndim != 2:
		raise ValueError(f'wave must be 2D (C,N), got shape={x.shape}')

	fi = as_int_rate(fs_in, 'fs_in')

	# DC寄りを落としてから積分（cumsum）してドリフトを抑える
	x = x - x.mean(axis=1, keepdims=True)
	v = np.cumsum(x, axis=1) * (float(pseudovel_scale) / float(fi))

	return np.asarray(v, dtype=np.float32)


def resample_window_poly(
	x: np.ndarray,
	*,
	fs_in: float,
	fs_out: float,
	out_len: int,
) -> np.ndarray:
	"""resample_polyで (C,N) を (C,out_len) にする。

	- resampleしない場合も out_len 一致を厳密チェック
	- フィルタや正規化は別段で実施
	"""
	w = np.asarray(x, dtype=float)
	if w.ndim != 2:
		raise ValueError(f'x must be 2D (C,N), got shape={w.shape}')

	fi = as_int_rate(fs_in, 'fs_in')
	fo = as_int_rate(fs_out, 'fs_out')

	if int(fi) != int(fo):
		r = Fraction(int(fo), int(fi))  # out/in
		up = int(r.numerator)
		down = int(r.denominator)
		if (int(w.shape[1]) * int(up)) % int(down) != 0:
			raise ValueError(
				f'resample length not integral: n={w.shape[1]} up={up} down={down}'
			)
		w = resample_poly(w, up=up, down=down, axis=1)

	if int(w.shape[1]) != int(out_len):
		raise ValueError(f'out_len mismatch: got={w.shape[1]} expected={out_len}')

	return np.asarray(w, dtype=np.float32)


def bandpass_window(
	x: np.ndarray,
	*,
	fs: float,
	post_bp_low_hz: float,
	post_bp_high_hz: float,
	post_bp_order: int,
) -> np.ndarray:
	"""bandpass（0-phase IIR）を (C,N) に適用する。

	- 正規化（zscore）はここでは行わない（EqT直前で統一）
	"""
	w = np.asarray(x, dtype=float)
	if w.ndim != 2:
		raise ValueError(f'x must be 2D (C,N), got shape={w.shape}')

	y = bandpass_iir_filtfilt(
		w,
		fs=float(fs),
		fstop_lo=float(post_bp_low_hz) * 0.8,
		fpass_lo=float(post_bp_low_hz),
		fpass_hi=float(post_bp_high_hz),
		fstop_hi=float(post_bp_high_hz) * 1.2,
	)

	_ = int(post_bp_order)  # placeholder: kept for signature compatibility

	return np.asarray(y, dtype=np.float32)
