from __future__ import annotations

from typing import Union

import numpy as np
import scipy.signal as sp_signal
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


def _as_2d_float(data: np.ndarray) -> tuple[np.ndarray, bool]:
	x = np.asarray(data, dtype=float)
	one = x.ndim == 1
	if one:
		x = x[None, :]
	if x.ndim != 2:
		raise ValueError(f'data must be 1D or 2D, got shape={x.shape}')
	return x, one


def _require_min_len(x2d: np.ndarray, min_len: int = 8) -> None:
	n = int(x2d.shape[1])
	if n < min_len:
		raise ValueError(f'trace too short for filtering: n={n} < {min_len}')


def _safe_padlen(a: np.ndarray, b: np.ndarray, n: int) -> int:
	# filtfilt: padlen < n
	raw = 3 * (max(len(a), len(b)) - 1)
	return max(0, min(raw, n - 1))


def lowcut_iir_filtfilt(
	data: np.ndarray,
	fs: float,
	*,
	fstop: float = 5.0,
	fpass: float = 10.0,
	gpass: float = 1.0,
	gstop: float = 40.0,
) -> np.ndarray:
	"""Zero-phase IIR high-pass via filtfilt. data: (N,) or (C,N)."""
	x, one = _as_2d_float(data)
	_require_min_len(x)

	nyq = float(fs) / 2.0
	Wp, Ws = float(fpass) / nyq, float(fstop) / nyq
	if not (0.0 < Ws < Wp < 1.0):
		raise ValueError('Need 0 < fstop < fpass < Nyquist.')

	b, a = sp_signal.iirdesign(wp=Wp, ws=Ws, gpass=gpass, gstop=gstop, ftype='ellip')
	padlen = _safe_padlen(a, b, x.shape[1])
	y = sp_signal.filtfilt(b, a, x, axis=1, padlen=padlen)
	return y[0] if one else y


def bandpass_iir_filtfilt(
	data: np.ndarray,
	fs: float,
	*,
	fstop_lo: float = 0.5,
	fpass_lo: float = 1.0,
	fpass_hi: float = 20.0,
	fstop_hi: float = 30.0,
	gpass: float = 1.0,
	gstop: float = 40.0,
) -> np.ndarray:
	"""Zero-phase IIR band-pass via filtfilt. data: (N,) or (C,N).
	条件: 0 < fstop_lo < fpass_lo < fpass_hi < fstop_hi < fs/2
	"""
	x, one = _as_2d_float(data)
	_require_min_len(x)

	nyq = float(fs) / 2.0
	fstop_hi2 = min(float(fstop_hi), 0.9 * nyq)
	fpass_hi2 = min(float(fpass_hi), 0.85 * nyq)

	Wp = [float(fpass_lo) / nyq, fpass_hi2 / nyq]
	Ws = [float(fstop_lo) / nyq, fstop_hi2 / nyq]
	if not (0.0 < Ws[0] < Wp[0] < Wp[1] < Ws[1] < 1.0):
		raise ValueError(
			'Need 0 < fstop_lo < fpass_lo < fpass_hi < fstop_hi < Nyquist.'
		)

	b, a = sp_signal.iirdesign(wp=Wp, ws=Ws, gpass=gpass, gstop=gstop, ftype='ellip')
	padlen = _safe_padlen(a, b, x.shape[1])
	y = sp_signal.filtfilt(b, a, x, axis=1, padlen=padlen)
	return y[0] if one else y


def percentile_clip(x: np.ndarray, p: float = 99.5) -> np.ndarray:
	"""全要素でパーセンタイルクリップ（チャンネル別にしたいなら別関数にするのがおすすめ）"""
	a = np.asarray(x, dtype=float)
	lo = float(np.percentile(a, 100.0 - p))
	hi = float(np.percentile(a, p))
	if hi <= lo:
		return a
	return np.clip(a, lo, hi)


def smooth_ma_same(x: np.ndarray, win: int) -> np.ndarray:
	"""中心合わせ（same）の移動平均。xは1Dのみ。"""
	a = np.asarray(x, dtype=float)
	if a.ndim != 1:
		raise ValueError(f'smooth_ma_same expects 1D, got shape={a.shape}')
	if win is None or int(win) <= 1:
		return a
	w = int(win)
	kernel = np.ones(w, dtype=float) / float(w)
	return np.convolve(a, kernel, mode='same')


def zscore_channelwise(x: ArrayLike, axis: int = -1, eps: float = 1e-6) -> ArrayLike:
	"""チャネル毎に z-score 正規化（平均0・標準偏差1）"""
	if isinstance(x, torch.Tensor):
		m = x.mean(dim=axis, keepdim=True)
		# torchはunbiasedでズレが出るので明示的に寄せる
		s = x.std(dim=axis, keepdim=True, unbiased=False).clamp_min(eps)
		return (x - m) / s

	if isinstance(x, np.ndarray):
		m = x.mean(axis=axis, keepdims=True)
		s = x.std(axis=axis, keepdims=True)
		s = np.maximum(s, eps)
		return (x - m) / s

	raise TypeError(f'Unsupported type for zscore_channelwise: {type(x)}')


def mad_scale_1d(x: np.ndarray, *, eps: float = 1e-6, c: float = 1.4826) -> np.ndarray:
	"""Robust scaling using MAD.
	y = (x - median(x)) / (c * MAD(x) + eps)
	c=1.4826 makes MAD consistent with std for normal dist.
	"""
	x = np.asarray(x, dtype=float)
	med = float(np.median(x))
	mad = float(np.median(np.abs(x - med)))
	den = c * mad + float(eps)
	return (x - med) / den
