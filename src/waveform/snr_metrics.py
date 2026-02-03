from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SNRSpec:
	"""SNR configuration (waveform around phase pick).

	Windows are relative to pick time (seconds), evaluated on the processed trace
	at fs_target_hz.

	SNR variants computed:
	- energy: sum(signal^2) / sum(noise^2)
	- rms: rms(signal) / rms(noise)
	- stalta: max(STA_power in signal window) / agg(LTA_power in noise window)
	"""

	fs_target_hz: int = 100

	noise_from_s: float = -3.0
	noise_to_s: float = -0.5
	signal_from_s: float = 0.0
	signal_to_s: float = 3.0

	# numerical eps for denominators
	eps_energy: float = 1e-12

	# STA/LTA (power) params
	sta_s: float = 0.3
	lta_s: float = 2.0
	lta_agg: str = 'median'  # "median" or "mean"


@dataclass(frozen=True)
class SNRMetrics:
	# window validity (fixed windows)
	ok_windows: bool
	reason: str

	# diagnostics on fixed windows
	noise_rms: float
	signal_rms: float
	noise_energy: float
	signal_energy: float

	# SNR variants (dB)
	snr_db_energy: float
	snr_db_rms: float
	snr_db_stalta: float

	# STA/LTA internals (power)
	sta_max_pow: float
	noise_lta_pow: float


def _idx(pick_idx: float, t_s: float, fs: float) -> int:
	return int(round(float(pick_idx) + float(t_s) * float(fs)))


def _safe_log10(x: float) -> float:
	return float(np.log10(max(float(x), 1e-300)))


def _rolling_mean_power(x: np.ndarray, win_n: int) -> np.ndarray:
	"""mean(x^2) over a sliding window (valid).
	Uses cumulative sum for stable / reasonably fast computation.
	"""
	x = np.asarray(x, dtype=float)
	if win_n <= 1:
		return x * x
	if x.size < win_n:
		return np.empty(0, dtype=float)

	x2 = x * x
	cs = np.cumsum(x2, dtype=float)
	cs = np.concatenate(([0.0], cs))
	return (cs[win_n:] - cs[:-win_n]) / float(win_n)


def compute_snr_metrics(
	x_proc: np.ndarray, *, pick_idx: float, fs: float, spec: SNRSpec
) -> SNRMetrics:
	"""Compute all SNR variants on a *processed* 1D trace (detrend+bandpass+resample done).

	- energy uses sum of squares in the fixed windows
	- rms uses RMS amplitude ratio in the fixed windows (20log10)
	- stalta uses rolling mean power:
	    signal: max over STA power inside signal window
	    noise: agg over LTA power inside noise window
	"""
	x = np.asarray(x_proc, dtype=float)
	if x.ndim != 1:
		return SNRMetrics(
			ok_windows=False,
			reason=f'bad_shape:{x.shape}',
			noise_rms=float('nan'),
			signal_rms=float('nan'),
			noise_energy=float('nan'),
			signal_energy=float('nan'),
			snr_db_energy=float('nan'),
			snr_db_rms=float('nan'),
			snr_db_stalta=float('nan'),
			sta_max_pow=float('nan'),
			noise_lta_pow=float('nan'),
		)

	n0 = _idx(pick_idx, spec.noise_from_s, fs)
	n1 = _idx(pick_idx, spec.noise_to_s, fs)
	s0 = _idx(pick_idx, spec.signal_from_s, fs)
	s1 = _idx(pick_idx, spec.signal_to_s, fs)

	if not (0 <= n0 < n1 <= x.size and 0 <= s0 < s1 <= x.size):
		return SNRMetrics(
			ok_windows=False,
			reason='windows_out_of_range',
			noise_rms=float('nan'),
			signal_rms=float('nan'),
			noise_energy=float('nan'),
			signal_energy=float('nan'),
			snr_db_energy=float('nan'),
			snr_db_rms=float('nan'),
			snr_db_stalta=float('nan'),
			sta_max_pow=float('nan'),
			noise_lta_pow=float('nan'),
		)

	noise = x[n0:n1]
	signal = x[s0:s1]

	# RMS (amplitude)
	noise_rms = float(np.sqrt(np.mean(noise * noise)))
	signal_rms = float(np.sqrt(np.mean(signal * signal)))

	# Energy via cumulative sum (explicitly, per request)
	# Equivalent to sum(noise^2), sum(signal^2), but computed cleanly.
	noise_energy = float(np.sum(noise * noise))
	signal_energy = float(np.sum(signal * signal))

	# energy SNR (power ratio) -> 10log10
	ratio_e = signal_energy / (noise_energy + float(spec.eps_energy))
	snr_db_energy = 10.0 * _safe_log10(ratio_e)

	# RMS SNR (amplitude ratio) -> 20log10
	ratio_r = signal_rms / (noise_rms + float(spec.eps_energy))
	snr_db_rms = 20.0 * _safe_log10(ratio_r)

	# STA/LTA SNR (power ratio) -> 10log10
	sta_s = float(spec.sta_s)
	lta_s = float(spec.lta_s)
	sta_n = max(int(round(sta_s * fs)), 2)
	lta_n = max(int(round(lta_s * fs)), 2)

	sta_max_pow = float('nan')
	noise_lta_pow = float('nan')
	snr_db_stalta = float('nan')

	# If insufficient window length, leave stalta as NaN (do not fail the whole row)
	if sta_n <= signal.size and lta_n <= noise.size:
		sta_pow = _rolling_mean_power(signal, sta_n)
		lta_pow = _rolling_mean_power(noise, lta_n)

		if sta_pow.size > 0 and lta_pow.size > 0:
			sta_max_pow = float(np.max(sta_pow))

			agg = str(spec.lta_agg).strip().lower()
			if agg == 'median':
				noise_lta_pow = float(np.median(lta_pow))
			elif agg == 'mean':
				noise_lta_pow = float(np.mean(lta_pow))
			else:
				raise ValueError(
					f"bad spec.lta_agg: {spec.lta_agg} (use 'median' or 'mean')"
				)

			ratio_s = sta_max_pow / (noise_lta_pow + float(spec.eps_energy))
			snr_db_stalta = 10.0 * _safe_log10(ratio_s)

	return SNRMetrics(
		ok_windows=True,
		reason='',
		noise_rms=noise_rms,
		signal_rms=signal_rms,
		noise_energy=noise_energy,
		signal_energy=signal_energy,
		snr_db_energy=snr_db_energy,
		snr_db_rms=snr_db_rms,
		snr_db_stalta=snr_db_stalta,
		sta_max_pow=sta_max_pow,
		noise_lta_pow=noise_lta_pow,
	)
