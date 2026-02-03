from __future__ import annotations

import numpy as np


def _detect_local_peaks_2d(
	prob: np.ndarray,
	thr: float,
	min_sep: int,
	gate: np.ndarray | None = None,
	gate_threshold: float = 0.3,
) -> list[tuple[int, int, float]]:
	"""Detect local maxima (peaks) in a 2D probability map with optional gating and per-channel de-dup.

	This function is designed for phase probability outputs (e.g., EqTransformer P/S probabilities)
	organized as (channel, time). It returns candidate peak points that:
	  1) exceed a threshold `thr`,
	  2) are local maxima along the time axis,
	  3) optionally satisfy a gating mask (e.g., detection probability),
	  4) are pruned so that peaks within `min_sep` samples keep only the higher-probability one.

	Definition of "local peak":
	  - For interior samples (1..N-2): p[t] >= p[t-1] and p[t] >= p[t+1] and p[t] >= thr
	  - For edges:
	      t=0:     p[0] >= p[1]   and p[0] >= thr
	      t=N-1:   p[N-1] >= p[N-2] and p[N-1] >= thr
	  - Uses >= comparisons, so plateau-like peaks can yield multiple candidates; `min_sep` pruning
	    typically collapses near-duplicates.

	Gating:
	  - If `gate` is provided, it must have the same shape as `prob`.
	  - A peak candidate is accepted only where `gate >= DET_THRESHOLD` (global constant),
	    in addition to the local peak conditions above.

	De-duplication (`min_sep`):
	  - Applied independently within each channel.
	  - Candidates are sorted by time; if two are closer than `min_sep` samples, the higher-probability
	    one is kept (greedy).

	Args:
		prob:
			2D array of probabilities with shape (C, N). Any dtype is accepted; it is viewed as float32.
		thr:
			Probability threshold. Only points with prob >= thr can be peaks.
		min_sep:
			Minimum separation between peaks in samples, applied per channel.
			If min_sep <= 0, no pruning is performed.
		gate:
			Optional 2D gating probabilities with shape (C, N). If given, peaks are accepted only when
			gate >= gate_threshold .

	Returns:
		A list of (c_idx, t_idx, val) where:
		  - c_idx: channel index [0..C-1]
		  - t_idx: time sample index [0..N-1]
		  - val:   prob[c_idx, t_idx]
		The output is sorted in a stable-ish way by (channel, time).

	Raises:
		ValueError:
			- if prob is not 2D
			- if gate is provided but has a different shape than prob

	"""
	if prob.ndim != 2:
		raise ValueError(f'prob must be 2D (C,N), got {prob.shape}')
	C, N = prob.shape
	if N < 1:
		return []
	p = prob.astype(np.float32, copy=False)

	if gate is not None:
		if gate.shape != prob.shape:
			raise ValueError(
				f'gate shape mismatch: gate={gate.shape} prob={prob.shape}'
			)
		g = gate.astype(np.float32, copy=False)
		m_gate = g >= float(gate_threshold)
	else:
		m_gate = None

	peaks: list[tuple[int, int, float]] = []

	if N == 1:
		m = p[:, 0] >= float(thr)
		if m_gate is not None:
			m = m & m_gate[:, 0]
		idx = np.where(m)[0]
		for c in idx.tolist():
			peaks.append((int(c), 0, float(p[c, 0])))
		return peaks

	# edges
	left_ok = (p[:, 0] >= float(thr)) & (p[:, 0] >= p[:, 1])
	right_ok = (p[:, N - 1] >= float(thr)) & (p[:, N - 1] >= p[:, N - 2])
	if m_gate is not None:
		left_ok = left_ok & m_gate[:, 0]
		right_ok = right_ok & m_gate[:, N - 1]

	for c in np.where(left_ok)[0].tolist():
		peaks.append((int(c), 0, float(p[c, 0])))
	for c in np.where(right_ok)[0].tolist():
		peaks.append((int(c), N - 1, float(p[c, N - 1])))

	if N >= 3:
		mid = p[:, 1 : N - 1]
		is_pk = (mid >= p[:, 0 : N - 2]) & (mid >= p[:, 2:N]) & (mid >= float(thr))
		if m_gate is not None:
			is_pk = is_pk & m_gate[:, 1 : N - 1]

		c_idx, t_mid = np.where(is_pk)
		# np.where returns row-major; within each row, t is increasing
		for c, t in zip(c_idx.tolist(), t_mid.tolist(), strict=False):
			t_idx = int(t + 1)
			peaks.append((int(c), t_idx, float(p[c, t_idx])))

	if not peaks or int(min_sep) <= 0:
		return peaks

	# prune within each channel
	by_c: dict[int, list[tuple[int, float]]] = {}
	for c, t, v in peaks:
		by_c.setdefault(int(c), []).append((int(t), float(v)))

	out: list[tuple[int, int, float]] = []
	for c, tv in by_c.items():
		tv_sorted = sorted(tv, key=lambda x: x[0])
		kept: list[tuple[int, float]] = [tv_sorted[0]]
		for t, v in tv_sorted[1:]:
			t0, v0 = kept[-1]
			if t - t0 >= int(min_sep):
				kept.append((t, v))
			elif v > v0:
				kept[-1] = (t, v)
		for t, v in kept:
			out.append((int(c), int(t), float(v)))

	# stable-ish ordering (channel major, time)
	out.sort(key=lambda x: (x[0], x[1]))
	return out


def extract_pick_near_ref(
	score_1d: np.ndarray,
	ref_pick_idx: float,
	*,
	fs_hz: float,
	search_pre_sec: float,
	search_post_sec: float,
	thr: float,
	min_sep_sec: float,
	clip_search_window: bool = True,
	search_i1_inclusive: bool = True,
) -> dict[str, object]:
	"""Extract a single peak near a reference pick within a local search window."""
	fs_val = float(fs_hz)
	if not np.isfinite(fs_val) or fs_val <= 0.0:
		raise ValueError('fs_hz must be > 0')
	pre_sec = float(search_pre_sec)
	post_sec = float(search_post_sec)
	if not np.isfinite(pre_sec) or pre_sec < 0.0:
		raise ValueError('search_pre_sec must be >= 0')
	if not np.isfinite(post_sec) or post_sec < 0.0:
		raise ValueError('search_post_sec must be >= 0')

	score = np.asarray(score_1d, dtype=float)
	if score.ndim != 1:
		raise ValueError(f'score_1d must be 1D, got shape={score.shape}')

	n = int(score.shape[0])
	if n <= 0:
		return {
			'found_peak': False,
			'est_pick_idx': float('nan'),
			'score_at_pick': float('nan'),
			'n_peaks': 0,
			'search_i0': 0,
			'search_i1': (-1),
			'fail_reason': 'empty_score',
		}

	if not np.isfinite(score).any():
		return {
			'found_peak': False,
			'est_pick_idx': float('nan'),
			'score_at_pick': float('nan'),
			'n_peaks': 0,
			'search_i0': 0,
			'search_i1': -1,
			'fail_reason': 'nan_or_inf_score',
		}

	if not np.isfinite(float(ref_pick_idx)):
		return {
			'found_peak': False,
			'est_pick_idx': float('nan'),
			'score_at_pick': float('nan'),
			'n_peaks': 0,
			'search_i0': 0,
			'search_i1': (-1),
			'fail_reason': 'ref_pick_invalid',
		}

	ref_i = int(round(float(ref_pick_idx)))
	pre_n = int(round(pre_sec * fs_val))
	post_n = int(round(post_sec * fs_val))

	i0 = int(ref_i - pre_n)
	i1 = int(ref_i + post_n)
	if not search_i1_inclusive:
		i1 -= 1

	if clip_search_window:
		i0 = max(0, i0)
		i1 = min(n - 1, i1)
	elif i0 < 0 or i1 >= n:
		return {
			'found_peak': False,
			'est_pick_idx': float('nan'),
			'score_at_pick': float('nan'),
			'n_peaks': 0,
			'search_i0': int(i0),
			'search_i1': int(i1),
			'fail_reason': 'search_window_out_of_bounds',
		}

	if i0 > i1:
		return {
			'found_peak': False,
			'est_pick_idx': float('nan'),
			'score_at_pick': float('nan'),
			'n_peaks': 0,
			'search_i0': int(i0),
			'search_i1': int(i1),
			'fail_reason': 'empty_search_window',
		}

	win = score[i0 : i1 + 1]
	min_sep = max(1, int(round(float(min_sep_sec) * fs_val)))
	peaks = _detect_local_peaks_2d(win[None, :], thr=float(thr), min_sep=int(min_sep))
	n_peaks = len(peaks)
	if n_peaks <= 0:
		return {
			'found_peak': False,
			'est_pick_idx': float('nan'),
			'score_at_pick': float('nan'),
			'n_peaks': 0,
			'search_i0': int(i0),
			'search_i1': int(i1),
			'fail_reason': 'no_peak',
		}

	best = sorted(peaks, key=lambda x: (-float(x[2]), int(x[1])))[0]
	est_i = int(i0 + int(best[1]))
	return {
		'found_peak': True,
		'est_pick_idx': float(est_i),
		'score_at_pick': float(best[2]),
		'n_peaks': int(n_peaks),
		'search_i0': int(i0),
		'search_i1': int(i1),
		'fail_reason': '',
	}
