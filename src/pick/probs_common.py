from __future__ import annotations

from typing import Any, Callable

import numpy as np


def normalize_zne(zne: np.ndarray) -> np.ndarray:
	"""Normalize input waveform into (3, N) ZNE layout."""
	if zne.ndim != 2:
		raise ValueError(f'zne must be 2D, got shape={zne.shape}')

	C, N = zne.shape
	if C != 3 and zne.shape[1] == 3:
		zne = zne.T
		C, N = zne.shape

	if C == 1:
		zne = np.vstack([zne, np.zeros((2, N), dtype=zne.dtype)])
		C, N = zne.shape

	if C != 3:
		raise ValueError(f'expected 3 components, got C={C} shape={zne.shape}')

	return zne


def iterate_overlapping_windows(
	zne: np.ndarray,
	*,
	window_len: int,
	hop_len: int,
	batch_size: int,
	to_tensor: Callable[[np.ndarray], Any],
	process_batch: Callable[[list[tuple[int, Any]]], None],
) -> None:
	"""Iterate over overlapping windows and dispatch to a batch processor."""
	N_eff = int(zne.shape[1])
	L = int(window_len)
	H = int(hop_len)
	buf: list[tuple[int, Any]] = []

	if N_eff < L:
		w = np.zeros((3, L), dtype=np.float32)
		w[:, :N_eff] = zne[:, :N_eff].astype(np.float32, copy=False)
		buf.append((0, to_tensor(w)))
		process_batch(buf)
	else:
		for s in range(0, N_eff - L + 1, H):
			w = zne[:, s : s + L].astype(np.float32, copy=False)
			buf.append((int(s), to_tensor(w)))
			if len(buf) >= int(batch_size):
				process_batch(buf)
		if buf:
			process_batch(buf)


def extract_station_probs(
	meta: dict[str, Any],
	sta: str,
	npts: int,
) -> dict[str, np.ndarray]:
	probs = meta.get('probs', None)
	if not isinstance(probs, dict):
		raise ValueError("meta['probs'] missing or invalid")

	p = probs.get('P', None)
	s = probs.get('S', None)
	if p is None or s is None:
		raise ValueError(f'missing P/S probs: station={sta}')

	p = np.asarray(p, dtype=np.float32)
	s = np.asarray(s, dtype=np.float32)
	if p.ndim != 1 or s.ndim != 1:
		raise ValueError(f'P/S probs must be 1D: station={sta}')
	if int(p.shape[0]) != npts or int(s.shape[0]) != npts:
		raise ValueError(
			f'P/S probs length mismatch: station={sta} got={(p.shape[0], s.shape[0])} expected={npts}'
		)

	return {'P': p, 'S': s}
