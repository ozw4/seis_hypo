from __future__ import annotations

import numpy as np
import torch

from pick.eqt_probs import _get_eqt
from waveform.filters import zscore_tracewise


class EqTWindowRunner3C:
	"""Run EqTransformer on 3-component station windows."""

	def __init__(self, weights: str, in_samples: int, batch_stations: int):
		self.model = _get_eqt(str(weights), int(in_samples))
		self.device = next(self.model.parameters()).device
		self.in_samples = int(in_samples)
		self.batch_stations = int(batch_stations)

	@torch.no_grad()
	def predict_window(
		self, wave: np.ndarray
	) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""wave: (B, 3, L=in_samples) float32
		returns: (D, P, S) each (B, L) float32
		"""
		if wave.ndim != 3:
			raise ValueError(f'wave must be 3D (B,3,L), got {wave.shape}')

		wave_f32 = np.asarray(wave, dtype=np.float32)
		B, C, L = wave_f32.shape
		if int(C) != 3:
			raise ValueError(f'wave second axis must be 3, got {wave.shape}')
		if int(L) != int(self.in_samples):
			raise ValueError(
				f'wave length mismatch: got {L}, expected {self.in_samples}'
			)

		det = np.empty((B, L), dtype=np.float32)
		p = np.empty((B, L), dtype=np.float32)
		s = np.empty((B, L), dtype=np.float32)

		for i0 in range(0, B, self.batch_stations):
			i1 = min(i0 + self.batch_stations, B)
			x = torch.from_numpy(wave_f32[i0:i1, :, :]).to(self.device)
			x = zscore_tracewise(x, axis=-1, eps=1e-6)

			y_det, y_p, y_s = self.model(x)

			det[i0:i1, :] = y_det.detach().cpu().numpy().astype(np.float32, copy=False)
			p[i0:i1, :] = y_p.detach().cpu().numpy().astype(np.float32, copy=False)
			s[i0:i1, :] = y_s.detach().cpu().numpy().astype(np.float32, copy=False)

		return det, p, s
