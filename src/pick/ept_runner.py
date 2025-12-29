from __future__ import annotations

import numpy as np
import torch

# Reuse your repo functions (same style as pipeline_loki_waveform_stacking_eqt)
from pick.eqt_probs import _get_eqt
from waveform.filters import zscore_tracewise


class EqTWindowRunner:
	"""Run EqTransformer on windowed DAS traces (each channel treated as one trace).
	Uses:
	  - pick.eqt_probs._get_eqt
	  - waveform.filters.zscore_tracewise
	"""

	def __init__(self, weights: str, in_samples: int, batch_traces: int):
		self.model = _get_eqt(str(weights), int(in_samples))
		self.device = next(self.model.parameters()).device
		self.in_samples = int(in_samples)
		self.batch_traces = int(batch_traces)

	@torch.no_grad()
	def predict_window(
		self, wave: np.ndarray
	) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""wave: (C, L=in_samples) float32
		returns: (D, P, S) each (C, L) float32
		"""
		if wave.ndim != 2:
			raise ValueError(f'wave must be 2D (C,L), got {wave.shape}')
		C, L = wave.shape
		if int(L) != int(self.in_samples):
			raise ValueError(
				f'wave length mismatch: got {L}, expected {self.in_samples}'
			)

		det = np.empty((C, L), dtype=np.float32)
		p = np.empty((C, L), dtype=np.float32)
		s = np.empty((C, L), dtype=np.float32)

		for i0 in range(0, C, self.batch_traces):
			i1 = min(i0 + self.batch_traces, C)
			w = torch.from_numpy(wave[i0:i1, :]).to(self.device)  # (B,L)

			# EqT expects 3 channels; DAS is 1 component so put it in U, keep others U.
			x = torch.zeros((i1 - i0, 3, L), device=self.device, dtype=w.dtype)
			x[:, 0, :] = w
			x[:, 1, :] = w
			x[:, 2, :] = w
			x = zscore_tracewise(x, axis=-1, eps=1e-6)

			y_det, y_p, y_s = self.model(x)

			det[i0:i1, :] = y_det.detach().cpu().numpy().astype(np.float32, copy=False)
			p[i0:i1, :] = y_p.detach().cpu().numpy().astype(np.float32, copy=False)
			s[i0:i1, :] = y_s.detach().cpu().numpy().astype(np.float32, copy=False)

		return det, p, s
