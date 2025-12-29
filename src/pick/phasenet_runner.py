# file: src/pick/phasenet_runner.py
from __future__ import annotations

import numpy as np
import torch

from pick.phasenet_probs import _get_phasenet


def _labels_to_indices(labels: str) -> tuple[int, int]:
	lab = str(labels)
	if 'P' not in lab or 'S' not in lab:
		raise ValueError(
			f"PhaseNet labels must include 'P' and 'S', got labels={lab!r}"
		)
	return int(lab.index('P')), int(lab.index('S'))


class PhaseNetWindowRunner:
	"""Run PhaseNet on windowed DAS traces (each channel treated as one trace).

	Note:
	  - det_gate_enable is not supported at pipeline-level (enforce elsewhere).
	  - This runner returns a dummy det array of ones for signature compatibility.

	"""

	def __init__(self, weights: str, in_samples: int, batch_traces: int):
		self.model = _get_phasenet(str(weights), int(in_samples))
		self.device = next(self.model.parameters()).device
		self.in_samples = int(in_samples)
		self.batch_traces = int(batch_traces)

		self.idx_p, self.idx_s = _labels_to_indices(
			getattr(self.model, 'labels', 'NPS')
		)

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

		wave_f32 = np.asarray(wave, dtype=np.float32)

		det = np.ones((C, L), dtype=np.float32)
		p = np.empty((C, L), dtype=np.float32)
		s = np.empty((C, L), dtype=np.float32)

		for i0 in range(0, C, self.batch_traces):
			i1 = min(i0 + self.batch_traces, C)

			w = torch.from_numpy(wave_f32[i0:i1, :]).to(self.device)  # (B,L)

			# PhaseNet expects 3 components; DAS is 1 component so replicate to 3ch.
			x = torch.zeros((i1 - i0, 3, L), device=self.device, dtype=w.dtype)
			x[:, 0, :] = w
			x[:, 1, :] = w
			x[:, 2, :] = w
			# Use PhaseNet's own preprocess hook for consistency with SeisBench.
			x = self.model.annotate_batch_pre(x, {})

			y = self.model(x)  # (B, K, L)
			y_np = y.detach().cpu().numpy()

			p[i0:i1, :] = y_np[:, self.idx_p, :].astype(np.float32, copy=False)
			s[i0:i1, :] = y_np[:, self.idx_s, :].astype(np.float32, copy=False)

		return det, p, s
