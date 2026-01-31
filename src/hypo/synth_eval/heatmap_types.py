from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridAxes:
	"""Grid axes for synthetic eval heatmaps (meters)."""

	x_m: np.ndarray
	y_m: np.ndarray
	z_m: np.ndarray

	def shape_zyx(self) -> tuple[int, int, int]:
		return (int(self.z_m.size), int(self.y_m.size), int(self.x_m.size))

	def center_x_index(self) -> int:
		return int(self.x_m.size // 2)

	def center_y_index(self) -> int:
		return int(self.y_m.size // 2)
