"""Velocity-model loading helpers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from pathlib import Path

def load_velocity_json(path: Path) -> dict:
	"""Load and validate the velocity-model JSON."""
	vel = json.loads(path.read_text(encoding='utf-8'))
	for key in ['z', 'p', 's']:
		if key not in vel:
			raise ValueError(f"velocity json must have key '{key}'")
	return vel
