# file: src/pick/weights_util.py
from collections.abc import Mapping
from pathlib import Path

import torch


def _is_local_weights_spec(weights: str) -> bool:
	w = str(weights).strip()
	if not w:
		return False
	if w.startswith(('~', '/', './', '../')):
		return True
	if ('/' in w) or ('\\' in w):
		return True
	suf = Path(w).suffix.lower()
	return suf in {'.pt', '.pth', '.ckpt'}


def _extract_state_dict(obj: object) -> Mapping[str, torch.Tensor]:
	if isinstance(obj, Mapping):
		# 純粋な state_dict
		if obj and all(isinstance(k, str) for k in obj.keys()):
			vals = list(obj.values())
			if vals and all(torch.is_tensor(v) for v in vals):
				return obj  # type: ignore[return-value]

		# よくあるチェックポイント形式: {"state_dict": ...}
		if 'state_dict' in obj and isinstance(obj['state_dict'], Mapping):
			sd = obj['state_dict']
			if sd and all(isinstance(k, str) for k in sd.keys()):
				vals = list(sd.values())
				if vals and all(torch.is_tensor(v) for v in vals):
					return sd  # type: ignore[return-value]

	raise ValueError(
		'Invalid checkpoint format. Please save a pure state_dict (Mapping[str, Tensor]) '
		'or a dict with a "state_dict" key containing that mapping.'
	)
