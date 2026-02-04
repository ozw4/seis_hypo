# file: src/pick/phasenet_labels.py
from __future__ import annotations


def labels_to_indices(labels: str) -> tuple[int, int, int]:
	lab = str(labels)
	if 'P' not in lab or 'S' not in lab:
		raise ValueError(
			f"PhaseNet labels must include 'P' and 'S', got labels={lab!r}"
		)
	idx_p = int(lab.index('P'))
	idx_s = int(lab.index('S'))
	idx_n = int(lab.index('N')) if 'N' in lab else -1
	return idx_n, idx_p, idx_s
