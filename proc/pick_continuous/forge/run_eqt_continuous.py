# %%
# file: proc/prepare_data/forge/run_eqt_pick.py
from __future__ import annotations

from pathlib import Path

from pipelines.das_eqt_pipelines import pipeline_das_eqt_pick_to_csv

# =========================
# Parameters (edit here)
# =========================
ZARR_DIR = Path('/home/dcuser/daseventnet/data/silixa/raw_78B_block_ds10.zarr')
ZARR_DATASET = 'block'

EQT_IN_SAMPLES = 6000
EQT_OVERLAP = 3000
EQT_WEIGHTS = 'original'
EQT_BATCH_TRACES = 64

# None = all channels, or set (start, stop) for a contiguous subset
CHANNEL_RANGE: tuple[int, int] | None = None

# Peak detection / gating
DET_GATE_ENABLE = True
DET_THRESHOLD = 0.30
P_THRESHOLD = 0.10
S_THRESHOLD = 0.10
MIN_PICK_SEP_SAMPLES = 50

# How to merge probabilities on overlap between adjacent windows: 'max' or 'mean'
OVERLAP_MERGE = 'max'

OUT_CSV = Path('out/das_eqt_picks.csv')

# Safety
MAX_WINDOWS: int | None = None
PRINT_EVERY_WINDOWS = 50
# =========================


def main() -> None:
	pipeline_das_eqt_pick_to_csv(
		zarr_dir=ZARR_DIR,
		dataset_name=ZARR_DATASET,
		out_csv=OUT_CSV,
		eqt_weights=EQT_WEIGHTS,
		eqt_in_samples=EQT_IN_SAMPLES,
		eqt_overlap=EQT_OVERLAP,
		eqt_batch_traces=EQT_BATCH_TRACES,
		channel_range=CHANNEL_RANGE,
		det_gate_enable=DET_GATE_ENABLE,
		det_threshold=DET_THRESHOLD,
		p_threshold=P_THRESHOLD,
		s_threshold=S_THRESHOLD,
		min_pick_sep_samples=MIN_PICK_SEP_SAMPLES,
		overlap_merge=OVERLAP_MERGE,
		max_windows=MAX_WINDOWS,
		print_every_windows=PRINT_EVERY_WINDOWS,
	)


if __name__ == '__main__':
	main()
