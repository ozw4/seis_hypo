# %%
# file: proc/prepare_data/forge/run_eqt_pick.py
from __future__ import annotations

from pathlib import Path

from pipelines.das_eqt_pipelines import pipeline_das_eqt_pick_to_csv

# =========================
# Parameters (edit here)
# =========================
ZARR_DIR = Path('/home/dcuser/daseventnet/data/silixa/forge_dfit_block_78AB_250Hz.zarr')
ZARR_DATASET = 'block'

EQT_IN_SAMPLES = 6000
EQT_OVERLAP = 3000
EQT_WEIGHTS = 'original'
EQT_BATCH_TRACES = 64

# NOTE:
# channel_range is contiguous-only. For A/B discontiguous selection, use WELL_*_KEEP below.
CHANNEL_RANGE: tuple[int, int] | None = None

# ---- NEW: shrink to report-correct well ranges (0-based, inclusive) ----
APPLY_WELL_AB_KEEP = True
WELL_A_KEEP_0BASED_INCL = (92, 1062)
WELL_B_KEEP_0BASED_INCL = (1216, 2385)

# Peak detection / gating
DET_GATE_ENABLE = True
DET_THRESHOLD = 0.30
P_THRESHOLD = 0.10
S_THRESHOLD = 0.10
MIN_PICK_SEP_SAMPLES = 50

# How to merge probabilities on overlap between adjacent windows: 'max' or 'mean'
OVERLAP_MERGE = 'max'

OUT_CSV = Path('out/das_eqt_picks_woconvert.csv')

# Safety
MAX_WINDOWS: int | None = None
PRINT_EVERY_WINDOWS = 50

# ---- (if you already added conversion flags earlier, keep them as-is here) ----
CONVERT_STRAINRATE_TO_PSEUDOVEL = False
TARGET_FS_HZ = 100.0
POST_BP_LOW_HZ = 25.0
POST_BP_HIGH_HZ = 45.0
POST_BP_ORDER = 4
PSEUDOVEL_SCALE = 1.0
ZSCORE_PER_TRACE = True
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
		# ---- NEW ----
		apply_well_ab_keep=APPLY_WELL_AB_KEEP,
		well_a_keep_0based_incl=WELL_A_KEEP_0BASED_INCL,
		well_b_keep_0based_incl=WELL_B_KEEP_0BASED_INCL,
		# ---- conversion (if enabled) ----
		convert_strainrate_to_pseudovel=CONVERT_STRAINRATE_TO_PSEUDOVEL,
		target_fs_hz=TARGET_FS_HZ,
		post_bp_low_hz=POST_BP_LOW_HZ,
		post_bp_high_hz=POST_BP_HIGH_HZ,
		post_bp_order=POST_BP_ORDER,
		pseudovel_scale=PSEUDOVEL_SCALE,
		zscore_per_trace=ZSCORE_PER_TRACE,
	)


if __name__ == '__main__':
	main()
