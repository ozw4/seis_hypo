# file: src/pipelines/das_eqt_pipelines.py
from pathlib import Path

from pick.ept_runner import EqTWindowRunner
from pipelines.das_pick_to_csv_common import (
	DasEqtPickStats,
	run_das_pick_to_csv_pipeline,
)

# (If you already added scipy-based conversion helpers earlier, keep them in this file.
#  The changes below are only about A/B shrinking + channel-id mapping.)


def pipeline_das_eqt_pick_to_csv(
	*,
	zarr_dir: Path,
	dataset_name: str,
	out_csv: Path,
	eqt_weights: str = 'original',
	eqt_in_samples: int = 6000,
	eqt_overlap: int = 3000,
	eqt_batch_traces: int = 64,
	channel_range: tuple[int, int] | None = None,
	det_gate_enable: bool = True,
	det_threshold: float = 0.30,
	p_threshold: float = 0.10,
	s_threshold: float = 0.10,
	min_pick_sep_samples: int = 50,
	overlap_merge: str = 'max',
	max_windows: int | None = None,
	print_every_windows: int = 50,
	# ---- NEW: discontiguous A/B keep ----
	apply_well_ab_keep: bool = False,
	well_a_keep_0based_incl: tuple[int, int] = (92, 1062),
	well_b_keep_0based_incl: tuple[int, int] = (1216, 2385),
	# ---- (keep your conversion args here if already added) ----
	convert_strainrate_to_pseudovel: bool = False,
	target_fs_hz: float = 100.0,
	post_bp_low_hz: float = 25.0,
	post_bp_high_hz: float = 45.0,
	post_bp_order: int = 4,
	pseudovel_scale: float = 1.0,
) -> DasEqtPickStats:
	runner = EqTWindowRunner(
		weights=str(eqt_weights),
		in_samples=int(eqt_in_samples),
		batch_traces=int(eqt_batch_traces),
	)
	return run_das_pick_to_csv_pipeline(
		zarr_dir=Path(zarr_dir),
		dataset_name=str(dataset_name),
		out_csv=Path(out_csv),
		in_samples=int(eqt_in_samples),
		overlap=int(eqt_overlap),
		channel_range=channel_range,
		runner=runner,
		det_gate_enable=bool(det_gate_enable),
		det_threshold=float(det_threshold),
		p_threshold=float(p_threshold),
		s_threshold=float(s_threshold),
		min_pick_sep_samples=int(min_pick_sep_samples),
		overlap_merge=str(overlap_merge),
		max_windows=max_windows,
		print_every_windows=int(print_every_windows),
		apply_well_ab_keep=bool(apply_well_ab_keep),
		well_a_keep_0based_incl=tuple(well_a_keep_0based_incl),
		well_b_keep_0based_incl=tuple(well_b_keep_0based_incl),
		convert_strainrate_to_pseudovel=bool(convert_strainrate_to_pseudovel),
		target_fs_hz=float(target_fs_hz),
		post_bp_low_hz=float(post_bp_low_hz),
		post_bp_high_hz=float(post_bp_high_hz),
		post_bp_order=int(post_bp_order),
		pseudovel_scale=float(pseudovel_scale),
		in_samples_label='eqt_in_samples',
		overlap_label='eqt_overlap',
		require_tb_multiple=False,
		use_det=True,
	)
