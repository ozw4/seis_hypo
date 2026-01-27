# file: src/pipelines/das_phasenet_pipelines.py
from pathlib import Path

from pick.phasenet_runner import PhaseNetWindowRunner
from pipelines.das_pick_to_csv_common import (
	DasEqtPickStats,
	run_das_pick_to_csv_pipeline,
)


def pipeline_das_phasenet_pick_to_csv(
	*,
	zarr_dir: Path,
	dataset_name: str,
	out_csv: Path,
	phasenet_weights: str = 'instance',
	phasenet_in_samples: int = 3001,
	phasenet_overlap: int = 1500,
	phasenet_batch_traces: int = 256,
	channel_range: tuple[int, int] | None = None,
	p_threshold: float = 0.10,
	s_threshold: float = 0.10,
	min_pick_sep_samples: int = 50,
	overlap_merge: str = 'max',
	max_windows: int | None = None,
	print_every_windows: int = 50,
	# ---- discontiguous A/B keep ----
	apply_well_ab_keep: bool = False,
	well_a_keep_0based_incl: tuple[int, int] = (92, 1062),
	well_b_keep_0based_incl: tuple[int, int] = (1216, 2385),
	# ---- conversion / preprocess ----
	convert_strainrate_to_pseudovel: bool = False,
	target_fs_hz: float = 100.0,
	post_bp_low_hz: float = 25.0,
	post_bp_high_hz: float = 45.0,
	post_bp_order: int = 4,
	pseudovel_scale: float = 1.0,
) -> DasEqtPickStats:
	"""PhaseNet版（DAS連続→CSV）。
	注意: det_gate_enable は非対応（P/Sのみでピーク検出）。
	"""
	runner = PhaseNetWindowRunner(
		weights=str(phasenet_weights),
		in_samples=int(phasenet_in_samples),
		batch_traces=int(phasenet_batch_traces),
	)
	return run_das_pick_to_csv_pipeline(
		zarr_dir=Path(zarr_dir),
		dataset_name=str(dataset_name),
		out_csv=Path(out_csv),
		in_samples=int(phasenet_in_samples),
		overlap=int(phasenet_overlap),
		channel_range=channel_range,
		runner=runner,
		det_gate_enable=False,
		det_threshold=0.0,
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
		in_samples_label='phasenet_in_samples',
		overlap_label='phasenet_overlap',
		require_tb_multiple=True,
		use_det=False,
	)
