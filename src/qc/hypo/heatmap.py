from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hypo.synth_eval.heatmap_types import GridAxes


@dataclass(frozen=True)
class HeatmapSlicesConfig:
	xy_all_depths: bool
	xz_center_y: bool
	yz_center_x: bool


@dataclass(frozen=True)
class HeatmapScaleConfig:
	percentile: float
	global_across_slices: bool
	dz_symmetric: bool


@dataclass(frozen=True)
class HeatmapOutputConfig:
	save_npy: bool
	save_axes_json: bool
	out_dirname: str


@dataclass(frozen=True)
class HeatmapConfig:
	enabled: bool
	metrics: list[str]
	slices: HeatmapSlicesConfig
	scale: HeatmapScaleConfig
	output: HeatmapOutputConfig


@dataclass(frozen=True)
class HeatmapArtifacts:
	axes_json: Path | None
	metric_npy: dict[str, Path]
	metric_pngs: dict[str, dict[str, list[Path]]]


def load_grid_axes_from_index_csv(index_csv: Path) -> GridAxes:
	"""Build axes from index.csv x_m/y_m/z_m (unique + sorted)."""
	raise NotImplementedError


def map_true_xyz_to_zyx_indices(
	df_eval: pd.DataFrame,
	axes: GridAxes,
	*,
	x_col: str = 'x_m_true',
	y_col: str = 'y_m_true',
	z_col: str = 'z_m_true',
) -> np.ndarray:
	"""Return (N,3) int array of [iz, iy, ix] with exact-match mapping."""
	raise NotImplementedError


def build_metric_grid_zyx(
	indices_zyx: np.ndarray,
	values: np.ndarray,
	shape_zyx: tuple[int, int, int],
	*,
	agg: str = 'median',
) -> np.ndarray:
	"""Build grid[z,y,x], aggregate duplicates by nanmedian, keep missing as NaN."""
	raise NotImplementedError


def build_metric_grids_zyx(
	df_eval: pd.DataFrame,
	axes: GridAxes,
	metrics: list[str],
) -> dict[str, np.ndarray]:
	"""Build grid[z,y,x] per metric; raise if columns are missing."""
	raise NotImplementedError


def compute_vmin_vmax(
	metric: str,
	grid_zyx: np.ndarray,
	*,
	percentile: float,
	dz_symmetric: bool,
) -> tuple[float, float]:
	"""Compute vmin/vmax (p99; dz_m uses symmetric scale on |dz|)."""
	raise NotImplementedError


def write_axes_json(out_json: Path, axes: GridAxes) -> Path:
	"""Save axes.json with x_m/y_m/z_m + shape/order/center metadata."""
	raise NotImplementedError


def write_metric_grid_npy(out_npy: Path, grid_zyx: np.ndarray) -> Path:
	"""Save grid[z,y,x] as .npy and return path."""
	raise NotImplementedError


def run_heatmap_qc(
	*,
	df_eval: pd.DataFrame,
	dataset_dir: Path,
	run_dir: Path,
	cfg: HeatmapConfig,
) -> HeatmapArtifacts:
	"""Orchestrate heatmap QC: axes -> grids -> save npy/json -> render PNGs."""
	raise NotImplementedError
