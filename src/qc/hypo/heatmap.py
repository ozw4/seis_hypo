from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hypo.synth_eval.heatmap_scale import compute_vmin_vmax
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
	vmin: float | None = None
	vmax: float | None = None


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


@dataclass(frozen=True)
class HeatmapMetricDisplay:
	display_name: str
	colorbar_label: str
	fixed_range: tuple[float, float] | None = None


def get_heatmap_metric_display(metric: str) -> HeatmapMetricDisplay:
	"""Return plot display metadata for a heatmap metric."""
	if metric == 'GAP':
		return HeatmapMetricDisplay(
			display_name='Azimuthal GAP',
			colorbar_label='GAP (deg)',
			fixed_range=(0.0, 360.0),
		)
	return HeatmapMetricDisplay(
		display_name=metric,
		colorbar_label=metric,
	)


def load_grid_axes_from_index_csv(index_csv: Path) -> GridAxes:
	"""Build axes from index.csv x_m/y_m/z_m (unique + sorted)."""
	if not index_csv.is_file():
		raise FileNotFoundError(f'missing: {index_csv}')

	df = pd.read_csv(index_csv)
	required = ['x_m', 'y_m', 'z_m']
	missing = [c for c in required if c not in df.columns]
	if missing:
		raise ValueError(f'index.csv missing columns: {missing}')

	x_vals = _require_int_values(df, 'x_m')
	y_vals = _require_int_values(df, 'y_m')
	z_vals = _require_int_values(df, 'z_m')

	x_axis = np.unique(x_vals)
	y_axis = np.unique(y_vals)
	z_axis = np.unique(z_vals)

	if x_axis.size == 0 or y_axis.size == 0 or z_axis.size == 0:
		raise ValueError('index.csv has empty axis values')

	return GridAxes(x_m=x_axis, y_m=y_axis, z_m=z_axis)


def map_true_xyz_to_zyx_indices(
	df_eval: pd.DataFrame,
	axes: GridAxes,
	*,
	x_col: str = 'x_m_true',
	y_col: str = 'y_m_true',
	z_col: str = 'z_m_true',
) -> np.ndarray:
	"""Return (N,3) int array of [iz, iy, ix] with exact-match mapping."""
	required = [x_col, y_col, z_col]
	missing = [c for c in required if c not in df_eval.columns]
	if missing:
		raise ValueError(f'eval_metrics.csv missing columns: {missing}')

	x_vals = _require_int_values(df_eval, x_col)
	y_vals = _require_int_values(df_eval, y_col)
	z_vals = _require_int_values(df_eval, z_col)

	x_map = _build_axis_index(axes.x_m, 'x_m')
	y_map = _build_axis_index(axes.y_m, 'y_m')
	z_map = _build_axis_index(axes.z_m, 'z_m')

	n = x_vals.size
	indices = np.empty((n, 3), dtype=int)

	for i in range(n):
		xv = int(x_vals[i])
		yv = int(y_vals[i])
		zv = int(z_vals[i])

		if xv not in x_map or yv not in y_map or zv not in z_map:
			raise ValueError(f'coordinate not on grid axes: (x={xv}, y={yv}, z={zv})')

		indices[i, 0] = z_map[zv]
		indices[i, 1] = y_map[yv]
		indices[i, 2] = x_map[xv]

	return indices


def build_metric_grid_zyx(
	indices_zyx: np.ndarray,
	values: np.ndarray,
	shape_zyx: tuple[int, int, int],
	*,
	agg: str = 'median',
) -> np.ndarray:
	"""Build grid[z,y,x], aggregate duplicates by nanmedian, keep missing as NaN."""
	if agg != 'median':
		raise ValueError(f'unsupported agg: {agg}')

	idx = np.asarray(indices_zyx)
	if idx.ndim != 2 or idx.shape[1] != 3:
		raise ValueError(f'indices_zyx must be (N,3), got {idx.shape}')

	vals = np.asarray(values, dtype=float).reshape(-1)
	if idx.shape[0] != vals.size:
		raise ValueError(
			f'indices_zyx and values length mismatch: {idx.shape[0]} vs {vals.size}'
		)

	nz, ny, nx = (int(shape_zyx[0]), int(shape_zyx[1]), int(shape_zyx[2]))
	if nz <= 0 or ny <= 0 or nx <= 0:
		raise ValueError(f'invalid grid shape: {shape_zyx}')

	grid = np.full((nz, ny, nx), np.nan, dtype=float)

	cell_values: dict[tuple[int, int, int], list[float]] = {}
	for (iz, iy, ix), v in zip(idx, vals):
		iz_i, iy_i, ix_i = int(iz), int(iy), int(ix)
		if iz_i < 0 or iz_i >= nz or iy_i < 0 or iy_i >= ny or ix_i < 0 or ix_i >= nx:
			raise IndexError(
				'indices_zyx out of range: '
				f'({iz_i}, {iy_i}, {ix_i}) for shape {shape_zyx}'
			)
		cell_values.setdefault((iz_i, iy_i, ix_i), []).append(float(v))

	for (iz, iy, ix), vals_cell in cell_values.items():
		arr = np.asarray(vals_cell, dtype=float)
		if arr.size == 0 or not np.isfinite(arr).any():
			grid[iz, iy, ix] = np.nan
		else:
			grid[iz, iy, ix] = float(np.nanmedian(arr))

	return grid


def build_metric_grids_zyx(
	df_eval: pd.DataFrame,
	axes: GridAxes,
	metrics: list[str],
) -> dict[str, np.ndarray]:
	"""Build grid[z,y,x] per metric; raise if columns are missing."""
	missing = [m for m in metrics if m not in df_eval.columns]
	if missing:
		raise ValueError(f'eval_metrics.csv missing columns: {missing}')

	indices = map_true_xyz_to_zyx_indices(df_eval, axes)
	shape = axes.shape_zyx()

	grids: dict[str, np.ndarray] = {}
	for metric in metrics:
		values = pd.to_numeric(df_eval[metric], errors='raise').to_numpy(float)
		grids[metric] = build_metric_grid_zyx(indices, values, shape)

	return grids


def resolve_vmin_vmax(
	metric: str,
	grid_zyx: np.ndarray,
	*,
	scale: HeatmapScaleConfig,
) -> tuple[float, float]:
	"""Resolve plot scale, preferring explicit config when both bounds exist."""
	display = get_heatmap_metric_display(metric)
	if display.fixed_range is not None:
		return display.fixed_range

	if (scale.vmin is None) != (scale.vmax is None):
		raise ValueError(
			'heatmap.scale.vmin and heatmap.scale.vmax '
			'must be both specified or both omitted'
		)

	if scale.vmin is not None and scale.vmax is not None:
		vmin = float(scale.vmin)
		vmax = float(scale.vmax)
		if not np.isfinite(vmin) or not np.isfinite(vmax):
			raise ValueError('heatmap.scale.vmin and heatmap.scale.vmax must be finite')
		if vmax <= vmin:
			raise ValueError('heatmap.scale.vmax must be > heatmap.scale.vmin')
		return (vmin, vmax)

	return compute_vmin_vmax(
		metric,
		grid_zyx,
		percentile=scale.percentile,
		dz_symmetric=scale.dz_symmetric,
	)


def write_axes_json(out_json: Path, axes: GridAxes) -> Path:
	"""Save axes.json with x_m/y_m/z_m + shape/order/center metadata."""
	out_json.parent.mkdir(parents=True, exist_ok=True)
	center_x = float(axes.x_m[axes.center_x_index()])
	center_y = float(axes.y_m[axes.center_y_index()])
	payload = {
		'x_m': np.asarray(axes.x_m).astype(float).tolist(),
		'y_m': np.asarray(axes.y_m).astype(float).tolist(),
		'z_m': np.asarray(axes.z_m).astype(float).tolist(),
		'shape': {
			'nx': int(axes.x_m.size),
			'ny': int(axes.y_m.size),
			'nz': int(axes.z_m.size),
		},
		'order': ['z', 'y', 'x'],
		'center_x_m': center_x,
		'center_y_m': center_y,
	}
	out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
	return out_json


def write_metric_grid_npy(out_npy: Path, grid_zyx: np.ndarray) -> Path:
	"""Save grid[z,y,x] as .npy and return path."""
	out_npy.parent.mkdir(parents=True, exist_ok=True)
	np.save(out_npy, np.asarray(grid_zyx, dtype=float))
	return out_npy


def _require_int_values(df: pd.DataFrame, col: str) -> np.ndarray:
	series = pd.to_numeric(df[col], errors='raise')
	if series.isna().any():
		raise ValueError(f'{col} has missing values')
	vals = series.to_numpy(float)
	if not np.isfinite(vals).all():
		raise ValueError(f'{col} has non-finite values')
	if not np.equal(vals, np.round(vals)).all():
		raise ValueError(f'{col} must be integer-valued (meters)')
	return vals.astype(int)


def _build_axis_index(axis_vals: np.ndarray, name: str) -> dict[int, int]:
	axis = np.asarray(axis_vals, dtype=float)
	if axis.size == 0:
		raise ValueError(f'{name} axis is empty')
	if not np.isfinite(axis).all():
		raise ValueError(f'{name} axis has non-finite values')
	if not np.equal(axis, np.round(axis)).all():
		raise ValueError(f'{name} axis must be integer-valued (meters)')
	axis_int = axis.astype(int)
	if np.unique(axis_int).size != axis_int.size:
		raise ValueError(f'{name} axis has duplicates')
	return {int(v): int(i) for i, v in enumerate(axis_int)}


def run_heatmap_qc(
	*,
	df_eval: pd.DataFrame,
	dataset_dir: Path,
	run_dir: Path,
	cfg: HeatmapConfig,
) -> HeatmapArtifacts:
	"""Orchestrate heatmap QC: axes -> grids -> save npy/json -> render PNGs."""
	if not cfg.metrics:
		raise ValueError('heatmap.metrics must be non-empty')
	if not cfg.scale.global_across_slices:
		raise ValueError('heatmap.scale.global_across_slices must be True (fixed)')
	if not cfg.scale.dz_symmetric:
		raise ValueError('heatmap.scale.dz_symmetric must be True (fixed)')

	index_csv = dataset_dir / 'index.csv'
	axes = load_grid_axes_from_index_csv(index_csv)

	grids = build_metric_grids_zyx(df_eval, axes, cfg.metrics)

	out_root = run_dir / cfg.output.out_dirname
	out_root.mkdir(parents=True, exist_ok=True)

	axes_json: Path | None = None
	if cfg.output.save_axes_json:
		axes_json = write_axes_json(out_root / 'axes.json', axes)

	metric_npy: dict[str, Path] = {}
	if cfg.output.save_npy:
		for metric, grid in grids.items():
			metric_npy[metric] = write_metric_grid_npy(out_root / f'{metric}.npy', grid)

	from viz.hypo.synth_eval import (
		save_heatmap_xy_slices,
		save_heatmap_xz_center_y,
		save_heatmap_yz_center_x,
	)

	metric_pngs: dict[str, dict[str, list[Path]]] = {}
	for metric, grid in grids.items():
		vmin, vmax = resolve_vmin_vmax(metric, grid, scale=cfg.scale)
		display = get_heatmap_metric_display(metric)
		metric_dir = out_root / metric
		metric_dir.mkdir(parents=True, exist_ok=True)

		pngs_xy: list[Path] = []
		pngs_xz: list[Path] = []
		pngs_yz: list[Path] = []

		if cfg.slices.xy_all_depths:
			pngs_xy = save_heatmap_xy_slices(
				grid,
				axes,
				metric_dir,
				metric=metric,
				display_name=display.display_name,
				colorbar_label=display.colorbar_label,
				vmin=vmin,
				vmax=vmax,
			)
		if cfg.slices.xz_center_y:
			pngs_xz = [
				save_heatmap_xz_center_y(
					grid,
					axes,
					metric_dir,
					metric=metric,
					display_name=display.display_name,
					colorbar_label=display.colorbar_label,
					vmin=vmin,
					vmax=vmax,
					center_y_index=axes.center_y_index(),
				)
			]
		if cfg.slices.yz_center_x:
			pngs_yz = [
				save_heatmap_yz_center_x(
					grid,
					axes,
					metric_dir,
					metric=metric,
					display_name=display.display_name,
					colorbar_label=display.colorbar_label,
					vmin=vmin,
					vmax=vmax,
					center_x_index=axes.center_x_index(),
				)
			]

		metric_pngs[metric] = {'xy': pngs_xy, 'xz': pngs_xz, 'yz': pngs_yz}

	return HeatmapArtifacts(
		axes_json=axes_json,
		metric_npy=metric_npy,
		metric_pngs=metric_pngs,
	)
