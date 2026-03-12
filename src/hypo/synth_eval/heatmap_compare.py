from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from common.yaml_config import read_yaml_mapping
from hypo.synth_eval.heatmap_grid import (
	axis_extent_km_from_centers,
	validate_heatmap_grid_shape,
)
from hypo.synth_eval.heatmap_scale import compute_vmin_vmax
from hypo.synth_eval.heatmap_types import GridAxes
from viz.core.fig_io import save_figure

VALID_SLICES = {'xy', 'xz', 'yz'}
UNCERTAINTY_METRICS = {'ERH', 'ERZ'}
REQUIRED_AXES_JSON_KEYS = {
	'x_m',
	'y_m',
	'z_m',
	'shape',
	'order',
	'center_x_m',
	'center_y_m',
}


@dataclass(frozen=True)
class CompareScaleConfig:
	percentile: float | None
	vmin: float | None
	vmax: float | None


@dataclass(frozen=True)
class FigureSizePerPanel:
	width: float
	height: float


@dataclass(frozen=True)
class CompareInputConfig:
	label: str
	heatmap_root: Path


@dataclass(frozen=True)
class CompareConfig:
	metric: str
	slice_name: str
	coord_m: float
	output_png: Path
	ncols: int
	figsize_per_panel: FigureSizePerPanel
	title: str | None
	uncertainty_scale_sec: float | None
	scale: CompareScaleConfig
	inputs: list[CompareInputConfig]


@dataclass(frozen=True)
class LoadedHeatmapInput:
	label: str
	heatmap_root: Path
	axes: GridAxes
	grid_zyx: np.ndarray


@dataclass(frozen=True)
class PanelData:
	label: str
	data_2d: np.ndarray
	extent_km: tuple[float, float, float, float]
	xlabel: str
	ylabel: str
	invert_y: bool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--config', required=True, type=Path, help='compare config yaml'
	)
	return parser.parse_args(argv)


def _require_mapping(obj: Any, field: str) -> dict[str, Any]:
	if not isinstance(obj, dict):
		raise ValueError(f'{field} must be a mapping')
	return obj


def _is_real_number(value: Any) -> bool:
	return isinstance(value, Real) and not isinstance(value, bool)


def _require_finite_float(value: Any, field: str) -> float:
	if not _is_real_number(value):
		raise ValueError(f'{field} must be a finite number')
	out = float(value)
	if not np.isfinite(out):
		raise ValueError(f'{field} must be a finite number')
	return out


def _require_positive_float(value: Any, field: str) -> float:
	out = _require_finite_float(value, field)
	if out <= 0.0:
		raise ValueError(f'{field} must be > 0')
	return out


def _read_nonempty_str(obj: dict[str, Any], key: str, *, field: str) -> str:
	if key not in obj:
		raise ValueError(f'{field} is required')
	value = obj[key]
	if not isinstance(value, str):
		raise ValueError(f'{field} must be a non-empty string')
	text = value.strip()
	if text == '':
		raise ValueError(f'{field} must be a non-empty string')
	return text


def _read_optional_nonempty_str(
	obj: dict[str, Any], key: str, *, field: str
) -> str | None:
	if key not in obj or obj[key] is None:
		return None
	value = obj[key]
	if not isinstance(value, str):
		raise ValueError(f'{field} must be a non-empty string')
	text = value.strip()
	if text == '':
		raise ValueError(f'{field} must be a non-empty string')
	return text


def _read_int_ge1(obj: dict[str, Any], key: str, *, field: str) -> int:
	if key not in obj:
		raise ValueError(f'{field} is required')
	value = obj[key]
	if not isinstance(value, int) or isinstance(value, bool):
		raise ValueError(f'{field} must be an int >= 1')
	if value < 1:
		raise ValueError(f'{field} must be an int >= 1')
	return int(value)


def _read_scale_config(obj: Any) -> CompareScaleConfig:
	scale = _require_mapping(obj, 'compare.scale')
	raw_vmin = scale.get('vmin')
	raw_vmax = scale.get('vmax')
	has_vmin = raw_vmin is not None
	has_vmax = raw_vmax is not None
	if has_vmin != has_vmax:
		raise ValueError(
			'compare.scale.vmin and compare.scale.vmax must be both specified or both omitted'
		)

	percentile: float | None = None
	if 'percentile' in scale and scale['percentile'] is not None:
		percentile = _require_finite_float(
			scale['percentile'], 'compare.scale.percentile'
		)
		if not (0.0 < percentile <= 100.0):
			raise ValueError('compare.scale.percentile must satisfy 0.0 < p <= 100.0')

	vmin: float | None = None
	vmax: float | None = None
	if has_vmin and has_vmax:
		vmin = _require_finite_float(raw_vmin, 'compare.scale.vmin')
		vmax = _require_finite_float(raw_vmax, 'compare.scale.vmax')
		if vmax <= vmin:
			raise ValueError('compare.scale.vmax must be > compare.scale.vmin')
	elif percentile is None:
		raise ValueError(
			'compare.scale.percentile is required when compare.scale.vmin/vmax are omitted'
		)

	return CompareScaleConfig(percentile=percentile, vmin=vmin, vmax=vmax)


def load_compare_config(path: Path) -> CompareConfig:
	obj = read_yaml_mapping(path)
	compare = _require_mapping(obj.get('compare'), 'compare')

	metric = _read_nonempty_str(compare, 'metric', field='compare.metric')
	slice_name = _read_nonempty_str(compare, 'slice', field='compare.slice').lower()
	if slice_name not in VALID_SLICES:
		raise ValueError('compare.slice must be one of xy, xz, yz')

	coord_m = _require_finite_float(compare.get('coord_m'), 'compare.coord_m')
	output_png = Path(
		_read_nonempty_str(compare, 'output_png', field='compare.output_png')
	).expanduser()
	ncols = _read_int_ge1(compare, 'ncols', field='compare.ncols')

	figsize_obj = _require_mapping(
		compare.get('figsize_per_panel'), 'compare.figsize_per_panel'
	)
	figsize = FigureSizePerPanel(
		width=_require_positive_float(
			figsize_obj.get('width'), 'compare.figsize_per_panel.width'
		),
		height=_require_positive_float(
			figsize_obj.get('height'), 'compare.figsize_per_panel.height'
		),
	)

	inputs_obj = compare.get('inputs')
	if not isinstance(inputs_obj, list):
		raise ValueError('compare.inputs must be a list')
	if len(inputs_obj) < 2:
		raise ValueError('compare.inputs must contain at least 2 items')

	inputs: list[CompareInputConfig] = []
	for index, raw_input in enumerate(inputs_obj):
		field = f'compare.inputs[{index}]'
		item = _require_mapping(raw_input, field)
		inputs.append(
			CompareInputConfig(
				label=_read_nonempty_str(item, 'label', field=f'{field}.label'),
				heatmap_root=Path(
					_read_nonempty_str(
						item,
						'heatmap_root',
						field=f'{field}.heatmap_root',
					)
				).expanduser(),
			)
		)

	return CompareConfig(
		metric=metric,
		slice_name=slice_name,
		coord_m=coord_m,
		output_png=output_png,
		ncols=ncols,
		figsize_per_panel=figsize,
		title=_read_optional_nonempty_str(compare, 'title', field='compare.title'),
		uncertainty_scale_sec=(
			None
			if 'uncertainty_scale_sec' not in compare
			or compare['uncertainty_scale_sec'] is None
			else _require_positive_float(
				compare['uncertainty_scale_sec'],
				'compare.uncertainty_scale_sec',
			)
		),
		scale=_read_scale_config(compare.get('scale')),
		inputs=inputs,
	)


def _read_axes_shape(obj: Any) -> tuple[int, int, int]:
	shape = _require_mapping(obj, 'axes.json.shape')
	nx = shape.get('nx')
	ny = shape.get('ny')
	nz = shape.get('nz')
	for value, field in (
		(nx, 'axes.json.shape.nx'),
		(ny, 'axes.json.shape.ny'),
		(nz, 'axes.json.shape.nz'),
	):
		if not isinstance(value, int) or isinstance(value, bool):
			raise ValueError(f'{field} must be a positive int')
		if value < 1:
			raise ValueError(f'{field} must be a positive int')
	return (int(nz), int(ny), int(nx))


def _read_axis_array(obj: dict[str, Any], key: str) -> np.ndarray:
	field = f'axes.json.{key}'
	if key not in obj:
		raise ValueError(f'{field} is required')
	value = obj[key]
	if not isinstance(value, list):
		raise ValueError(f'{field} must be a 1D list of finite integer-valued meters')
	if not value:
		raise ValueError(f'{field} must contain at least 2 values')
	if not all(_is_real_number(v) for v in value):
		raise ValueError(f'{field} must be a 1D list of finite integer-valued meters')
	axis = np.asarray(value, dtype=float)
	if axis.ndim != 1:
		raise ValueError(f'{field} must be a 1D list of finite integer-valued meters')
	if axis.size < 2:
		raise ValueError(f'{field} must contain at least 2 values')
	if not np.isfinite(axis).all():
		raise ValueError(f'{field} must be a 1D list of finite integer-valued meters')
	if not np.equal(axis, np.round(axis)).all():
		raise ValueError(f'{field} must be integer-valued (meters)')
	axis_extent_km_from_centers(axis)
	return axis


def load_axes_json(path: Path) -> GridAxes:
	if not path.is_file():
		raise FileNotFoundError(f'missing: {path}')
	obj = json.loads(path.read_text(encoding='utf-8'))
	axes_json = _require_mapping(obj, 'axes.json')
	missing = sorted(REQUIRED_AXES_JSON_KEYS - set(axes_json.keys()))
	if missing:
		raise ValueError(f'axes.json missing keys: {missing}')

	order = axes_json['order']
	if order != ['z', 'y', 'x']:
		raise ValueError("axes.json.order must be ['z', 'y', 'x']")

	axes = GridAxes(
		x_m=_read_axis_array(axes_json, 'x_m'),
		y_m=_read_axis_array(axes_json, 'y_m'),
		z_m=_read_axis_array(axes_json, 'z_m'),
	)
	shape_zyx = _read_axes_shape(axes_json['shape'])
	if axes.shape_zyx() != shape_zyx:
		raise ValueError(
			f'axes.json shape mismatch: axes={axes.shape_zyx()} shape={shape_zyx}'
		)

	center_x_m = _require_finite_float(axes_json['center_x_m'], 'axes.json.center_x_m')
	center_y_m = _require_finite_float(axes_json['center_y_m'], 'axes.json.center_y_m')
	if center_x_m != float(axes.x_m[axes.center_x_index()]):
		raise ValueError('axes.json.center_x_m does not match x_m center')
	if center_y_m != float(axes.y_m[axes.center_y_index()]):
		raise ValueError('axes.json.center_y_m does not match y_m center')

	return axes


def _require_matching_axes(
	reference: GridAxes, candidate: GridAxes, *, label: str
) -> None:
	for axis_name in ('x_m', 'y_m', 'z_m'):
		ref_axis = getattr(reference, axis_name)
		cand_axis = getattr(candidate, axis_name)
		if not np.array_equal(ref_axis, cand_axis):
			raise ValueError(f'axes mismatch for {axis_name}: label={label}')


def load_metric_grid_zyx(path: Path, axes: GridAxes) -> np.ndarray:
	if not path.is_file():
		raise FileNotFoundError(f'missing: {path}')
	grid = np.load(path, allow_pickle=False)
	grid_zyx = np.asarray(grid, dtype=float)
	validate_heatmap_grid_shape(grid_zyx, axes)
	return grid_zyx


def scale_compare_metric_grid(
	metric: str,
	grid_zyx: np.ndarray,
	uncertainty_scale_sec: float | None,
) -> np.ndarray:
	grid = np.asarray(grid_zyx, dtype=float)
	if uncertainty_scale_sec is None:
		return grid
	scale_sec = _require_positive_float(
		uncertainty_scale_sec,
		'compare.uncertainty_scale_sec',
	)
	if metric not in UNCERTAINTY_METRICS:
		return grid
	return grid * scale_sec


def load_heatmap_input(
	input_cfg: CompareInputConfig,
	*,
	metric: str,
	uncertainty_scale_sec: float | None,
	reference_axes: GridAxes | None = None,
) -> LoadedHeatmapInput:
	root = input_cfg.heatmap_root
	if not root.is_dir():
		raise FileNotFoundError(f'heatmap_root not found: {root}')
	axes = load_axes_json(root / 'axes.json')
	if reference_axes is not None:
		_require_matching_axes(reference_axes, axes, label=input_cfg.label)
	grid_zyx = scale_compare_metric_grid(
		metric,
		load_metric_grid_zyx(root / f'{metric}.npy', axes),
		uncertainty_scale_sec,
	)
	return LoadedHeatmapInput(
		label=input_cfg.label,
		heatmap_root=root,
		axes=axes,
		grid_zyx=grid_zyx,
	)


def _coord_axis_name(slice_name: str) -> str:
	return {
		'xy': 'z_m',
		'xz': 'y_m',
		'yz': 'x_m',
	}[slice_name]


def _coord_axis_values(axes: GridAxes, slice_name: str) -> np.ndarray:
	return getattr(axes, _coord_axis_name(slice_name))


def _coord_axis_tag(slice_name: str) -> str:
	return _coord_axis_name(slice_name).replace('_m', '')


def _coord_index(axes: GridAxes, slice_name: str, coord_m: float) -> int:
	if coord_m != float(round(coord_m)):
		raise ValueError('compare.coord_m must be integer-valued (meters)')
	coord_int = int(round(coord_m))
	axis_name = _coord_axis_name(slice_name)
	axis = _coord_axis_values(axes, slice_name).astype(int)
	matches = np.flatnonzero(axis == coord_int)
	if matches.size == 0:
		raise ValueError(
			f'compare.coord_m={coord_int} is not present in {axis_name} for slice={slice_name}'
		)
	return int(matches[0])


def extract_slice_2d(
	grid_zyx: np.ndarray,
	axes: GridAxes,
	*,
	slice_name: str,
	coord_m: float,
) -> np.ndarray:
	validate_heatmap_grid_shape(grid_zyx, axes)
	index = _coord_index(axes, slice_name, coord_m)
	if slice_name == 'xy':
		return np.asarray(grid_zyx[index, :, :], dtype=float)
	if slice_name == 'xz':
		return np.asarray(grid_zyx[:, index, :], dtype=float)
	if slice_name == 'yz':
		return np.asarray(grid_zyx[:, :, index], dtype=float)
	raise ValueError(f'unsupported slice: {slice_name}')


def slice_extent_labels(
	axes: GridAxes, *, slice_name: str
) -> tuple[tuple[float, float, float, float], str, str, bool]:
	if slice_name == 'xy':
		xmin_km, xmax_km = axis_extent_km_from_centers(axes.x_m)
		ymin_km, ymax_km = axis_extent_km_from_centers(axes.y_m)
		return ((xmin_km, xmax_km, ymin_km, ymax_km), 'X (km)', 'Y (km)', False)
	if slice_name == 'xz':
		xmin_km, xmax_km = axis_extent_km_from_centers(axes.x_m)
		zmin_km, zmax_km = axis_extent_km_from_centers(axes.z_m)
		return ((xmin_km, xmax_km, zmin_km, zmax_km), 'X (km)', 'Depth (km)', True)
	if slice_name == 'yz':
		ymin_km, ymax_km = axis_extent_km_from_centers(axes.y_m)
		zmin_km, zmax_km = axis_extent_km_from_centers(axes.z_m)
		return ((ymin_km, ymax_km, zmin_km, zmax_km), 'Y (km)', 'Depth (km)', True)
	raise ValueError(f'unsupported slice: {slice_name}')


def build_panel_data(
	loaded: LoadedHeatmapInput, *, slice_name: str, coord_m: float
) -> PanelData:
	data_2d = extract_slice_2d(
		loaded.grid_zyx,
		loaded.axes,
		slice_name=slice_name,
		coord_m=coord_m,
	)
	extent_km, xlabel, ylabel, invert_y = slice_extent_labels(
		loaded.axes, slice_name=slice_name
	)
	return PanelData(
		label=loaded.label,
		data_2d=data_2d,
		extent_km=extent_km,
		xlabel=xlabel,
		ylabel=ylabel,
		invert_y=invert_y,
	)


def resolve_compare_scale(
	metric: str,
	data_slices_2d: list[np.ndarray],
	scale: CompareScaleConfig,
) -> tuple[float, float]:
	if scale.vmin is not None and scale.vmax is not None:
		return (float(scale.vmin), float(scale.vmax))
	stack = np.stack([np.asarray(data, dtype=float) for data in data_slices_2d], axis=0)
	if not np.isfinite(stack).any():
		raise ValueError('all extracted slices are non-finite')
	if scale.percentile is None:
		raise ValueError(
			'compare.scale.percentile is required when compare.scale.vmin/vmax are omitted'
		)
	return compute_vmin_vmax(
		metric,
		stack,
		percentile=scale.percentile,
		dz_symmetric=True,
	)


def compute_layout(n_panels: int, ncols: int) -> tuple[int, int]:
	if n_panels < 1:
		raise ValueError('n_panels must be >= 1')
	if ncols < 1:
		raise ValueError('ncols must be >= 1')
	nrows = (n_panels + ncols - 1) // ncols
	return (nrows, ncols)


def _default_title(cfg: CompareConfig) -> str:
	coord_tag = int(round(cfg.coord_m))
	title = f'{cfg.metric} {cfg.slice_name} {_coord_axis_tag(cfg.slice_name)}={coord_tag} m'
	if cfg.metric in UNCERTAINTY_METRICS and cfg.uncertainty_scale_sec is not None:
		return f'{title} (scaled to {cfg.uncertainty_scale_sec:g} s)'
	return title


def save_heatmap_comparison(
	panels: list[PanelData],
	cfg: CompareConfig,
	*,
	vmin: float,
	vmax: float,
) -> Path:
	if not panels:
		raise ValueError('panels must be non-empty')

	nrows, ncols = compute_layout(len(panels), cfg.ncols)
	fig, ax_grid = plt.subplots(
		nrows,
		ncols,
		figsize=(
			cfg.figsize_per_panel.width * ncols,
			cfg.figsize_per_panel.height * nrows,
		),
		squeeze=False,
	)
	flat_axes = list(ax_grid.reshape(-1))
	used_axes: list[Any] = []
	image = None

	for ax, panel in zip(flat_axes, panels):
		image = ax.imshow(
			panel.data_2d,
			origin='lower',
			extent=panel.extent_km,
			vmin=float(vmin),
			vmax=float(vmax),
		)
		if panel.invert_y:
			ax.invert_yaxis()
		ax.set_title(panel.label)
		ax.set_xlabel(panel.xlabel)
		ax.set_ylabel(panel.ylabel)
		used_axes.append(ax)

	for ax in flat_axes[len(panels) :]:
		fig.delaxes(ax)

	if image is None:
		raise ValueError('no panel image was created')

	fig.suptitle(cfg.title if cfg.title is not None else _default_title(cfg))
	fig.subplots_adjust(
		left=0.10,
		right=0.86,
		bottom=0.10,
		top=0.88,
		wspace=0.30,
		hspace=0.30,
	)
	y0 = min(ax.get_position().y0 for ax in used_axes)
	y1 = max(ax.get_position().y1 for ax in used_axes)
	cbar_ax = fig.add_axes([0.89, y0, 0.02, y1 - y0])
	fig.colorbar(image, cax=cbar_ax)
	return save_figure(fig, cfg.output_png, dpi=200)


def load_all_inputs(cfg: CompareConfig) -> list[LoadedHeatmapInput]:
	loaded: list[LoadedHeatmapInput] = []
	reference_axes: GridAxes | None = None
	for input_cfg in cfg.inputs:
		item = load_heatmap_input(
			input_cfg,
			metric=cfg.metric,
			uncertainty_scale_sec=cfg.uncertainty_scale_sec,
			reference_axes=reference_axes,
		)
		if reference_axes is None:
			reference_axes = item.axes
		loaded.append(item)
	return loaded


def run_compare(cfg: CompareConfig) -> Path:
	if cfg.uncertainty_scale_sec is not None:
		if cfg.metric in UNCERTAINTY_METRICS:
			print(
				f'[INFO] compare.uncertainty_scale_sec={cfg.uncertainty_scale_sec:g} '
				f'(applied to {cfg.metric})'
			)
		else:
			print(
				f'[INFO] compare.uncertainty_scale_sec={cfg.uncertainty_scale_sec:g} '
				f'(not applied for metric={cfg.metric})'
			)
	loaded = load_all_inputs(cfg)
	panels = [
		build_panel_data(item, slice_name=cfg.slice_name, coord_m=cfg.coord_m)
		for item in loaded
	]
	vmin, vmax = resolve_compare_scale(
		cfg.metric,
		[data.data_2d for data in panels],
		cfg.scale,
	)
	out_png = save_heatmap_comparison(panels, cfg, vmin=vmin, vmax=vmax)
	print(out_png)
	return out_png


def run_compare_from_config_path(config_path: Path) -> Path:
	return run_compare(load_compare_config(config_path))


def main(argv: list[str] | None = None) -> Path:
	args = parse_args(argv)
	return run_compare_from_config_path(args.config.expanduser().resolve())


if __name__ == '__main__':
	main()
