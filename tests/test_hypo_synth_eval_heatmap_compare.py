from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg', force=True)

import numpy as np
import pytest
import yaml

import hypo.synth_eval.heatmap_compare as hc
from hypo.synth_eval.heatmap_types import GridAxes
from qc.hypo.heatmap import write_axes_json


def _make_axes() -> GridAxes:
	return GridAxes(
		x_m=np.array([0, 1000, 2000], dtype=float),
		y_m=np.array([0, 1000, 2000, 3000], dtype=float),
		z_m=np.array([0, 1000], dtype=float),
	)


def _make_grid(*, offset: float = 0.0) -> np.ndarray:
	axes = _make_axes()
	values = np.arange(np.prod(axes.shape_zyx()), dtype=float).reshape(axes.shape_zyx())
	return values + float(offset)


def _write_heatmap_root(root: Path, *, metric: str, grid_zyx: np.ndarray) -> None:
	root.mkdir(parents=True, exist_ok=True)
	write_axes_json(root / 'axes.json', _make_axes())
	np.save(root / f'{metric}.npy', np.asarray(grid_zyx, dtype=float))


def _write_compare_yaml(
	tmp_path: Path,
	*,
	metric: str,
	slice_name: str,
	coord_m: float,
	output_png: Path,
	ncols: int,
	scale: dict[str, object],
	inputs: list[tuple[str, Path]],
	title: str | None = 'Heatmap comparison',
	uncertainty_scale_sec: object | None = None,
	include_uncertainty_scale_sec: bool = False,
) -> Path:
	cfg = {
		'compare': {
			'metric': metric,
			'slice': slice_name,
			'coord_m': coord_m,
			'output_png': str(output_png),
			'ncols': ncols,
			'figsize_per_panel': {
				'width': 4.2,
				'height': 4.2,
			},
			'scale': scale,
			'inputs': [
				{
					'label': label,
					'heatmap_root': str(root),
				}
				for label, root in inputs
			],
		}
	}
	if title is not None:
		cfg['compare']['title'] = title
	if include_uncertainty_scale_sec:
		cfg['compare']['uncertainty_scale_sec'] = uncertainty_scale_sec

	path = tmp_path / 'compare_heatmaps.yaml'
	path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
	return path


def _patch_save_figure_capture(monkeypatch):
	cap = {}

	def _save_figure(
		fig,
		out_png,
		*,
		dpi=200,
		bbox_inches=None,
		pad_inches=None,
		tight_layout=False,
		close=True,
	):
		out_png = Path(out_png)
		out_png.parent.mkdir(parents=True, exist_ok=True)

		if tight_layout:
			fig.tight_layout()

		save_kwargs = {'dpi': int(dpi)}
		if bbox_inches is not None:
			save_kwargs['bbox_inches'] = bbox_inches
		if pad_inches is not None:
			save_kwargs['pad_inches'] = float(pad_inches)

		fig.savefig(out_png, **save_kwargs)
		cap['fig'] = fig
		cap['out_png'] = out_png

		if close:
			plt.close(fig)

		return out_png

	monkeypatch.setattr(hc, 'save_figure', _save_figure)
	return cap


def _panel_axes(fig) -> list[object]:
	return [ax for ax in fig.axes if len(ax.images) == 1]


def test_load_axes_json_missing_required_key_raises(tmp_path: Path) -> None:
	axes_json = tmp_path / 'axes.json'
	axes_json.write_text(
		json.dumps(
			{
				'x_m': [0, 1000],
				'y_m': [0, 1000],
				'z_m': [0, 1000],
				'shape': {'nx': 2, 'ny': 2, 'nz': 2},
				'order': ['z', 'y', 'x'],
				'center_x_m': 1000.0,
			}
		)
		+ '\n',
		encoding='utf-8',
	)

	with pytest.raises(ValueError, match='axes.json missing keys'):
		hc.load_axes_json(axes_json)


def test_load_metric_grid_shape_mismatch_raises(tmp_path: Path) -> None:
	axes = _make_axes()
	root = tmp_path / 'heatmaps'
	root.mkdir(parents=True, exist_ok=True)
	write_axes_json(root / 'axes.json', axes)
	np.save(root / 'err3d_m.npy', np.zeros((2, 3, 3), dtype=float))

	with pytest.raises(ValueError, match='grid_zyx shape mismatch'):
		hc.load_metric_grid_zyx(root / 'err3d_m.npy', axes)


def test_load_compare_config_reads_uncertainty_scale_sec(tmp_path: Path) -> None:
	root_a = tmp_path / 'run_a' / 'heatmaps'
	root_b = tmp_path / 'run_b' / 'heatmaps'
	cfg_path = _write_compare_yaml(
		tmp_path,
		metric='ERZ',
		slice_name='xy',
		coord_m=1000.0,
		output_png=tmp_path / 'compare.png',
		ncols=2,
		scale={'percentile': 99.0},
		inputs=[('run_a', root_a), ('run_b', root_b)],
		uncertainty_scale_sec=0.05,
		include_uncertainty_scale_sec=True,
	)

	cfg = hc.load_compare_config(cfg_path)

	assert cfg.uncertainty_scale_sec == 0.05


def test_load_compare_config_allows_null_uncertainty_scale_sec(tmp_path: Path) -> None:
	root_a = tmp_path / 'run_a' / 'heatmaps'
	root_b = tmp_path / 'run_b' / 'heatmaps'
	cfg_path = _write_compare_yaml(
		tmp_path,
		metric='ERZ',
		slice_name='xy',
		coord_m=1000.0,
		output_png=tmp_path / 'compare.png',
		ncols=2,
		scale={'percentile': 99.0},
		inputs=[('run_a', root_a), ('run_b', root_b)],
		uncertainty_scale_sec=None,
		include_uncertainty_scale_sec=True,
	)

	cfg = hc.load_compare_config(cfg_path)

	assert cfg.uncertainty_scale_sec is None


@pytest.mark.parametrize(
	'bad_value',
	[
		0.0,
		-1.0,
		float('nan'),
		float('inf'),
		True,
	],
)
def test_load_compare_config_rejects_invalid_uncertainty_scale_sec(
	tmp_path: Path, bad_value: object
) -> None:
	root_a = tmp_path / 'run_a' / 'heatmaps'
	root_b = tmp_path / 'run_b' / 'heatmaps'
	cfg_path = _write_compare_yaml(
		tmp_path,
		metric='ERZ',
		slice_name='xy',
		coord_m=1000.0,
		output_png=tmp_path / 'compare.png',
		ncols=2,
		scale={'percentile': 99.0},
		inputs=[('run_a', root_a), ('run_b', root_b)],
		uncertainty_scale_sec=bad_value,
		include_uncertainty_scale_sec=True,
	)

	with pytest.raises(ValueError, match='compare.uncertainty_scale_sec'):
		hc.load_compare_config(cfg_path)


def test_resolve_compare_scale_uses_explicit_bounds() -> None:
	vmin, vmax = hc.resolve_compare_scale(
		'err3d_m',
		[np.array([[1.0, 2.0]], dtype=float)],
		hc.CompareScaleConfig(percentile=None, vmin=0.0, vmax=500.0),
	)
	assert vmin == 0.0
	assert vmax == 500.0


@pytest.mark.parametrize(
	'scale',
	[
		{'percentile': 99.0, 'vmin': 0.0},
		{'percentile': 99.0, 'vmax': 500.0},
	],
)
def test_load_compare_config_rejects_half_specified_scale(
	tmp_path: Path, scale: dict[str, object]
) -> None:
	root_a = tmp_path / 'run_a' / 'heatmaps'
	root_b = tmp_path / 'run_b' / 'heatmaps'
	cfg_path = _write_compare_yaml(
		tmp_path,
		metric='err3d_m',
		slice_name='xy',
		coord_m=1000.0,
		output_png=tmp_path / 'compare.png',
		ncols=2,
		scale=scale,
		inputs=[('run_a', root_a), ('run_b', root_b)],
	)

	with pytest.raises(
		ValueError,
		match='compare.scale.vmin and compare.scale.vmax must be both specified or both omitted',
	):
		hc.load_compare_config(cfg_path)


@pytest.mark.parametrize(
	('vmin', 'vmax'),
	[
		(1.0, 1.0),
		(2.0, 1.0),
	],
)
def test_load_compare_config_rejects_non_increasing_scale(
	tmp_path: Path, vmin: float, vmax: float
) -> None:
	root_a = tmp_path / 'run_a' / 'heatmaps'
	root_b = tmp_path / 'run_b' / 'heatmaps'
	cfg_path = _write_compare_yaml(
		tmp_path,
		metric='err3d_m',
		slice_name='xy',
		coord_m=1000.0,
		output_png=tmp_path / 'compare.png',
		ncols=2,
		scale={'percentile': 99.0, 'vmin': vmin, 'vmax': vmax},
		inputs=[('run_a', root_a), ('run_b', root_b)],
	)

	with pytest.raises(
		ValueError, match='compare.scale.vmax must be > compare.scale.vmin'
	):
		hc.load_compare_config(cfg_path)


def test_resolve_compare_scale_auto_err3d_uses_zero_to_percentile() -> None:
	data_slices = [
		np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
		np.array([[5.0, np.nan], [6.0, 7.0]], dtype=float),
	]
	percentile = 80.0

	vmin, vmax = hc.resolve_compare_scale(
		'err3d_m',
		data_slices,
		hc.CompareScaleConfig(percentile=percentile, vmin=None, vmax=None),
	)

	assert vmin == 0.0
	assert np.isclose(vmax, np.nanpercentile(np.stack(data_slices), percentile))


def test_resolve_compare_scale_auto_dz_is_symmetric() -> None:
	data_slices = [
		np.array([[-3.0, 1.0], [2.0, 4.0]], dtype=float),
		np.array([[0.0, -5.0], [6.0, np.nan]], dtype=float),
	]
	percentile = 75.0

	vmin, vmax = hc.resolve_compare_scale(
		'dz_m',
		data_slices,
		hc.CompareScaleConfig(percentile=percentile, vmin=None, vmax=None),
	)

	expected = float(np.nanpercentile(np.abs(np.stack(data_slices)), percentile))
	assert np.isclose(vmax, expected)
	assert np.isclose(vmin, -expected)


@pytest.mark.parametrize('metric', ['ERH', 'ERZ'])
def test_scale_compare_metric_grid_scales_uncertainty_metrics(metric: str) -> None:
	grid = _make_grid()
	scaled = hc.scale_compare_metric_grid(metric, grid, 0.05)
	assert np.allclose(scaled, grid * 0.05)


@pytest.mark.parametrize('metric', ['err3d_m', 'dz_m'])
def test_scale_compare_metric_grid_leaves_non_target_metrics_unchanged(
	metric: str,
) -> None:
	grid = _make_grid()
	scaled = hc.scale_compare_metric_grid(metric, grid, 0.05)
	assert np.array_equal(scaled, grid)


def test_scale_compare_metric_grid_leaves_erh_unchanged_when_scale_is_none() -> None:
	grid = _make_grid()
	scaled = hc.scale_compare_metric_grid('ERH', grid, None)
	assert np.array_equal(scaled, grid)


def test_extract_slice_xy_xz_yz() -> None:
	axes = _make_axes()
	grid = _make_grid()

	xy = hc.extract_slice_2d(grid, axes, slice_name='xy', coord_m=1000.0)
	xz = hc.extract_slice_2d(grid, axes, slice_name='xz', coord_m=2000.0)
	yz = hc.extract_slice_2d(grid, axes, slice_name='yz', coord_m=1000.0)

	assert np.array_equal(xy, grid[1, :, :])
	assert np.array_equal(xz, grid[:, 2, :])
	assert np.array_equal(yz, grid[:, :, 1])


def test_extract_slice_raises_when_coord_missing() -> None:
	with pytest.raises(ValueError, match='compare.coord_m=500 is not present'):
		hc.extract_slice_2d(
			_make_grid(),
			_make_axes(),
			slice_name='xy',
			coord_m=500.0,
		)


def test_compute_layout_rows() -> None:
	assert hc.compute_layout(5, 3) == (2, 3)


def test_save_heatmap_comparison_removes_unused_subplot(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	cap = _patch_save_figure_capture(monkeypatch)
	panels = [
		hc.PanelData(
			label=f'run_{index}',
			data_2d=np.arange(12, dtype=float).reshape((4, 3)) + index,
			extent_km=(-0.5, 2.5, -0.5, 3.5),
			xlabel='X (km)',
			ylabel='Y (km)',
			invert_y=False,
		)
		for index in range(5)
	]
	cfg = hc.CompareConfig(
		metric='err3d_m',
		slice_name='xy',
		coord_m=1000.0,
		output_png=tmp_path / 'compare.png',
		ncols=3,
		figsize_per_panel=hc.FigureSizePerPanel(width=4.0, height=4.0),
		title='Layout test',
		uncertainty_scale_sec=None,
		scale=hc.CompareScaleConfig(percentile=99.0, vmin=0.0, vmax=10.0),
		inputs=[
			hc.CompareInputConfig(
				label=f'run_{index}', heatmap_root=tmp_path / f'run_{index}'
			)
			for index in range(5)
		],
	)

	out_png = hc.save_heatmap_comparison(panels, cfg, vmin=0.0, vmax=10.0)

	assert out_png.is_file()
	fig = cap['fig']
	panel_axes = _panel_axes(fig)
	assert len(panel_axes) == 5
	assert len(fig.axes) == 6
	colorbar_ax = [ax for ax in fig.axes if len(ax.images) == 0][0]
	max_panel_x1 = max(ax.get_position().x1 for ax in panel_axes)
	assert colorbar_ax.get_position().x0 > max_panel_x1


def test_run_compare_from_config_path_rejects_axes_mismatch(tmp_path: Path) -> None:
	root_a = tmp_path / 'run_a' / 'heatmaps'
	root_b = tmp_path / 'run_b' / 'heatmaps'
	_write_heatmap_root(root_a, metric='err3d_m', grid_zyx=_make_grid())
	root_b.mkdir(parents=True, exist_ok=True)
	write_axes_json(
		root_b / 'axes.json',
		GridAxes(
			x_m=np.array([0, 1500, 3000], dtype=float),
			y_m=_make_axes().y_m,
			z_m=_make_axes().z_m,
		),
	)
	np.save(root_b / 'err3d_m.npy', _make_grid())

	cfg_path = _write_compare_yaml(
		tmp_path,
		metric='err3d_m',
		slice_name='xy',
		coord_m=1000.0,
		output_png=tmp_path / 'compare.png',
		ncols=2,
		scale={'percentile': 99.0},
		inputs=[('run_a', root_a), ('run_b', root_b)],
	)

	with pytest.raises(ValueError, match='axes mismatch for x_m'):
		hc.run_compare_from_config_path(cfg_path)


def test_run_compare_scales_erz_for_auto_percentile(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	capsys: pytest.CaptureFixture[str],
) -> None:
	cap = _patch_save_figure_capture(monkeypatch)
	roots = [
		tmp_path / 'run_a' / 'heatmaps',
		tmp_path / 'run_b' / 'heatmaps',
	]
	grids = [
		_make_grid(offset=10.0),
		_make_grid(offset=20.0),
	]
	for root, grid in zip(roots, grids):
		_write_heatmap_root(root, metric='ERZ', grid_zyx=grid)

	cfg_path = _write_compare_yaml(
		tmp_path,
		metric='ERZ',
		slice_name='xy',
		coord_m=1000.0,
		output_png=tmp_path / 'outputs' / 'compare_erz_xy_z1000.png',
		ncols=2,
		scale={'percentile': 90.0},
		inputs=[('run_a', roots[0]), ('run_b', roots[1])],
		title=None,
		uncertainty_scale_sec=0.05,
		include_uncertainty_scale_sec=True,
	)

	cfg = hc.load_compare_config(cfg_path)
	loaded = hc.load_all_inputs(cfg)

	assert np.allclose(loaded[0].grid_zyx, grids[0] * 0.05)
	assert np.allclose(loaded[1].grid_zyx, grids[1] * 0.05)

	out_png = hc.run_compare_from_config_path(cfg_path)

	assert out_png.is_file()
	lines = capsys.readouterr().out.strip().splitlines()
	assert lines[0] == '[INFO] compare.uncertainty_scale_sec=0.05 (applied to ERZ)'
	assert lines[-1] == str(out_png)

	fig = cap['fig']
	panel_by_title = {ax.get_title(): ax for ax in _panel_axes(fig)}
	assert np.allclose(
		np.asarray(panel_by_title['run_a'].images[0].get_array()),
		grids[0][1, :, :] * 0.05,
	)
	assert np.allclose(
		np.asarray(panel_by_title['run_b'].images[0].get_array()),
		grids[1][1, :, :] * 0.05,
	)
	expected_stack = np.stack([grid[1, :, :] * 0.05 for grid in grids], axis=0)
	expected_vmax = float(np.nanpercentile(expected_stack, 90.0))
	assert {
		tuple(float(v) for v in ax.images[0].get_clim())
		for ax in panel_by_title.values()
	} == {(0.0, expected_vmax)}
	assert fig._suptitle is not None
	assert fig._suptitle.get_text() == 'ERZ xy z=1000 m (scaled to 0.05 s)'


def test_run_compare_scales_erz_for_explicit_bounds(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	cap = _patch_save_figure_capture(monkeypatch)
	roots = [
		tmp_path / 'run_a' / 'heatmaps',
		tmp_path / 'run_b' / 'heatmaps',
	]
	grids = [
		_make_grid(offset=0.0),
		_make_grid(offset=5.0),
	]
	for root, grid in zip(roots, grids):
		_write_heatmap_root(root, metric='ERZ', grid_zyx=grid)

	cfg_path = _write_compare_yaml(
		tmp_path,
		metric='ERZ',
		slice_name='xy',
		coord_m=1000.0,
		output_png=tmp_path / 'outputs' / 'compare_erz_explicit.png',
		ncols=2,
		scale={'percentile': 99.0, 'vmin': 0.0, 'vmax': 1.0},
		inputs=[('run_a', roots[0]), ('run_b', roots[1])],
		uncertainty_scale_sec=0.05,
		include_uncertainty_scale_sec=True,
	)

	out_png = hc.run_compare_from_config_path(cfg_path)

	assert out_png.is_file()
	panel_by_title = {ax.get_title(): ax for ax in _panel_axes(cap['fig'])}
	assert np.allclose(
		np.asarray(panel_by_title['run_a'].images[0].get_array()),
		grids[0][1, :, :] * 0.05,
	)
	assert np.allclose(
		np.asarray(panel_by_title['run_b'].images[0].get_array()),
		grids[1][1, :, :] * 0.05,
	)
	assert {
		tuple(float(v) for v in ax.images[0].get_clim())
		for ax in panel_by_title.values()
	} == {(0.0, 1.0)}


def test_run_compare_leaves_non_target_metric_unchanged(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	capsys: pytest.CaptureFixture[str],
) -> None:
	cap = _patch_save_figure_capture(monkeypatch)
	roots = [
		tmp_path / 'run_a' / 'heatmaps',
		tmp_path / 'run_b' / 'heatmaps',
	]
	grids = [
		_make_grid(offset=0.0),
		_make_grid(offset=10.0),
	]
	for root, grid in zip(roots, grids):
		_write_heatmap_root(root, metric='err3d_m', grid_zyx=grid)

	cfg_path = _write_compare_yaml(
		tmp_path,
		metric='err3d_m',
		slice_name='xy',
		coord_m=1000.0,
		output_png=tmp_path / 'outputs' / 'compare_err3d_xy_z1000.png',
		ncols=2,
		scale={'percentile': 90.0},
		inputs=[('run_a', roots[0]), ('run_b', roots[1])],
		uncertainty_scale_sec=0.05,
		include_uncertainty_scale_sec=True,
	)

	cfg = hc.load_compare_config(cfg_path)
	loaded = hc.load_all_inputs(cfg)

	assert np.array_equal(loaded[0].grid_zyx, grids[0])
	assert np.array_equal(loaded[1].grid_zyx, grids[1])

	out_png = hc.run_compare_from_config_path(cfg_path)

	assert out_png.is_file()
	lines = capsys.readouterr().out.strip().splitlines()
	assert (
		lines[0]
		== '[INFO] compare.uncertainty_scale_sec=0.05 (not applied for metric=err3d_m)'
	)
	panel_by_title = {ax.get_title(): ax for ax in _panel_axes(cap['fig'])}
	assert np.allclose(
		np.asarray(panel_by_title['run_a'].images[0].get_array()),
		grids[0][1, :, :],
	)
	assert np.allclose(
		np.asarray(panel_by_title['run_b'].images[0].get_array()),
		grids[1][1, :, :],
	)


def test_run_compare_from_config_path_creates_png_with_single_colorbar(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	capsys: pytest.CaptureFixture[str],
) -> None:
	cap = _patch_save_figure_capture(monkeypatch)
	roots = [
		tmp_path / 'run_a' / 'heatmaps',
		tmp_path / 'run_b' / 'heatmaps',
		tmp_path / 'run_c' / 'heatmaps',
	]
	grids = [
		_make_grid(offset=0.0),
		_make_grid(offset=10.0),
		_make_grid(offset=20.0),
	]
	for root, grid in zip(roots, grids):
		_write_heatmap_root(root, metric='err3d_m', grid_zyx=grid)

	output_png = tmp_path / 'outputs' / 'compare_err3d_xy_z1000.png'
	cfg_path = _write_compare_yaml(
		tmp_path,
		metric='err3d_m',
		slice_name='xy',
		coord_m=1000.0,
		output_png=output_png,
		ncols=2,
		scale={'percentile': 90.0},
		inputs=[
			('run_a', roots[0]),
			('run_b', roots[1]),
			('run_c', roots[2]),
		],
	)

	out_png = hc.run_compare_from_config_path(cfg_path)

	assert out_png == output_png
	assert out_png.is_file()
	assert capsys.readouterr().out.strip().splitlines()[-1] == str(out_png)

	expected_stack = np.stack([grid[1, :, :] for grid in grids], axis=0)
	expected_vmax = float(np.nanpercentile(expected_stack, 90.0))

	fig = cap['fig']
	panel_axes = _panel_axes(fig)
	assert len(panel_axes) == 3
	assert len(fig.axes) == 4
	colorbar_ax = [ax for ax in fig.axes if len(ax.images) == 0][0]
	max_panel_x1 = max(ax.get_position().x1 for ax in panel_axes)
	assert colorbar_ax.get_position().x0 > max_panel_x1
	assert all(ax.get_title() in {'run_a', 'run_b', 'run_c'} for ax in panel_axes)
	assert {tuple(float(v) for v in ax.images[0].get_clim()) for ax in panel_axes} == {
		(0.0, expected_vmax)
	}
