from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg', force=True)

import numpy as np
import pandas as pd

from qc.hypo.heatmap import (
	HeatmapConfig,
	HeatmapOutputConfig,
	HeatmapScaleConfig,
	HeatmapSlicesConfig,
	run_heatmap_qc,
)


def _make_index_csv(dataset_dir: Path) -> Path:
	dataset_dir.mkdir(parents=True, exist_ok=True)
	x_vals = [0, 1000]
	y_vals = [0, 1000]
	z_vals = [0, 1000]
	rows = []
	for x in x_vals:
		for y in y_vals:
			for z in z_vals:
				rows.append({'x_m': x, 'y_m': y, 'z_m': z})
	df = pd.DataFrame(rows)
	path = dataset_dir / 'index.csv'
	df.to_csv(path, index=False)
	return path


def _make_eval_df() -> pd.DataFrame:
	x_vals = [0, 1000]
	y_vals = [0, 1000]
	z_vals = [0, 1000]
	rows = []
	i = 0
	for x in x_vals:
		for y in y_vals:
			for z in z_vals:
				rows.append(
					{
						'x_m_true': x,
						'y_m_true': y,
						'z_m_true': z,
						'err3d_m': float(i + 1),
						'horiz_m': float(i + 2),
						'dz_m': float((i % 3) - 1),
					}
				)
				i += 1
	return pd.DataFrame(rows)


def test_run_heatmap_qc_integration(tmp_path: Path) -> None:
	dataset_dir = tmp_path / 'dataset'
	run_dir = tmp_path / 'run'
	run_dir.mkdir(parents=True, exist_ok=True)

	_make_index_csv(dataset_dir)
	df_eval = _make_eval_df()

	cfg = HeatmapConfig(
		enabled=True,
		metrics=['err3d_m', 'horiz_m', 'dz_m'],
		slices=HeatmapSlicesConfig(
			xy_all_depths=True, xz_center_y=True, yz_center_x=True
		),
		scale=HeatmapScaleConfig(
			percentile=99.0, global_across_slices=True, dz_symmetric=True
		),
		output=HeatmapOutputConfig(
			save_npy=True, save_axes_json=True, out_dirname='heatmaps'
		),
	)

	art = run_heatmap_qc(
		df_eval=df_eval,
		dataset_dir=dataset_dir,
		run_dir=run_dir,
		cfg=cfg,
	)

	assert art.axes_json is not None
	assert art.axes_json.is_file()
	loaded = art.axes_json.read_text(encoding='utf-8')
	for key in ['x_m', 'y_m', 'z_m', 'shape', 'order', 'center_x_m', 'center_y_m']:
		assert key in loaded

	for metric in cfg.metrics:
		assert metric in art.metric_npy
		npy_path = art.metric_npy[metric]
		assert npy_path.is_file()
		grid = np.load(npy_path)
		assert grid.shape == (2, 2, 2)

		assert metric in art.metric_pngs
		pngs = art.metric_pngs[metric]
		assert set(pngs.keys()) == {'xy', 'xz', 'yz'}
		assert len(pngs['xy']) == 2
		assert len(pngs['xz']) == 1
		assert len(pngs['yz']) == 1
		for p in pngs['xy'] + pngs['xz'] + pngs['yz']:
			assert p.is_file()
			assert p.stat().st_size > 0


def test_run_heatmap_qc_integration_with_explicit_scale(tmp_path: Path) -> None:
	dataset_dir = tmp_path / 'dataset'
	run_dir = tmp_path / 'run'
	run_dir.mkdir(parents=True, exist_ok=True)

	_make_index_csv(dataset_dir)
	df_eval = _make_eval_df()

	cfg = HeatmapConfig(
		enabled=True,
		metrics=['err3d_m', 'horiz_m', 'dz_m'],
		slices=HeatmapSlicesConfig(
			xy_all_depths=True, xz_center_y=True, yz_center_x=True
		),
		scale=HeatmapScaleConfig(
			percentile=99.0,
			global_across_slices=True,
			dz_symmetric=True,
			vmin=-2.0,
			vmax=20.0,
		),
		output=HeatmapOutputConfig(
			save_npy=True, save_axes_json=True, out_dirname='heatmaps_explicit'
		),
	)

	art = run_heatmap_qc(
		df_eval=df_eval,
		dataset_dir=dataset_dir,
		run_dir=run_dir,
		cfg=cfg,
	)

	assert art.axes_json is not None
	assert art.axes_json.is_file()

	for metric in cfg.metrics:
		assert metric in art.metric_npy
		assert art.metric_npy[metric].is_file()

		pngs = art.metric_pngs[metric]
		assert len(pngs['xy']) == 2
		assert len(pngs['xz']) == 1
		assert len(pngs['yz']) == 1
		for p in pngs['xy'] + pngs['xz'] + pngs['yz']:
			assert p.is_file()
			assert p.stat().st_size > 0
