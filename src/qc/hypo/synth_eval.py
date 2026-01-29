from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from hypo.synth_eval.validation import (
	require_abs,
	require_dirname_only,
	require_filename_only,
)
from viz.hypo.synth_eval import save_dxdy_scatter, save_hist, save_true_pred_xyz_3view


@dataclass(frozen=True)
class Config:
	dataset_dir: str
	outputs_dir: str
	receiver_geometry: str


def load_config(path: Path) -> Config:
	obj = yaml.safe_load(path.read_text(encoding='utf-8'))
	return Config(
		dataset_dir=str(obj['dataset_dir']),
		outputs_dir=str(obj['outputs_dir']),
		receiver_geometry=str(obj['receiver_geometry']),
	)


def _load_stations_from_run_output(
	*,
	run_dir: Path,
	dataset_dir: Path,
	receiver_geometry: str,
) -> tuple[np.ndarray, np.ndarray]:
	"""Load station xyz and DAS mask based on run output station_synth.csv.

	Stations are sourced from run_dir/station_synth.csv to ensure QC follows the
	actual station selection used in the run.
	"""
	station_csv = run_dir / 'station_synth.csv'
	if not station_csv.is_file():
		raise FileNotFoundError(f'missing: {station_csv}')

	df_sta = pd.read_csv(station_csv)
	for c in ['station_code', 'receiver_index']:
		if c not in df_sta.columns:
			raise ValueError(f'station_synth.csv missing column: {c}')

	s_code = df_sta['station_code']
	if s_code.isna().any():
		raise ValueError('station_synth.csv has missing station_code')
	codes = s_code.astype(str).map(str.strip)
	if (codes == '').any():
		raise ValueError('station_synth.csv has empty station_code')

	stations_is_das = codes.str.upper().str.startswith('D').to_numpy(dtype=bool)

	ri = pd.to_numeric(df_sta['receiver_index'], errors='raise')
	if ri.isna().any():
		raise ValueError('station_synth.csv has missing receiver_index')

	vals = ri.to_numpy(float)
	if not np.isfinite(vals).all():
		raise ValueError('receiver_index has non-finite values')
	if not np.equal(vals, np.round(vals)).all():
		raise ValueError('receiver_index must be integer-valued')
	idx = vals.astype(int)

	if idx.size == 0:
		raise ValueError('station_synth.csv has no stations')
	if np.unique(idx).size != idx.size:
		raise ValueError('receiver_index has duplicates')

	require_filename_only(receiver_geometry, 'receiver_geometry')
	geom_path = dataset_dir / 'geometry' / receiver_geometry
	if not geom_path.is_file():
		raise FileNotFoundError(f'missing: {geom_path}')

	recv_xyz_m = np.load(geom_path).astype(float)
	if recv_xyz_m.ndim != 2 or recv_xyz_m.shape[1] != 3:
		raise ValueError(f'receiver geometry must be (N,3), got {recv_xyz_m.shape}')

	min_i = int(idx.min())
	max_i = int(idx.max())
	if min_i < 0 or max_i >= recv_xyz_m.shape[0]:
		raise IndexError(
			'receiver_index out of range: '
			f'min={min_i} max={max_i} receivers={recv_xyz_m.shape[0]}'
		)

	stations_xyz_m = recv_xyz_m[idx]
	return stations_xyz_m, stations_is_das


def run_qc(config_path: Path) -> None:
	if not config_path.is_file():
		raise FileNotFoundError(f'config not found: {config_path}')

	cfg = load_config(config_path)

	dataset_dir = Path(cfg.dataset_dir)
	require_abs(dataset_dir, 'dataset_dir')
	require_dirname_only(cfg.outputs_dir, 'outputs_dir')
	require_filename_only(cfg.receiver_geometry, 'receiver_geometry')

	runs_root = config_path.resolve().parent.parent / 'runs'
	run_dir = runs_root / cfg.outputs_dir
	if not run_dir.is_dir():
		raise FileNotFoundError(f'run_dir not found: {run_dir}')

	index_csv = dataset_dir / 'index.csv'
	if not index_csv.is_file():
		raise FileNotFoundError(f'missing: {index_csv}')

	eval_csv = run_dir / 'eval_metrics.csv'
	if not eval_csv.is_file():
		raise FileNotFoundError(f'missing: {eval_csv}')

	df = pd.read_csv(eval_csv)
	n_eval = len(df)
	n_truth = len(pd.read_csv(index_csv))

	required_cols = ['horiz_m', 'dz_m', 'err3d_m']
	for c in required_cols:
		if c not in df.columns:
			raise ValueError(f'eval_metrics.csv missing column: {c}')

	# 件数整合
	summary_lines = []
	summary_lines.append(f'run_dir: {run_dir}')
	summary_lines.append(f'truth events (index.csv): {n_truth}')
	summary_lines.append(f'eval rows (eval_metrics.csv): {n_eval}')

	# 基本統計
	stats = df[['horiz_m', 'dz_m', 'err3d_m']].describe(percentiles=[0.5, 0.9, 0.95])
	stats_path = run_dir / 'qc_basic_stats.csv'
	stats.to_csv(stats_path)

	# 外れ値トップ10
	cols_show = [
		c
		for c in [
			'seq',
			'event_id',
			'event_id_str',
			'horiz_m',
			'dz_m',
			'err3d_m',
			'RMS',
			'ERH',
			'ERZ',
		]
		if c in df.columns
	]
	outliers = df.sort_values('err3d_m', ascending=False).head(10)[cols_show]
	outliers_path = run_dir / 'qc_outliers_top10.csv'
	outliers.to_csv(outliers_path, index=False)

	# プロット（各図1枚、色指定なし）
	save_hist(
		df['err3d_m'], run_dir / 'qc_err3d_hist.png', '3D error histogram', 'err3d_m'
	)
	save_hist(
		df['horiz_m'],
		run_dir / 'qc_horiz_hist.png',
		'Horizontal error histogram',
		'horiz_m',
	)
	save_hist(df['dz_m'], run_dir / 'qc_dz_hist.png', 'Depth error histogram', 'dz_m')

	if 'dx_m' in df.columns and 'dy_m' in df.columns:
		save_dxdy_scatter(
			df['dx_m'].to_numpy(),
			df['dy_m'].to_numpy(),
			run_dir / 'qc_dxdy_scatter.png',
		)

	# テキスト要約
	qc_txt = run_dir / 'qc_summary.txt'
	qc_txt.write_text('\n'.join(summary_lines) + '\n', encoding='utf-8')

	print(stats)
	print(f'[OK] wrote: {stats_path}')
	print(f'[OK] wrote: {outliers_path}')
	print(f'[OK] wrote: {qc_txt}')

	# --- True vs HypoInverse (XY/XZ/YZ 3-view) + station locations ---
	need_xyz = ['x_m_true', 'y_m_true', 'z_m_true', 'x_m_hyp', 'y_m_hyp', 'z_m_hyp']
	missing_xyz = [c for c in need_xyz if c not in df.columns]
	if missing_xyz:
		raise KeyError(
			f'missing columns for 3-view plot: {missing_xyz}. available={list(df.columns)}'
		)

	true_xyz_m = df[['x_m_true', 'y_m_true', 'z_m_true']].to_numpy(float)
	pred_xyz_m = df[['x_m_hyp', 'y_m_hyp', 'z_m_hyp']].to_numpy(float)

	mask = np.isfinite(true_xyz_m).all(axis=1) & np.isfinite(pred_xyz_m).all(axis=1)
	true_xyz_m = true_xyz_m[mask]
	pred_xyz_m = pred_xyz_m[mask]

	stations_xyz_m, stations_is_das = _load_stations_from_run_output(
		run_dir=run_dir,
		dataset_dir=dataset_dir,
		receiver_geometry=cfg.receiver_geometry,
	)

	xy_png = run_dir / 'xy_true_vs_hyp.png'
	save_true_pred_xyz_3view(
		true_xyz_m,
		pred_xyz_m,
		xy_png,
		stations_xyz_m=stations_xyz_m,
		stations_is_das=stations_is_das,
		title='True vs HypoInverse (3-view)',
	)
	print(f'[OK] wrote: {xy_png}')
