from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from hypo.synth_eval.validation import require_abs, require_dirname_only
from viz.hypo.synth_eval import plot_xy_true_vs_hyp, save_dxdy_scatter, save_hist


@dataclass(frozen=True)
class Config:
	dataset_dir: str
	outputs_dir: str


def load_config(path: Path) -> Config:
	obj = yaml.safe_load(path.read_text(encoding='utf-8'))
	return Config(
		dataset_dir=str(obj['dataset_dir']),
		outputs_dir=str(obj['outputs_dir']),
	)


def run_qc(config_path: Path) -> None:
	if not config_path.is_file():
		raise FileNotFoundError(f'config not found: {config_path}')

	cfg = load_config(config_path)

	dataset_dir = Path(cfg.dataset_dir)
	require_abs(dataset_dir, 'dataset_dir')
	require_dirname_only(cfg.outputs_dir, 'outputs_dir')

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
	xy_png = run_dir / 'xy_true_vs_hyp.png'
	plot_xy_true_vs_hyp(df, xy_png)
	print(f'[OK] wrote: {xy_png}')
