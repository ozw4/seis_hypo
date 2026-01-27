# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.collections import LineCollection

CONFIG_PATH = (
	Path(__file__).resolve().parent / 'configs/example1_surface9_well1000m.yaml'
)


@dataclass(frozen=True)
class Config:
	dataset_dir: str
	outputs_dir: str


def save_true_pred_xy_plot(
	true_xy: np.ndarray,
	pred_xy: np.ndarray,
	out_png: str,
	*,
	title: str | None = None,
	xlabel: str = 'X',
	ylabel: str = 'Y',
	marker_size: float = 20.0,
) -> None:
	true_xy = np.asarray(true_xy, dtype=float)
	pred_xy = np.asarray(pred_xy, dtype=float)

	if true_xy.shape != pred_xy.shape or true_xy.ndim != 2 or true_xy.shape[1] != 2:
		raise ValueError(
			f'true_xy/pred_xy は同じ形の (N,2) が必要です: true={true_xy.shape}, pred={pred_xy.shape}'
		)

	segs = np.stack([true_xy, pred_xy], axis=1)  # (N, 2, 2)

	fig, ax = plt.subplots(figsize=(7, 7))

	ax.scatter(true_xy[:, 0], true_xy[:, 1], s=marker_size, marker='o', label='True')
	ax.scatter(pred_xy[:, 0], pred_xy[:, 1], s=marker_size, marker='x', label='Pred')

	lc = LineCollection(segs, linestyles='dotted', linewidths=1.0, alpha=0.7)
	ax.add_collection(lc)

	all_xy = np.vstack([true_xy, pred_xy])
	x_min, x_max = float(all_xy[:, 0].min()), float(all_xy[:, 0].max())
	y_min, y_max = float(all_xy[:, 1].min()), float(all_xy[:, 1].max())

	x_pad = (x_max - x_min) * 0.05
	y_pad = (y_max - y_min) * 0.05
	if x_pad == 0.0:
		x_pad = 1.0
	if y_pad == 0.0:
		y_pad = 1.0

	ax.set_xlim(x_min - x_pad, x_max + x_pad)
	ax.set_ylim(y_min - y_pad, y_max + y_pad)
	ax.set_aspect('equal', adjustable='box')
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	if title:
		ax.set_title(title)
	ax.grid(True)
	ax.legend()

	fig.tight_layout()
	fig.savefig(out_png, dpi=200)
	plt.close(fig)


def _plot_xy_true_vs_hyp(df_eval: pd.DataFrame, out_png: Path) -> None:
	need = ['x_m_true', 'y_m_true', 'x_m_hyp', 'y_m_hyp']
	missing = [c for c in need if c not in df_eval.columns]
	if missing:
		raise KeyError(f'missing columns: {missing}. available={list(df_eval.columns)}')

	x0 = df_eval['x_m_true'].to_numpy(float)
	y0 = df_eval['y_m_true'].to_numpy(float)
	x1 = df_eval['x_m_hyp'].to_numpy(float)
	y1 = df_eval['y_m_hyp'].to_numpy(float)

	mask = np.isfinite(x0) & np.isfinite(y0) & np.isfinite(x1) & np.isfinite(y1)
	x0, y0, x1, y1 = x0[mask], y0[mask], x1[mask], y1[mask]

	fig, ax = plt.subplots()
	ax.scatter(x0, y0, label='True')
	ax.scatter(x1, y1, label='HypoInverse')

	for a, b, c, d in zip(x0, y0, x1, y1):
		ax.plot([a, c], [b, d], linestyle=':', linewidth=0.8)  # 点線リンク

	ax.set_xlabel('X (m)')
	ax.set_ylabel('Y (m)')
	ax.set_aspect('equal', adjustable='datalim')
	ax.legend()
	fig.tight_layout()

	out_png.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_png, dpi=200)
	plt.close(fig)


def _load_config(path: Path) -> Config:
	obj = yaml.safe_load(path.read_text(encoding='utf-8'))
	return Config(
		dataset_dir=str(obj['dataset_dir']),
		outputs_dir=str(obj['outputs_dir']),
	)


def _require_abs(p: Path, key: str) -> None:
	if not p.is_absolute():
		raise ValueError(f'{key} must be an absolute path: {p}')


def _require_dirname_only(name: str, key: str) -> None:
	if '/' in name or '\\' in name:
		raise ValueError(f'{key} must be directory name only: {name}')
	if name.strip() == '':
		raise ValueError(f'{key} must be non-empty')


def _save_hist(series: pd.Series, out_png: Path, title: str, xlabel: str) -> None:
	plt.figure()
	plt.hist(series.dropna().to_numpy(), bins=60)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel('count')
	plt.tight_layout()
	plt.savefig(out_png, dpi=150)
	plt.close()


def main() -> None:
	if not CONFIG_PATH.is_file():
		raise FileNotFoundError(f'config not found: {CONFIG_PATH}')

	cfg = _load_config(CONFIG_PATH)

	dataset_dir = Path(cfg.dataset_dir)
	_require_abs(dataset_dir, 'dataset_dir')
	_require_dirname_only(cfg.outputs_dir, 'outputs_dir')

	run_dir = Path(__file__).resolve().parent / 'runs' / cfg.outputs_dir
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
	_save_hist(
		df['err3d_m'], run_dir / 'qc_err3d_hist.png', '3D error histogram', 'err3d_m'
	)
	_save_hist(
		df['horiz_m'],
		run_dir / 'qc_horiz_hist.png',
		'Horizontal error histogram',
		'horiz_m',
	)
	_save_hist(df['dz_m'], run_dir / 'qc_dz_hist.png', 'Depth error histogram', 'dz_m')

	if 'dx_m' in df.columns and 'dy_m' in df.columns:
		plt.figure()
		plt.scatter(df['dx_m'].to_numpy(), df['dy_m'].to_numpy(), s=10)
		plt.axhline(0)
		plt.axvline(0)
		plt.title('dx vs dy')
		plt.xlabel('dx_m')
		plt.ylabel('dy_m')
		plt.tight_layout()
		plt.savefig(run_dir / 'qc_dxdy_scatter.png', dpi=150)
		plt.close()

	# テキスト要約
	qc_txt = run_dir / 'qc_summary.txt'
	qc_txt.write_text('\n'.join(summary_lines) + '\n', encoding='utf-8')

	print(stats)
	print(f'[OK] wrote: {stats_path}')
	print(f'[OK] wrote: {outliers_path}')
	print(f'[OK] wrote: {qc_txt}')
	xy_png = run_dir / 'xy_true_vs_hyp.png'
	_plot_xy_true_vs_hyp(df, xy_png)
	print(f'[OK] wrote: {xy_png}')


if __name__ == '__main__':
	main()
