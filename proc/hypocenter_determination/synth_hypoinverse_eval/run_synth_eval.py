# %%
from __future__ import annotations

from pathlib import Path

from hypo.synth_eval.pipeline import run_synth_eval
from hypo.qc.plots import plot_xy_truth_vs_pred_linked
from hypo.synth_eval.metrics import evaluate

CONFIG_PATH = (
	Path(__file__).resolve().parent / 'configs/example1_surface9_well1000m.yaml'
)


def main() -> None:
	runs_root = Path(__file__).resolve().parent / 'runs'
	run_dir, df_eval, stats = run_synth_eval(CONFIG_PATH, runs_root=runs_root)
	print(stats)
	print(f'[OK] run_dir: {run_dir}')
	df_eval = evaluate(truth_df, prt_file, cfg.lat0, cfg.lon0)
	df_eval.to_csv(eval_csv, index=False)

	# === QC plot: truth vs pred (XY) ===
	out_png = run_dir / "qc_xy_truth_vs_pred.png"
	plot_xy_truth_vs_pred_linked(df_eval, out_png)

	stats = df_eval[['horiz_m', 'dz_m', 'err3d_m', 'RMS', 'ERH', 'ERZ']].describe(
		percentiles=[0.5, 0.9, 0.95]
	)
	stats.to_csv(eval_stats_csv)

	print(stats)
	print(f'[OK] wrote: {eval_csv}')
	print(f'[OK] wrote: {eval_stats_csv}')
	print(f'[OK] wrote: {out_png}')
これで python run_synth_eval.py 一発で、

if __name__ == '__main__':
	main()


