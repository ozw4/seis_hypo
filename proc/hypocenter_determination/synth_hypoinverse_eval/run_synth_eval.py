# %%
from __future__ import annotations

from pathlib import Path

from hypo.synth_eval.pipeline import run_synth_eval

CONFIG_PATH = (
	Path(__file__).resolve().parent / 'configs/example1_surface9_well1000m.yaml'
)


def main() -> None:
	runs_root = Path(__file__).resolve().parent / 'runs'
	run_dir, df_eval, stats = run_synth_eval(CONFIG_PATH, runs_root=runs_root)
	print(stats)
	print(f'[OK] run_dir: {run_dir}')
	run_dir = runs_root /
	eval_csv = run_dir / 'eval_metrics.csv'
	if not eval_csv.is_file():
		raise FileNotFoundError(f'missing: {eval_csv}')

	df_eval = pd.read_csv(eval_csv)
	out_png = run_dir / 'qc_xy_truth_vs_pred.png'
	plot_xy_truth_vs_pred_linked(df_eval, out_png)
	print(f'[OK] wrote: {out_png}')

