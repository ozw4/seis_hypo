# %%
from __future__ import annotations

from pathlib import Path

from hypo.synth_eval.pipeline import run_synth_eval
from hypo.synth_eval.qc import run_qc

CONFIG_PATH = (
	Path(__file__).resolve().parent / 'configs/example1_surface9_das1_well1000m.yaml'
)


def main() -> None:
	runs_root = Path(__file__).resolve().parent / 'runs'
	run_dir, df_eval, stats = run_synth_eval(CONFIG_PATH, runs_root=runs_root)
	print(stats)
	print(f'[OK] run_dir: {run_dir}')
	run_qc(CONFIG_PATH)


if __name__ == '__main__':
	main()
