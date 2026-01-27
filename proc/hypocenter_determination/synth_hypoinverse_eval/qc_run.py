# %%
from __future__ import annotations

from pathlib import Path

from src.hypo.synth_eval.qc import run_qc

CONFIG_PATH = (
	Path(__file__).resolve().parent / 'configs/example1_surface9_well1000m.yaml'
)


def main() -> None:
	run_qc(CONFIG_PATH)


if __name__ == '__main__':
	main()
