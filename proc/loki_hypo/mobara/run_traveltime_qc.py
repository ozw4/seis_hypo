# %%
# ---- QC 実行側の最小イメージ ----
from __future__ import annotations

from pathlib import Path

from common.config import QcConfig

# import はあなたの構成に合わせてOK
from common.load_config import load_config
from qc.nonlinloc.traveltime_tables import run_traveltime_tables_qc

YAML_PATH = Path('/workspace/data/config/traveltime_config.yaml')
PRESET = 'mobara'


def main() -> None:
	cfg = load_config(QcConfig, YAML_PATH, PRESET)
	artifacts = run_traveltime_tables_qc(cfg, preset=PRESET)

	print('QC finished.')
	print('fig_dir:', cfg.fig_dir)
	for k, v in artifacts.items():
		print(k, '->', v)


if __name__ == '__main__':
	main()
