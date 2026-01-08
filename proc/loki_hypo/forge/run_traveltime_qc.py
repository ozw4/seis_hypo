# %%
from __future__ import annotations

from pathlib import Path

from common.config import QcConfig
from common.load_config import load_config
from qc.traveltime_pipelines_qc import run_traveltime_tables_qc

YAML_PATH = Path('/workspace/data/config/traveltime_config.yaml')

# strict対策：QcConfigで読むのはQC用preset
PRESET_CFG = 'forge44_vpvs1p75_qc'

# 出力先(qc/<preset>)の名前は本体側に合わせたいならこっち
PRESET_OUT = 'forge44_vpvs1p75'


def main() -> None:
	cfg = load_config(QcConfig, YAML_PATH, PRESET_CFG)
	artifacts = run_traveltime_tables_qc(cfg, preset=PRESET_OUT)

	print('QC finished.')
	print('fig_dir:', cfg.fig_dir)
	for k, v in artifacts.items():
		print(k, '->', v)


if __name__ == '__main__':
	main()
