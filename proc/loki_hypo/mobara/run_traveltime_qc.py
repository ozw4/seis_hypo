# %%
# ---- QC 実行側の最小イメージ ----
from __future__ import annotations

import logging
from pathlib import Path

from common.config import QcConfig

# import はあなたの構成に合わせてOK
from common.load_config import load_config
from qc.nonlinloc.traveltime_tables import run_traveltime_tables_qc

logger = logging.getLogger(__name__)

YAML_PATH = Path('/workspace/data/config/traveltime_config.yaml')
PRESET = 'mobara'


def _configure_logging() -> None:
	if logging.getLogger().handlers:
		return
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s %(levelname)s %(name)s: %(message)s',
	)


def main() -> None:
	_configure_logging()
	cfg = load_config(QcConfig, YAML_PATH, PRESET)
	artifacts = run_traveltime_tables_qc(cfg, preset=PRESET)

	logger.info('QC finished.')
	logger.info('fig_dir: %s', cfg.fig_dir)
	for k, v in artifacts.items():
		logger.info('%s -> %s', k, v)


if __name__ == '__main__':
	main()
