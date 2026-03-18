# %%
from __future__ import annotations

import logging
from pathlib import Path

from common.config import TravelTimeBaseConfig
from common.load_config import load_config
from pipelines.traveltime_pipelines import (
	run_traveltime_pipeline,
)

logger = logging.getLogger(__name__)

# ---- YAML preset ----
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
	cfg = load_config(TravelTimeBaseConfig, YAML_PATH, PRESET)
	result = run_traveltime_pipeline(cfg)

	logger.info('TravelTime pipeline finished.')
	logger.info('Stations: %d', len(result.stations_df))
	logger.info(
		'Grid: nx=%d, ny=%d, nz=%d',
		result.grid.nx,
		result.grid.ny,
		result.grid.nz,
	)
	logger.info('LAYER: %s', result.layers_path)
	logger.info('LOKI header: %s', result.loki_header_path)
	logger.info('Control P: %s', result.control_p_path)
	logger.info('Control S: %s', result.control_s_path)


if __name__ == '__main__':
	main()
