# %%
from __future__ import annotations

from pathlib import Path

from common.config import TravelTimePipelineConfig
from common.load_config import load_config
from pipelines.traveltime_pipelines import (
	run_traveltime_pipeline,
)

# ---- YAML preset ----
YAML_PATH = Path('/workspace/data/config/traveltime_config.yaml')
PRESET = 'mobara'


def main() -> None:
	cfg = load_config(TravelTimePipelineConfig, YAML_PATH, PRESET)
	result = run_traveltime_pipeline(cfg)

	print('TravelTime pipeline finished.')
	print(f'Stations: {len(result.stations_df)}')
	print(f'Grid: nx={result.grid.nx}, ny={result.grid.ny}, nz={result.grid.nz}')
	print(f'LAYER: {result.layers_path}')
	print(f'LOKI header: {result.loki_header_path}')
	print(f'Control P: {result.control_p_path}')
	print(f'Control S: {result.control_s_path}')


if __name__ == '__main__':
	main()
