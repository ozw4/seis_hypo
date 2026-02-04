# %%
from __future__ import annotations

from pathlib import Path

from common.config import expand_dt_pick_error_experiments, load_dt_pick_error_config_v1
from pipelines.jma_dt_pick_error_table import run_jma_dt_pick_error_table


def main() -> None:
	pipeline_yaml = Path(
		'/workspace/proc/prepare_data/jma/config/dt_pick_error_pipeline.yaml'
	)
	cfg = load_dt_pick_error_config_v1(pipeline_yaml)
	runs = expand_dt_pick_error_experiments(cfg)
	for i, run_cfg in enumerate(runs, start=1):
		print(
			f'[run] {i}/{len(runs)} run_id={run_cfg.run.run_id} '
			f'out_dir={run_cfg.run.out_dir}'
		)
		run_jma_dt_pick_error_table(
			run_cfg,
			yaml_path=pipeline_yaml,
			preset='v1',
			continue_on_event_error=True,
			log_warnings=True,
			log_filename='run.log',
		)


if __name__ == '__main__':
	main()

# %%
