# %%
from __future__ import annotations

import shutil

from common.config import (
	LokiWaveformStackingInputs,
	LokiWaveformStackingPipelineConfig,
)
from common.load_config import load_config
from pipelines.loki_waveform_stacking_pipelines import pipeline_loki_waveform_stacking


def main() -> None:
	pipeline_yaml = '/workspace/data/config/loki_waveform_pipeline.yaml'
	preset = 'mobara'

	cfg = load_config(LokiWaveformStackingPipelineConfig, pipeline_yaml, preset)
	print(cfg)
	if not cfg.inputs_yaml.is_file():
		raise FileNotFoundError(f'inputs_yaml not found: {cfg.inputs_yaml}')

	inputs = load_config(LokiWaveformStackingInputs, cfg.inputs_yaml, cfg.inputs_preset)

	pipeline_loki_waveform_stacking(cfg, inputs)

	# 終了後に loki_data_path を削除（= proc/inputs/loki_events を掃除）
	# ※ cfg.loki_data_path が base_input_dir 等を指していないことは設定で保証すること
	if cfg.loki_data_path.is_dir():
		shutil.rmtree(cfg.loki_data_path)


if __name__ == '__main__':
	main()
