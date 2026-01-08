# file: proc/loki_hypo/run_loki_waveform_stacking_pipelines_stalta_das_pass1.py
# %%
from __future__ import annotations

from pathlib import Path

from common.config import LokiWaveformStackingInputs, LokiWaveformStackingPipelineConfig
from common.load_config import load_config
from pipelines.loki_stalta_pipelines import (
	pipeline_loki_waveform_stacking_stalta_pass1,
)

# ========= USER EDIT HERE =========
PIPELINE_YAML = Path('/workspace/data/config/loki_waveform_pipeline.yaml')
PIPELINE_PRESET = 'forge_das'

# DASイベントは cut_events_fromzarr_for_loki.py の出力 root 配下に
#   event_000001/
#     waveform.npy
#     meta.json
#     stations.csv
# が並ぶ前提。
# cfg.base_input_dir と cfg.event_glob(推奨: "event_*") は YAML 側で設定してね。

PASS1_OUT = 'pass1_stalta_p_das'
TRIAL = 0
PICK_JSON_NAME = 'pass1_picks_trial0.json'

# DASは1成分（Z想定）
COMPONENT = 'Z'
DAS_CHANNEL_CODE = 'DASZ'  # build_stream_from_forge_event_npy の channel_code
CHANNEL_PREFIX = 'HH'  # LOKI direct_input用 (例: HHP)
# =================================


def main() -> None:
	cfg = load_config(
		LokiWaveformStackingPipelineConfig, PIPELINE_YAML, PIPELINE_PRESET
	)
	inputs = load_config(LokiWaveformStackingInputs, cfg.inputs_yaml, cfg.inputs_preset)

	pick_json_by_event = pipeline_loki_waveform_stacking_stalta_pass1(
		cfg,
		inputs,
		component=str(COMPONENT),
		das_channel_code=str(DAS_CHANNEL_CODE),
		channel_prefix=str(CHANNEL_PREFIX),
		output_subdir=str(PASS1_OUT),
		trial=int(TRIAL),
		pick_json_name=str(PICK_JSON_NAME),
	)

	out_dir = Path(cfg.loki_output_path) / str(PASS1_OUT)
	print(
		f'[DONE] pass1(P-run) finished: out_dir={out_dir} n_events={len(pick_json_by_event)}'
	)


if __name__ == '__main__':
	main()
