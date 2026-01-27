# %%
# file: proc/loki_hypo/run_loki_waveform_stacking_pipelines_eqt.py
from __future__ import annotations

from pathlib import Path

from common.config import (
	EqTInputs,
	LokiWaveformStackingInputs,
	LokiWaveformStackingPipelineConfig,
)
from common.load_config import load_config
from common.read_yaml import fieldnames, read_yaml_preset
from pipelines.loki_waveform_stacking_pipelines import (
	list_event_dirs_filtered,
	pipeline_loki_waveform_stacking_eqt,
)
from qc.loki.compare import run_loki_vs_jma_qc
from viz.loki.coherence_xy import plot_loki_event_coherence_xy_overlay
from qc.loki.waveforms_with_loki_picks import plot_waveforms_with_picks_for_event
from viz.plot_config import PlotConfig
from waveform.preprocess import spec_from_inputs

# ========= USER EDIT HERE =========
PIPELINE_YAML = Path('/workspace/data/config/loki_waveform_pipeline_eqt.yaml')
PIPELINE_PRESET = 'mobara_eqt'

RUN_QC_COMPARE = True
RUN_PLOT_WAVEFORMS_WITH_PICKS = True
RUN_PLOT_COHERENCE_XY = True

PLOT_CONFIG_YAML = Path('/workspace/data/config/plot_config.yaml')
PLOT_CONFIG_PRESET = 'mobara_default'

PLOT_COMPONENTS = ('U', 'N')  # waveforms plot components
Y_TIME = 'relative'  # "samples" | "absolute" | "relative"
# =================================


def _build_inputs_and_eqt(
	yaml_path: Path,
	preset: str,
) -> tuple[LokiWaveformStackingInputs, EqTInputs]:
	raw = read_yaml_preset(yaml_path, preset)

	allowed_inputs = fieldnames(LokiWaveformStackingInputs)
	allowed_eqt = fieldnames(EqTInputs)
	allowed_union = allowed_inputs | allowed_eqt

	unknown = sorted([k for k in raw.keys() if k not in allowed_union])
	if unknown:
		raise ValueError(f'プリセット "{preset}" に未知のキーがあります: {unknown}')

	inputs_kwargs = {k: v for k, v in raw.items() if k in allowed_inputs}
	eqt_kwargs = {k: v for k, v in raw.items() if k in allowed_eqt}

	return LokiWaveformStackingInputs(**inputs_kwargs), EqTInputs(**eqt_kwargs)


def main() -> None:
	cfg = load_config(
		LokiWaveformStackingPipelineConfig, PIPELINE_YAML, PIPELINE_PRESET
	)

	cfg.loki_data_path.mkdir(parents=True, exist_ok=True)
	cfg.loki_output_path.mkdir(parents=True, exist_ok=True)

	inputs_yaml = Path(cfg.inputs_yaml)
	inputs_preset = str(cfg.inputs_preset)

	# 同じpresetから2つのdataclassを読む（余計なキーは load_config 側で弾かれる）
	inputs, eqt = _build_inputs_and_eqt(inputs_yaml, inputs_preset)
	base_sampling_rate_hz = int(inputs.base_sampling_rate_hz)
	pre_spec = spec_from_inputs(inputs)

	header_path = Path(cfg.loki_db_path) / Path(cfg.loki_hdr_filename)
	print(header_path)
	if not header_path.is_file():
		raise FileNotFoundError(f'header not found: {header_path}')

	# ---- run LOKI (EqT direct_input; proc 側は src を呼ぶだけにする) ----
	pipeline_loki_waveform_stacking_eqt(
		cfg,
		inputs,
		eqt_weights=str(eqt.eqt_weights),
		eqt_in_samples=int(eqt.eqt_in_samples),
		eqt_overlap=int(eqt.eqt_overlap),
		eqt_batch_size=int(eqt.eqt_batch_size),
		channel_prefix=str(eqt.eqt_channel_prefix),
	)

	event_dirs = list_event_dirs_filtered(cfg)

	# ---- QC compare (optional) ----
	if RUN_QC_COMPARE:
		plot_cfg = load_config(PlotConfig, PLOT_CONFIG_YAML, PLOT_CONFIG_PRESET)
		allowed_event_ids = {p.name for p in event_dirs}

		run_loki_vs_jma_qc(
			base_input_dir=Path(cfg.base_input_dir),
			loki_output_dir=Path(cfg.loki_output_path),
			header_path=header_path,
			event_glob=cfg.event_glob,
			plot_cfg=plot_cfg,
			use_build_compare_df=True,
			compare_csv_out=Path(cfg.loki_output_path) / 'compare_jma_vs_loki.csv',
			allowed_event_ids=allowed_event_ids,
			out_png=Path(cfg.loki_output_path) / 'loki_vs_jma.png',
		)

	# ---- Per-event plots (optional) ----
	if RUN_PLOT_WAVEFORMS_WITH_PICKS or RUN_PLOT_COHERENCE_XY:
		for event_dir in event_dirs:
			if RUN_PLOT_WAVEFORMS_WITH_PICKS:
				plot_waveforms_with_picks_for_event(
					event_dir=event_dir,
					loki_output_dir=Path(cfg.loki_output_path),
					header_path=header_path,
					base_sampling_rate_hz=base_sampling_rate_hz,
					components_order=('U', 'N', 'E'),
					plot_components=PLOT_COMPONENTS,
					y_time=Y_TIME,
					pre_spec=pre_spec,
				)

			if RUN_PLOT_COHERENCE_XY:
				out_png = plot_loki_event_coherence_xy_overlay(
					event_dir=event_dir,
					loki_output_dir=Path(cfg.loki_output_path),
					header_path=header_path,
					trial=0,
					dpi=200,
				)
				if out_png is None:
					print(
						f'[WARN] no corrmatrix for event={event_dir.name}, '
						'skip coherence plot'
					)


if __name__ == '__main__':
	main()
