# %%
from __future__ import annotations

import shutil
from pathlib import Path

from common.config import LokiWaveformStackingInputs, LokiWaveformStackingPipelineConfig
from common.load_config import load_config
from common.run_snapshot import save_many_yaml_and_effective
from pipelines.loki_waveform_stacking_pipelines import (
	list_event_dirs_filtered,
	pipeline_loki_waveform_stacking,
)
from qc.loki.compare import run_loki_vs_jma_qc
from viz.loki.coherence_xy import plot_loki_event_coherence_xy_overlay
from qc.loki.waveforms_with_loki_picks import plot_waveforms_with_picks_for_event
from viz.plot_config import PlotConfig
from waveform.preprocess import spec_from_inputs


def main() -> None:
	pipeline_yaml = Path('/workspace/data/config/loki_waveform_pipeline.yaml')
	pipeline_preset = 'mobara'

	cfg = load_config(
		LokiWaveformStackingPipelineConfig, pipeline_yaml, pipeline_preset
	)
	print(cfg)

	if not cfg.inputs_yaml.is_file():
		raise FileNotFoundError(f'inputs_yaml not found: {cfg.inputs_yaml}')

	inputs = load_config(LokiWaveformStackingInputs, cfg.inputs_yaml, cfg.inputs_preset)

	save_many_yaml_and_effective(
		out_dir=cfg.loki_output_path,
		items=[
			('pipeline', pipeline_yaml, pipeline_preset, cfg),
			('inputs', cfg.inputs_yaml, cfg.inputs_preset, inputs),
		],
	)

	# ---- LOKI 本体 ----
	pipeline_loki_waveform_stacking(cfg, inputs)

	# ---- 追加: プロット（フィルタ済みイベントのみ）----
	plot_components = ('U', 'N')
	y_time = 'relative'  # "samples" | "absolute" | "relative"

	header_path = Path(cfg.loki_db_path) / Path(cfg.loki_hdr_filename)
	if not header_path.is_file():
		raise FileNotFoundError(f'header not found: {header_path}')

	pre_spec = spec_from_inputs(inputs)

	event_dirs = list_event_dirs_filtered(cfg)
	for event_dir in event_dirs:
		plot_waveforms_with_picks_for_event(
			event_dir=event_dir,
			loki_output_dir=Path(cfg.loki_output_path),
			header_path=header_path,
			base_sampling_rate_hz=int(inputs.base_sampling_rate_hz),
			components_order=('U', 'N', 'E'),
			plot_components=plot_components,
			y_time=y_time,
			pre_spec=pre_spec,
		)
		out_png = plot_loki_event_coherence_xy_overlay(
			event_dir=event_dir,
			loki_output_dir=Path(cfg.loki_output_path),
			header_path=header_path,
			trial=0,
			dpi=200,
		)
		if out_png is None:
			print(
				f'[WARN] no corrmatrix for event={event_dir.name}, skip coherence plot'
			)
	print(f'Waveform plots written under: {cfg.loki_output_path}')
	# ---- LOKI vs JMA QC ----
	plot_config_yaml = Path('/workspace/data/config/plot_config.yaml')
	plot_setting = 'mobara_default'
	plot_cfg = load_config(PlotConfig, plot_config_yaml, plot_setting)

	run_loki_vs_jma_qc(
		base_input_dir=Path(cfg.base_input_dir),
		loki_output_dir=Path(cfg.loki_output_path),
		header_path=header_path,
		event_glob=cfg.event_glob,
		plot_cfg=plot_cfg,
		use_build_compare_df=True,
		compare_csv_out=Path(cfg.loki_output_path) / 'compare_jma_vs_loki.csv',
		allowed_event_ids={p.name for p in event_dirs},
		out_png=Path(cfg.loki_output_path) / 'loki_vs_jma.png',
	)

	# ---- cleanup ----
	loki_data_path = Path(cfg.loki_data_path)
	loki_output_path = Path(cfg.loki_output_path)
	if loki_data_path.is_dir():
		try:
			loki_data_path.relative_to(loki_output_path)
		except ValueError:
			raise RuntimeError(
				f'refusing to delete loki_data_path outside loki_output_path: {loki_data_path}'
			)
		shutil.rmtree(loki_data_path)


if __name__ == '__main__':
	main()
