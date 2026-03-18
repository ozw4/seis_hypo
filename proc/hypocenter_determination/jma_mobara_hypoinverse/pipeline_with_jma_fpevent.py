from __future__ import annotations

import argparse
from pathlib import Path

from hypo.jma_mobara_hypoinverse_config import load_jma_mobara_hypoinverse_config
from hypo.jma_mobara_hypoinverse_fpevent import (
	JmaFpEventHypoinverseDasFilter,
	JmaFpEventHypoinverseDasPhase,
	JmaFpEventHypoinverseInitialEvent,
	JmaFpEventHypoinversePlot,
	JmaFpEventHypoinverseRunConfig,
	JmaFpEventHypoinverseRunPaths,
	JmaFpEventHypoinverseTimeFilter,
	run_pipeline,
)

THIS_DIR = Path(__file__).resolve().parent
CONFIG_PATH = THIS_DIR / 'configs' / 'pipeline_with_jma_fpevent.example.yaml'


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			'Run the JMA Mobara Hypoinverse pipeline with ML picks + DAS picks.\n'
			'Example:\n'
			'  python '
			'proc/hypocenter_determination/jma_mobara_hypoinverse/'
			'pipeline_with_jma_fpevent.py '
			'--config '
			'proc/hypocenter_determination/jma_mobara_hypoinverse/configs/'
			'pipeline_with_jma_fpevent.example.yaml'
		),
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	parser.add_argument(
		'--config',
		type=Path,
		default=CONFIG_PATH,
		help='pipeline config yaml path',
	)
	return parser.parse_args()


def _load_run_config(config_path: Path) -> JmaFpEventHypoinverseRunConfig:
	legacy_config = load_jma_mobara_hypoinverse_config(config_path)

	return JmaFpEventHypoinverseRunConfig(
		paths=JmaFpEventHypoinverseRunPaths(
			sta_file=legacy_config.paths.sta_file,
			pcrh_file=legacy_config.paths.pcrh_file,
			scrh_file=legacy_config.paths.scrh_file,
			hypoinverse_exe=legacy_config.paths.exe_file,
			cmd_template_file=legacy_config.paths.cmd_file,
			measurement_csv=legacy_config.paths.measurement_csv,
			das_measurement_csv=legacy_config.paths.das_measurement_csv,
			das_epicenter_csv=legacy_config.paths.das_epicenter_csv,
			pick_npz=legacy_config.paths.pick_npz,
			station_with_das_csv=legacy_config.paths.station_with_das_csv,
			prefecture_shp=legacy_config.paths.prefecture_shp,
			plot_config_yaml=legacy_config.paths.plot_config_yaml,
			run_dir=legacy_config.paths.run_dir,
		),
		time_filter=JmaFpEventHypoinverseTimeFilter(
			target_start=legacy_config.time_filter.target_start,
			target_end=legacy_config.time_filter.target_end,
		),
		initial_event=JmaFpEventHypoinverseInitialEvent(
			use_jma_flag=legacy_config.initial_event.use_jma_flag,
			fix_depth=legacy_config.initial_event.fix_depth,
			default_depth_km=legacy_config.initial_event.default_depth_km,
			p_centroid_top_n=legacy_config.initial_event.p_centroid_top_n,
			origin_time_offset_sec=legacy_config.initial_event.origin_time_offset_sec,
		),
		das_filter=JmaFpEventHypoinverseDasFilter(
			dt_sec=legacy_config.das_filter.dt_sec,
			fiber_spacing_m=legacy_config.das_filter.fiber_spacing_m,
			channel_start=legacy_config.das_filter.channel_start,
			win_half_samples=legacy_config.das_filter.win_half_samples,
			residual_thresh_s=legacy_config.das_filter.residual_thresh_s,
			spacing_m=legacy_config.das_filter.spacing_m,
		),
		das_phase=JmaFpEventHypoinverseDasPhase(
			max_dt_sec=legacy_config.das_phase.max_dt_sec,
		),
		plot=JmaFpEventHypoinversePlot(
			plot_setting=legacy_config.plot.plot_setting,
			max_erh_km=legacy_config.plot_quality_filter.max_erh_km,
			max_erz_km=legacy_config.plot_quality_filter.max_erz_km,
			max_origin_time_err_sec=legacy_config.plot_quality_filter.max_origin_time_err_sec,
		),
	)


def main() -> None:
	args = parse_args()
	config_path = args.config.expanduser().resolve()
	run_pipeline(
		_load_run_config(config_path),
		script_path=Path(__file__).resolve(),
		config_path=config_path,
	)


if __name__ == '__main__':
	main()
