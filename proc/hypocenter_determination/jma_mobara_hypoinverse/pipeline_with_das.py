from __future__ import annotations

from pathlib import Path

from common.paths import (
	build_jma_mobara_hypoinverse_das_paths,
	build_workspace_roots,
)
from hypo.jma_mobara_hypoinverse_das import (
	JmaWithDasHypoinverseDasFilter,
	JmaWithDasHypoinverseDasPhase,
	JmaWithDasHypoinverseInitialEvent,
	JmaWithDasHypoinversePlot,
	JmaWithDasHypoinverseRunConfig,
	JmaWithDasHypoinverseRunPaths,
	JmaWithDasHypoinverseSweep,
	JmaWithDasHypoinverseTimeFilter,
	run_parameter_sweep,
)

THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = Path('/workspace')
ROOTS = build_workspace_roots(WORKSPACE_ROOT)
PATHS = build_jma_mobara_hypoinverse_das_paths(ROOTS)
CMD_TEMPLATE_FILE = THIS_DIR / 'template' / 'jma2001a_with_das.cmd'

RUN_DIR = Path('./result/test_mobara2020_jma_with_das')

CONFIG = JmaWithDasHypoinverseRunConfig(
	paths=JmaWithDasHypoinverseRunPaths(
		sta_file=PATHS.sta_file,
		station_csv=PATHS.station_csv,
		pcrh_file=PATHS.pcrh_file,
		scrh_file=PATHS.scrh_file,
		hypoinverse_exe=PATHS.hypoinverse_exe,
		cmd_template_file=CMD_TEMPLATE_FILE,
		epicenter_csv=PATHS.epicenter_csv,
		measurement_csv=PATHS.measurement_csv,
		das_measurement_csv=PATHS.das_measurement_csv,
		das_epicenter_csv=PATHS.das_epicenter_csv,
		prefecture_shp=PATHS.prefecture_shp,
		plot_config_yaml=PATHS.plot_config_yaml,
		run_dir=RUN_DIR,
	),
	time_filter=JmaWithDasHypoinverseTimeFilter(
		target_start='2020-02-15 00:00:00',
		target_end='2020-03-02 00:00:00',
		max_das_score=1,
	),
	initial_event=JmaWithDasHypoinverseInitialEvent(
		use_jma_flag=False,
		fix_depth=False,
		default_depth_km=10.0,
		p_centroid_top_n=5,
		origin_time_offset_sec=3.0,
	),
	das_filter=JmaWithDasHypoinverseDasFilter(
		dt_sec=0.01,
		fiber_spacing_m=1.0,
		channel_start=200,
		win_half_samples=500,
		residual_thresh_s=0.05,
		decimation_base_spacing_m=500.0,
	),
	das_phase=JmaWithDasHypoinverseDasPhase(
		max_dt_sec=10.0,
	),
	sweep=JmaWithDasHypoinverseSweep(
		das_total_weights=(1, 2, 3),
		use_das_channels=(5, 10, 20, 50, 100, 500),
	),
	plot=JmaWithDasHypoinversePlot(
		plot_setting='mobara_default',
	),
)


def main() -> None:
	run_parameter_sweep(CONFIG, script_path=Path(__file__).resolve())


if __name__ == '__main__':
	main()
