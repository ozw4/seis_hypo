from __future__ import annotations

from pathlib import Path

from common.paths import (
	build_jma_mobara_hypoinverse_paths,
	build_workspace_roots,
)
from hypo.jma_mobara_hypoinverse_jmaonly import (
	JmaOnlyHypoinverseInitialEvent,
	JmaOnlyHypoinverseRunPaths,
	JmaOnlyHypoinverseRunConfig,
	JmaOnlyHypoinverseTimeFilter,
	run_pipeline,
)

THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = Path('/workspace')
ROOTS = build_workspace_roots(WORKSPACE_ROOT)
PATHS = build_jma_mobara_hypoinverse_paths(ROOTS)
CMD_TEMPLATE_FILE = THIS_DIR / 'template' / 'jma2001a.cmd'
PLOT_SETTING = 'mobara_default'

RUN_DIR = Path('./result/test_mobara2020_jmaonly')

CONFIG = JmaOnlyHypoinverseRunConfig(
	paths=JmaOnlyHypoinverseRunPaths(
		sta_file=PATHS.sta_file,
		station_csv=PATHS.station_csv,
		pcrh_file=PATHS.pcrh_file,
		scrh_file=PATHS.scrh_file,
		hypoinverse_exe=PATHS.hypoinverse_exe,
		cmd_template_file=CMD_TEMPLATE_FILE,
		epicenter_csv=PATHS.epicenter_csv,
		measurement_csv=PATHS.measurement_csv,
		prefecture_shp=PATHS.prefecture_shp,
		plot_config_yaml=PATHS.plot_config_yaml,
		run_dir=RUN_DIR,
	),
	time_filter=JmaOnlyHypoinverseTimeFilter(
		target_start='2020-02-15 00:00:00',
		target_end='2020-03-02 00:00:00',
		max_das_score=1,
	),
	initial_event=JmaOnlyHypoinverseInitialEvent(
		use_jma_flag=False,
		fix_depth=False,
		default_depth_km=10.0,
		p_centroid_top_n=5,
		origin_time_offset_sec=3.0,
	),
	plot_setting=PLOT_SETTING,
)


def main() -> None:
	run_pipeline(CONFIG, script_path=Path(__file__).resolve())


if __name__ == '__main__':
	main()
