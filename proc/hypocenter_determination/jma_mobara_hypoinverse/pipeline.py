from __future__ import annotations

from pathlib import Path

from hypo.jma_mobara_hypoinverse_jmaonly import (
	JmaOnlyHypoinverseInitialEvent,
	JmaOnlyHypoinversePaths,
	JmaOnlyHypoinverseRunConfig,
	JmaOnlyHypoinverseTimeFilter,
	run_pipeline,
)

THIS_DIR = Path(__file__).resolve().parent

STA_FILE = Path('/workspace/data/station/jma/stations_hypoinverse.sta')
STATION_CSV = Path('/workspace/data/station/jma/station.csv')
PCRH_FILE = Path('/workspace/data/velocity/jma_crh/JMA2001A_P.crh')
SCRH_FILE = Path('/workspace/data/velocity/jma_crh/JMA2001A_S.crh')
EXE_FILE = Path('/workspace/external_source/hyp1.40/hypoinverse.exe')
CMD_TEMPLATE_FILE = THIS_DIR / 'template' / 'jma2001a.cmd'

EPICENTER_CSV = Path('/workspace/data/arrivetime/arrivetime_epicenters_mobara2020.csv')
MEASUREMENT_CSV = Path(
	'/workspace/data/arrivetime/arrivetime_measurements_mobara2020.csv'
)
PREFECTURE_SHP = Path('/workspace/data/N03-20240101_GML/N03-20240101_prefecture.shp')
PLOT_CONFIG_YAML = Path('/workspace/data/config/plot_config.yaml')
PLOT_SETTING = 'mobara_default'

RUN_DIR = Path('./result/test_mobara2020_jmaonly')

CONFIG = JmaOnlyHypoinverseRunConfig(
	paths=JmaOnlyHypoinversePaths(
		sta_file=STA_FILE,
		station_csv=STATION_CSV,
		pcrh_file=PCRH_FILE,
		scrh_file=SCRH_FILE,
		exe_file=EXE_FILE,
		cmd_template_file=CMD_TEMPLATE_FILE,
		epicenter_csv=EPICENTER_CSV,
		measurement_csv=MEASUREMENT_CSV,
		prefecture_shp=PREFECTURE_SHP,
		plot_config_yaml=PLOT_CONFIG_YAML,
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
