from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkspaceRoots:
	workspace_root: Path
	data_root: Path
	external_root: Path
	proc_root: Path


@dataclass(frozen=True)
class JmaMobaraHypoinversePaths:
	sta_file: Path
	station_csv: Path
	pcrh_file: Path
	scrh_file: Path
	hypoinverse_exe: Path
	epicenter_csv: Path
	measurement_csv: Path
	prefecture_shp: Path
	plot_config_yaml: Path


@dataclass(frozen=True)
class JmaMobaraHypoinverseDasPaths:
	sta_file: Path
	station_csv: Path
	pcrh_file: Path
	scrh_file: Path
	hypoinverse_exe: Path
	epicenter_csv: Path
	measurement_csv: Path
	das_measurement_csv: Path
	das_epicenter_csv: Path
	prefecture_shp: Path
	plot_config_yaml: Path


@dataclass(frozen=True)
class LokiPlotPaths:
	plot_config_yaml: Path
	prefecture_shp: Path


def build_workspace_roots(workspace_root: Path) -> WorkspaceRoots:
	return WorkspaceRoots(
		workspace_root=workspace_root,
		data_root=workspace_root / 'data',
		external_root=workspace_root / 'external_source',
		proc_root=workspace_root / 'proc',
	)


def build_jma_mobara_hypoinverse_paths(
	roots: WorkspaceRoots,
) -> JmaMobaraHypoinversePaths:
	data_root = roots.data_root
	return JmaMobaraHypoinversePaths(
		sta_file=data_root / 'station' / 'jma' / 'stations_hypoinverse.sta',
		station_csv=data_root / 'station' / 'jma' / 'station.csv',
		pcrh_file=data_root / 'velocity' / 'jma_crh' / 'JMA2001A_P.crh',
		scrh_file=data_root / 'velocity' / 'jma_crh' / 'JMA2001A_S.crh',
		hypoinverse_exe=roots.external_root / 'hyp1.40' / 'hypoinverse.exe',
		epicenter_csv=data_root / 'arrivetime' / 'arrivetime_epicenters_mobara2020.csv',
		measurement_csv=data_root
		/ 'arrivetime'
		/ 'arrivetime_measurements_mobara2020.csv',
		prefecture_shp=data_root / 'N03-20240101_GML' / 'N03-20240101_prefecture.shp',
		plot_config_yaml=data_root / 'config' / 'plot_config.yaml',
	)


def build_loki_plot_paths(roots: WorkspaceRoots) -> LokiPlotPaths:
	data_root = roots.data_root
	return LokiPlotPaths(
		plot_config_yaml=data_root / 'config' / 'plot_config.yaml',
		prefecture_shp=data_root / 'N03-20240101_GML' / 'N03-20240101_prefecture.shp',
	)


def build_jma_mobara_hypoinverse_das_paths(
	roots: WorkspaceRoots,
) -> JmaMobaraHypoinverseDasPaths:
	data_root = roots.data_root
	return JmaMobaraHypoinverseDasPaths(
		sta_file=data_root / 'station' / 'jma' / 'stations_hypoinverse_with_das.sta',
		station_csv=data_root / 'station' / 'jma' / 'station_with_das.csv',
		pcrh_file=data_root / 'velocity' / 'jma_crh' / 'JMA2001A_P.crh',
		scrh_file=data_root / 'velocity' / 'jma_crh' / 'JMA2001A_S.crh',
		hypoinverse_exe=roots.external_root / 'hyp1.40' / 'hypoinverse.exe',
		epicenter_csv=data_root
		/ 'arrivetime'
		/ 'NIED'
		/ 'arrivetime_epicenters_mobara2020.csv',
		measurement_csv=data_root
		/ 'arrivetime'
		/ 'NIED'
		/ 'arrivetime_measurements_mobara2020.csv',
		das_measurement_csv=Path(
			'/home/dcuser/mobara2025/proc/proc_continuous_das/das_picks_20200215_20200301.csv'
		),
		das_epicenter_csv=Path(
			'/home/dcuser/mobara2025/proc/proc_continuous_das/events_summary_20200215_20200301.csv'
		),
		prefecture_shp=data_root / 'N03-20240101_GML' / 'N03-20240101_prefecture.shp',
		plot_config_yaml=data_root / 'config' / 'plot_config.yaml',
	)
