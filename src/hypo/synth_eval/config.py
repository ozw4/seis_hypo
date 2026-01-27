from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class QcConfig:
	dataset_dir: str
	outputs_dir: str


@dataclass(frozen=True)
class PipelineConfig:
	dataset_dir: str
	sim_yaml: str
	outputs_dir: str
	template_cmd: str
	hypoinverse_exe: str
	receiver_geometry: str

	station_set: str
	lat0: float
	lon0: float
	origin0: str
	dt_sec: float
	max_events: int
	default_depth_km: float
	fix_depth: bool

	arc_use_jma_flag: bool
	arc_p_centroid_top_n: int
	arc_origin_time_offset_sec: float


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
	if not path.is_file():
		raise FileNotFoundError(f'config not found: {path}')
	obj = yaml.safe_load(path.read_text(encoding='utf-8'))
	if not isinstance(obj, dict):
		raise ValueError(f'config must be mapping: {path}')
	return obj


def load_qc_config(path: Path) -> QcConfig:
	obj = _read_yaml_mapping(path)
	return QcConfig(
		dataset_dir=str(obj['dataset_dir']),
		outputs_dir=str(obj['outputs_dir']),
	)


def load_pipeline_config(path: Path) -> PipelineConfig:
	obj = _read_yaml_mapping(path)
	return PipelineConfig(
		dataset_dir=str(obj['dataset_dir']),
		sim_yaml=str(obj['sim_yaml']),
		outputs_dir=str(obj['outputs_dir']),
		template_cmd=str(obj['template_cmd']),
		hypoinverse_exe=str(obj['hypoinverse_exe']),
		receiver_geometry=str(obj['receiver_geometry']),
		station_set=str(obj['station_set']),
		lat0=float(obj['lat0']),
		lon0=float(obj['lon0']),
		origin0=str(obj['origin0']),
		dt_sec=float(obj['dt_sec']),
		max_events=int(obj['max_events']),
		default_depth_km=float(obj['default_depth_km']),
		fix_depth=bool(obj['fix_depth']),
		arc_use_jma_flag=bool(obj['arc_use_jma_flag']),
		arc_p_centroid_top_n=int(obj['arc_p_centroid_top_n']),
		arc_origin_time_offset_sec=float(obj['arc_origin_time_offset_sec']),
	)
