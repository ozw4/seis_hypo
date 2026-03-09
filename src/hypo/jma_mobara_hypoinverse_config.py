from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd

from common.yaml_config import read_yaml_mapping

T = TypeVar('T')


def _require_mapping(value: object, label: str) -> dict[str, Any]:
	if not isinstance(value, dict):
		raise ValueError(f'{label} must be a mapping')
	return dict(value)


def _validate_keys(
	cls: type[T],
	params: dict[str, Any],
	*,
	label: str,
) -> dict[str, Any]:
	field_names = [f.name for f in fields(cls)]
	missing = [name for name in field_names if name not in params]
	if missing:
		raise KeyError(f'{label} is missing required keys: {missing}')

	unknown = [key for key in params if key not in field_names]
	if unknown:
		raise ValueError(f'{label} has unknown keys: {unknown}')

	return params


def _build_dataclass(
	cls: type[T],
	params: object,
	*,
	label: str,
) -> T:
	mapping = _require_mapping(params, label)
	validated = _validate_keys(cls, mapping, label=label)
	return cls(**validated)


def _resolve_path(value: str | Path, *, base_dir: Path) -> Path:
	path = Path(value).expanduser()
	if not path.is_absolute():
		path = base_dir / path
	return path.resolve()


def _require_existing_file(path: Path, *, label: str) -> Path:
	if not path.is_file():
		raise FileNotFoundError(f'{label} not found: {path}')
	return path


def _require_bool(value: object, *, label: str) -> bool:
	if type(value) is not bool:
		raise TypeError(f'{label} must be bool')
	return value


def _require_non_negative_float(value: object, *, label: str) -> float:
	if isinstance(value, bool) or not isinstance(value, (int, float)):
		raise TypeError(f'{label} must be a number')
	out = float(value)
	if out < 0:
		raise ValueError(f'{label} must be >= 0')
	return out


def _require_positive_float(value: object, *, label: str) -> float:
	out = _require_non_negative_float(value, label=label)
	if out <= 0:
		raise ValueError(f'{label} must be > 0')
	return out


def _require_non_negative_int(value: object, *, label: str) -> int:
	if isinstance(value, bool) or not isinstance(value, int):
		raise TypeError(f'{label} must be an integer')
	if value < 0:
		raise ValueError(f'{label} must be >= 0')
	return int(value)


def _require_positive_int(value: object, *, label: str) -> int:
	out = _require_non_negative_int(value, label=label)
	if out <= 0:
		raise ValueError(f'{label} must be >= 1')
	return out


def _require_timestamp(value: object, *, label: str) -> pd.Timestamp:
	ts = pd.Timestamp(value)
	if pd.isna(ts):
		raise ValueError(f'{label} must be a valid timestamp')
	return ts


def _require_non_empty_str(value: object, *, label: str) -> str:
	if not isinstance(value, str):
		raise TypeError(f'{label} must be a string')
	out = value.strip()
	if out == '':
		raise ValueError(f'{label} must not be empty')
	return out


@dataclass(frozen=True)
class JmaMobaraHypoinversePaths:
	sta_file: Path
	pcrh_file: Path
	scrh_file: Path
	exe_file: Path
	cmd_file: Path
	measurement_csv: Path
	das_measurement_csv: Path
	das_epicenter_csv: Path
	pick_npz: Path
	station_with_das_csv: Path
	prefecture_shp: Path
	plot_config_yaml: Path
	run_dir: Path


@dataclass(frozen=True)
class JmaMobaraHypoinversePlot:
	plot_setting: str

	def __post_init__(self) -> None:
		object.__setattr__(
			self,
			'plot_setting',
			_require_non_empty_str(self.plot_setting, label='plot.plot_setting'),
		)


@dataclass(frozen=True)
class JmaMobaraHypoinverseTimeFilter:
	target_start: pd.Timestamp
	target_end: pd.Timestamp

	def __post_init__(self) -> None:
		target_start = _require_timestamp(
			self.target_start, label='time_filter.target_start'
		)
		target_end = _require_timestamp(self.target_end, label='time_filter.target_end')
		if not target_start < target_end:
			raise ValueError('time_filter requires target_start < target_end')

		object.__setattr__(self, 'target_start', target_start)
		object.__setattr__(self, 'target_end', target_end)


@dataclass(frozen=True)
class JmaMobaraHypoinverseInitialEvent:
	use_jma_flag: bool
	fix_depth: bool
	default_depth_km: float
	p_centroid_top_n: int
	origin_time_offset_sec: float

	def __post_init__(self) -> None:
		object.__setattr__(
			self,
			'use_jma_flag',
			_require_bool(self.use_jma_flag, label='initial_event.use_jma_flag'),
		)
		object.__setattr__(
			self,
			'fix_depth',
			_require_bool(self.fix_depth, label='initial_event.fix_depth'),
		)
		object.__setattr__(
			self,
			'default_depth_km',
			_require_non_negative_float(
				self.default_depth_km, label='initial_event.default_depth_km'
			),
		)
		object.__setattr__(
			self,
			'p_centroid_top_n',
			_require_positive_int(
				self.p_centroid_top_n, label='initial_event.p_centroid_top_n'
			),
		)
		object.__setattr__(
			self,
			'origin_time_offset_sec',
			_require_non_negative_float(
				self.origin_time_offset_sec,
				label='initial_event.origin_time_offset_sec',
			),
		)


@dataclass(frozen=True)
class JmaMobaraHypoinverseDasFilter:
	dt_sec: float
	fiber_spacing_m: float
	channel_start: int
	win_half_samples: int
	residual_thresh_s: float
	spacing_m: float

	def __post_init__(self) -> None:
		object.__setattr__(
			self,
			'dt_sec',
			_require_positive_float(self.dt_sec, label='das_filter.dt_sec'),
		)
		object.__setattr__(
			self,
			'fiber_spacing_m',
			_require_positive_float(
				self.fiber_spacing_m, label='das_filter.fiber_spacing_m'
			),
		)
		object.__setattr__(
			self,
			'channel_start',
			_require_non_negative_int(
				self.channel_start, label='das_filter.channel_start'
			),
		)
		object.__setattr__(
			self,
			'win_half_samples',
			_require_non_negative_int(
				self.win_half_samples, label='das_filter.win_half_samples'
			),
		)
		object.__setattr__(
			self,
			'residual_thresh_s',
			_require_non_negative_float(
				self.residual_thresh_s, label='das_filter.residual_thresh_s'
			),
		)
		object.__setattr__(
			self,
			'spacing_m',
			_require_positive_float(self.spacing_m, label='das_filter.spacing_m'),
		)


@dataclass(frozen=True)
class JmaMobaraHypoinverseDasPhase:
	max_dt_sec: float

	def __post_init__(self) -> None:
		object.__setattr__(
			self,
			'max_dt_sec',
			_require_positive_float(self.max_dt_sec, label='das_phase.max_dt_sec'),
		)


@dataclass(frozen=True)
class JmaMobaraHypoinversePlotQualityFilter:
	max_erh_km: float
	max_erz_km: float

	def __post_init__(self) -> None:
		object.__setattr__(
			self,
			'max_erh_km',
			_require_non_negative_float(
				self.max_erh_km,
				label='plot_quality_filter.max_erh_km',
			),
		)
		object.__setattr__(
			self,
			'max_erz_km',
			_require_non_negative_float(
				self.max_erz_km,
				label='plot_quality_filter.max_erz_km',
			),
		)


@dataclass(frozen=True)
class JmaMobaraHypoinverseConfig:
	paths: JmaMobaraHypoinversePaths
	plot: JmaMobaraHypoinversePlot
	time_filter: JmaMobaraHypoinverseTimeFilter
	initial_event: JmaMobaraHypoinverseInitialEvent
	das_filter: JmaMobaraHypoinverseDasFilter
	das_phase: JmaMobaraHypoinverseDasPhase
	plot_quality_filter: JmaMobaraHypoinversePlotQualityFilter


def _build_paths(params: object, *, base_dir: Path) -> JmaMobaraHypoinversePaths:
	mapping = _require_mapping(params, 'paths')
	validated = _validate_keys(JmaMobaraHypoinversePaths, mapping, label='paths')

	resolved: dict[str, Path] = {}
	for key, value in validated.items():
		if not isinstance(value, (str, Path)):
			raise TypeError(f'paths.{key} must be a path-like string')
		resolved[key] = _resolve_path(value, base_dir=base_dir)

	for key in validated:
		if key == 'run_dir':
			continue
		_require_existing_file(resolved[key], label=f'paths.{key}')

	run_dir = resolved['run_dir']
	if run_dir.exists() and not run_dir.is_dir():
		raise NotADirectoryError(f'paths.run_dir is not a directory: {run_dir}')

	return JmaMobaraHypoinversePaths(**resolved)


def load_jma_mobara_hypoinverse_config(
	config_path: str | Path,
) -> JmaMobaraHypoinverseConfig:
	config_path = Path(config_path).expanduser().resolve()
	if not config_path.is_file():
		raise FileNotFoundError(f'config not found: {config_path}')

	obj = read_yaml_mapping(config_path)
	required_sections = {
		'paths',
		'plot',
		'time_filter',
		'initial_event',
		'das_filter',
		'das_phase',
		'plot_quality_filter',
	}
	missing_sections = sorted(required_sections.difference(obj))
	if missing_sections:
		raise KeyError(f'config is missing required sections: {missing_sections}')

	unknown_sections = sorted(set(obj).difference(required_sections))
	if unknown_sections:
		raise ValueError(f'config has unknown sections: {unknown_sections}')

	base_dir = config_path.parent
	return JmaMobaraHypoinverseConfig(
		paths=_build_paths(obj['paths'], base_dir=base_dir),
		plot=_build_dataclass(
			JmaMobaraHypoinversePlot,
			obj['plot'],
			label='plot',
		),
		time_filter=_build_dataclass(
			JmaMobaraHypoinverseTimeFilter,
			obj['time_filter'],
			label='time_filter',
		),
		initial_event=_build_dataclass(
			JmaMobaraHypoinverseInitialEvent,
			obj['initial_event'],
			label='initial_event',
		),
		das_filter=_build_dataclass(
			JmaMobaraHypoinverseDasFilter,
			obj['das_filter'],
			label='das_filter',
		),
		das_phase=_build_dataclass(
			JmaMobaraHypoinverseDasPhase,
			obj['das_phase'],
			label='das_phase',
		),
		plot_quality_filter=_build_dataclass(
			JmaMobaraHypoinversePlotQualityFilter,
			obj['plot_quality_filter'],
			label='plot_quality_filter',
		),
	)
