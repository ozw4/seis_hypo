from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
	from viz.plot_config import PlotConfig

logger = logging.getLogger(__name__)


def _require_non_negative_float(value: float, *, label: str) -> float:
	if isinstance(value, bool) or not isinstance(value, (int, float)):
		raise TypeError(f'{label} must be a number')
	out = float(value)
	if out < 0:
		raise ValueError(f'{label} must be >= 0')
	return out


def _require_positive_float(value: float, *, label: str) -> float:
	out = _require_non_negative_float(value, label=label)
	if out <= 0:
		raise ValueError(f'{label} must be > 0')
	return out


def _require_non_negative_int(value: int, *, label: str) -> int:
	if isinstance(value, bool) or not isinstance(value, int):
		raise TypeError(f'{label} must be an integer')
	if value < 0:
		raise ValueError(f'{label} must be >= 0')
	return value


def _require_positive_int(value: int, *, label: str) -> int:
	out = _require_non_negative_int(value, label=label)
	if out <= 0:
		raise ValueError(f'{label} must be >= 1')
	return out


def _require_bool(value: bool, *, label: str) -> bool:
	if type(value) is not bool:
		raise TypeError(f'{label} must be bool')
	return value


def _require_timestamp(value: pd.Timestamp | str, *, label: str) -> pd.Timestamp:
	ts = pd.Timestamp(value)
	if pd.isna(ts):
		raise ValueError(f'{label} must be a valid timestamp')
	return ts


def _require_non_empty_str(value: str, *, label: str) -> str:
	if not isinstance(value, str):
		raise TypeError(f'{label} must be a string')
	out = value.strip()
	if out == '':
		raise ValueError(f'{label} must not be empty')
	return out


@dataclass(frozen=True)
class JmaFpEventHypoinverseRunPaths:
	sta_file: Path
	pcrh_file: Path
	scrh_file: Path
	hypoinverse_exe: Path
	cmd_template_file: Path
	measurement_csv: Path
	das_measurement_csv: Path
	das_epicenter_csv: Path
	pick_npz: Path
	station_with_das_csv: Path
	prefecture_shp: Path
	plot_config_yaml: Path
	run_dir: Path


@dataclass(frozen=True)
class JmaFpEventHypoinverseTimeFilter:
	target_start: pd.Timestamp | str
	target_end: pd.Timestamp | str

	def __post_init__(self) -> None:
		target_start = _require_timestamp(
			self.target_start,
			label='time_filter.target_start',
		)
		target_end = _require_timestamp(
			self.target_end,
			label='time_filter.target_end',
		)
		if not target_start < target_end:
			raise ValueError('time_filter requires target_start < target_end')

		object.__setattr__(self, 'target_start', target_start)
		object.__setattr__(self, 'target_end', target_end)


@dataclass(frozen=True)
class JmaFpEventHypoinverseInitialEvent:
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
				self.default_depth_km,
				label='initial_event.default_depth_km',
			),
		)
		object.__setattr__(
			self,
			'p_centroid_top_n',
			_require_positive_int(
				self.p_centroid_top_n,
				label='initial_event.p_centroid_top_n',
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
class JmaFpEventHypoinversePlot:
	plot_setting: str
	max_erh_km: float
	max_erz_km: float
	max_origin_time_err_sec: float | None

	def __post_init__(self) -> None:
		object.__setattr__(
			self,
			'plot_setting',
			_require_non_empty_str(self.plot_setting, label='plot.plot_setting'),
		)
		object.__setattr__(
			self,
			'max_erh_km',
			_require_non_negative_float(
				self.max_erh_km,
				label='plot.max_erh_km',
			),
		)
		object.__setattr__(
			self,
			'max_erz_km',
			_require_non_negative_float(
				self.max_erz_km,
				label='plot.max_erz_km',
			),
		)
		if self.max_origin_time_err_sec is not None:
			object.__setattr__(
				self,
				'max_origin_time_err_sec',
				_require_non_negative_float(
					self.max_origin_time_err_sec,
					label='plot.max_origin_time_err_sec',
				),
			)


@dataclass(frozen=True)
class JmaFpEventHypoinverseDasFilter:
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
				self.fiber_spacing_m,
				label='das_filter.fiber_spacing_m',
			),
		)
		object.__setattr__(
			self,
			'channel_start',
			_require_non_negative_int(
				self.channel_start,
				label='das_filter.channel_start',
			),
		)
		object.__setattr__(
			self,
			'win_half_samples',
			_require_non_negative_int(
				self.win_half_samples,
				label='das_filter.win_half_samples',
			),
		)
		object.__setattr__(
			self,
			'residual_thresh_s',
			_require_non_negative_float(
				self.residual_thresh_s,
				label='das_filter.residual_thresh_s',
			),
		)
		object.__setattr__(
			self,
			'spacing_m',
			_require_positive_float(self.spacing_m, label='das_filter.spacing_m'),
		)


@dataclass(frozen=True)
class JmaFpEventHypoinverseDasPhase:
	max_dt_sec: float

	def __post_init__(self) -> None:
		object.__setattr__(
			self,
			'max_dt_sec',
			_require_positive_float(self.max_dt_sec, label='das_phase.max_dt_sec'),
		)


@dataclass(frozen=True)
class JmaFpEventHypoinverseRunConfig:
	paths: JmaFpEventHypoinverseRunPaths
	time_filter: JmaFpEventHypoinverseTimeFilter
	initial_event: JmaFpEventHypoinverseInitialEvent
	das_filter: JmaFpEventHypoinverseDasFilter
	das_phase: JmaFpEventHypoinverseDasPhase
	plot: JmaFpEventHypoinversePlot


def _require_file(path: Path, *, label: str) -> None:
	if not path.is_file():
		raise FileNotFoundError(f'{label} not found: {path}')


def _require_columns(
	df: pd.DataFrame,
	required: list[str],
	*,
	label: str,
) -> None:
	missing = [col for col in required if col not in df.columns]
	if missing:
		raise ValueError(f'{label} is missing required columns: {missing}')


def _write_script_snapshot(script_path: Path, run_dir: Path) -> Path:
	_require_file(script_path, label='script_path')
	snapshot_path = run_dir / ('bak_' + script_path.name)
	snapshot_path.write_text(
		script_path.read_text(encoding='utf-8'),
		encoding='utf-8',
	)
	logger.info('saved script snapshot: %s', snapshot_path)
	return snapshot_path


def _write_config_snapshot(config_path: Path, run_dir: Path) -> Path:
	_require_file(config_path, label='config_path')
	out = run_dir / 'config_used.yaml'
	copy2(config_path, out)
	logger.info('saved config snapshot: %s', out)
	return out


def _load_plot_params(plot_config_yaml: Path, plot_setting: str) -> PlotConfig:
	from common.load_config import load_config
	from viz.plot_config import PlotConfig

	_require_file(plot_config_yaml, label='plot_config_yaml')
	params = load_config(PlotConfig, plot_config_yaml, plot_setting)
	logger.info(
		'loaded plot config: yaml=%s setting=%s',
		plot_config_yaml,
		plot_setting,
	)
	return params


def _load_station_df_from_pick_npz(pick_npz: Path) -> pd.DataFrame:
	import numpy.core as numpy_core_compat

	_require_file(pick_npz, label='pick_npz')
	sys.modules.setdefault('numpy._core', numpy_core_compat)

	with np.load(pick_npz, allow_pickle=True) as data:
		required_keys = ['sta_code', 'sta_lat', 'sta_lon']
		missing = [key for key in required_keys if key not in data.files]
		if missing:
			raise KeyError(f'pick_npz is missing required arrays: {missing}')

		sta_code = data['sta_code']
		sta_lat = data['sta_lat']
		sta_lon = data['sta_lon']

	if sta_code.shape[0] != sta_lat.shape[0] or sta_code.shape[0] != sta_lon.shape[0]:
		raise ValueError('pick_npz station arrays must have the same length')

	if np.issubdtype(sta_code.dtype, np.bytes_):
		sta_code = sta_code.astype(str)

	station_df = pd.DataFrame(
		{
			'station': sta_code,
			'lat': sta_lat,
			'lon': sta_lon,
		}
	)
	_require_columns(station_df, ['station', 'lat', 'lon'], label='station_df')
	return station_df


def _filter_by_time_range(
	df: pd.DataFrame,
	*,
	time_col: str,
	parsed_col: str,
	target_start: pd.Timestamp,
	target_end: pd.Timestamp,
) -> pd.DataFrame:
	_require_columns(df, [time_col], label=f'time range input {time_col}')
	target_start_ts = pd.Timestamp(target_start)
	target_end_ts = pd.Timestamp(target_end)
	if not target_start_ts < target_end_ts:
		raise ValueError('time range requires target_start < target_end')

	out = df.copy()
	out[parsed_col] = pd.to_datetime(out[time_col])
	mask = (out[parsed_col] >= target_start_ts) & (out[parsed_col] < target_end_ts)
	return out.loc[mask].reset_index(drop=True)


def _filter_plot_df_by_quality(
	df: pd.DataFrame,
	*,
	max_erh_km: float,
	max_erz_km: float,
	max_origin_time_err_sec: float | None,
) -> pd.DataFrame:
	required_cols = ['ERH', 'ERZ']
	if max_origin_time_err_sec is not None:
		required_cols.append('origin_time_err_sec')
	_require_columns(df, required_cols, label='plot_quality_filter')

	mask = (
		df['ERH'].notna()
		& df['ERZ'].notna()
		& (df['ERH'] <= max_erh_km)
		& (df['ERZ'] <= max_erz_km)
	)
	if max_origin_time_err_sec is not None:
		mask = (
			mask
			& df['origin_time_err_sec'].notna()
			& (df['origin_time_err_sec'] <= max_origin_time_err_sec)
		)

	filtered_df = df.loc[mask].reset_index(drop=True)
	if filtered_df.empty:
		raise RuntimeError(
			'no events remain after plot_quality_filter: '
			f'count_before={len(df)}, '
			f'max_erh_km={max_erh_km}, '
			f'max_erz_km={max_erz_km}, '
			f'max_origin_time_err_sec={max_origin_time_err_sec}'
		)

	return filtered_df


def _write_plot_filter_event_csvs(
	run_dir: Path,
	*,
	initial_event_df: pd.DataFrame,
	prt_df: pd.DataFrame,
	prt_plot_df: pd.DataFrame,
) -> None:
	from hypo.hypoinverse_event_export import build_hypoinverse_event_export_df

	before_csv = run_dir / 'hypoinverse_events_before_plot_quality_filter.csv'
	after_csv = run_dir / 'hypoinverse_events_after_plot_quality_filter.csv'

	before_df = build_hypoinverse_event_export_df(initial_event_df, prt_df)
	before_df['passed_plot_quality_filter'] = before_df['seq'].isin(prt_plot_df['seq'])

	after_df = build_hypoinverse_event_export_df(initial_event_df, prt_plot_df)
	after_df['passed_plot_quality_filter'] = True

	before_df.to_csv(before_csv, index=False)
	after_df.to_csv(after_csv, index=False)
	logger.info('wrote plot-filter csvs: before=%s after=%s', before_csv, after_csv)


def _build_plot_extras(
	well_coord: tuple[float, float] | None,
) -> list[dict[str, object]] | None:
	if well_coord is None:
		return None
	return [
		{
			'label': 'mobara site',
			'xy': [(well_coord[1], well_coord[0])],
			'marker': '*',
			'color': 'royalblue',
			'size': 30,
			'annotate': False,
		}
	]


def _load_inputs(
	config: JmaFpEventHypoinverseRunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	from hypo.initial_event_builder import build_initial_events_from_ml_picks

	paths = config.paths
	station_df = _load_station_df_from_pick_npz(paths.pick_npz)

	_require_file(paths.measurement_csv, label='measurement_csv')
	eqt_df = pd.read_csv(paths.measurement_csv)
	_require_columns(
		eqt_df,
		['event_id', 'station_code', 'Phase', 'pick_time', 'event_time_peak', 'w_conf'],
		label='measurement_csv',
	)
	initial_event_df = build_initial_events_from_ml_picks(eqt_df, station_df)

	_require_file(paths.das_epicenter_csv, label='das_epicenter_csv')
	df_das_epic = pd.read_csv(paths.das_epicenter_csv)
	_require_columns(df_das_epic, ['event_time'], label='das_epicenter_csv')

	_require_file(paths.das_measurement_csv, label='das_measurement_csv')
	df_das_meas = pd.read_csv(paths.das_measurement_csv)
	_require_columns(df_das_meas, ['event_time_peak'], label='das_measurement_csv')

	return eqt_df, initial_event_df, df_das_epic, df_das_meas


def _write_arc(
	initial_event_df: pd.DataFrame,
	phases_all: list[dict[str, object]],
	*,
	station_with_das_csv: Path,
	arc_file: Path,
	initial_event: JmaFpEventHypoinverseInitialEvent,
) -> None:
	from hypo.arc import write_hypoinverse_arc_from_phases

	_require_file(station_with_das_csv, label='station_with_das_csv')
	write_hypoinverse_arc_from_phases(
		initial_event_df,
		phases_all,
		station_with_das_csv,
		arc_file,
		default_depth_km=initial_event.default_depth_km,
		use_jma_flag=initial_event.use_jma_flag,
		p_centroid_top_n=initial_event.p_centroid_top_n,
		origin_time_offset_sec=initial_event.origin_time_offset_sec,
		fix_depth=initial_event.fix_depth,
	)
	logger.info('wrote hypoinverse arc: %s', arc_file)


def _write_runtime_cmd(paths: JmaFpEventHypoinverseRunPaths) -> Path:
	from hypo.hypoinverse_cmd import write_cmd_template_paths

	_require_file(paths.cmd_template_file, label='cmd_template_file')
	_require_file(paths.sta_file, label='sta_file')
	_require_file(paths.pcrh_file, label='pcrh_file')
	_require_file(paths.scrh_file, label='scrh_file')

	cmd_run_file = paths.run_dir / 'hypoinverse_run.cmd'
	write_cmd_template_paths(
		paths.cmd_template_file,
		cmd_run_file,
		sta_file=str(paths.sta_file),
		pcrh_file=str(paths.pcrh_file),
		scrh_file=str(paths.scrh_file),
	)
	logger.info('wrote hypoinverse runtime cmd: %s', cmd_run_file)
	return cmd_run_file


def _run_hypoinverse(
	hypoinverse_exe: Path,
	cmd_run_file: Path,
	run_dir: Path,
) -> subprocess.CompletedProcess[str]:
	_require_file(hypoinverse_exe, label='hypoinverse_exe')
	_require_file(cmd_run_file, label='cmd_run_file')
	with cmd_run_file.open('rb') as stdin:
		result = subprocess.run(
			[str(hypoinverse_exe)],
			stdin=stdin,
			cwd=run_dir,
			capture_output=True,
			text=True,
			check=True,
		)

	logger.info('hypoinverse returncode=%s', result.returncode)
	if result.stdout:
		logger.info('hypoinverse stdout:\n%s', result.stdout.rstrip())
	if result.stderr:
		logger.info('hypoinverse stderr:\n%s', result.stderr.rstrip())
	return result


def _build_joined_output(
	initial_event_df: pd.DataFrame,
	prt_df: pd.DataFrame,
	*,
	out_join_csv: Path,
) -> pd.DataFrame:
	from hypo.hypoinverse_event_export import build_hypoinverse_event_export_df

	joined_df = build_hypoinverse_event_export_df(initial_event_df, prt_df)
	out_join_csv.parent.mkdir(parents=True, exist_ok=True)
	joined_df.to_csv(out_join_csv, index=False)
	logger.info('wrote joined catalog csv: %s rows=%s', out_join_csv, len(joined_df))
	return joined_df


def _render_outputs(
	joined_df: pd.DataFrame,
	*,
	prt_plot_df: pd.DataFrame,
	img_dir: Path,
	prefecture_shp: Path,
	plot_params: PlotConfig,
) -> None:
	from viz.events_map import plot_events_map_and_sections
	from viz.hypo.event_quality import plot_event_quality

	_require_file(prefecture_shp, label='prefecture_shp')
	img_dir.mkdir(parents=True, exist_ok=True)

	plot_df = joined_df[joined_df['seq'].isin(prt_plot_df['seq'])].reset_index(drop=True)
	if plot_df.empty:
		raise RuntimeError('no joined events remain for rendering after plot filter')

	plot_event_quality(
		plot_df,
		out_dir=img_dir,
		lat_col='lat_deg_init',
		lon_col='lon_deg_init',
		depth_col='depth_km_init',
		lat_col_jma='lat_deg_init',
		lon_col_jma='lon_deg_init',
		depth_col_jma='depth_km_init',
		mag_col_jma=None,
		hist_ranges={
			'RMS': (0.0, 1.5),
		},
	)

	out_location_png = img_dir / 'Hypoinv_event_location.png'
	plot_events_map_and_sections(
		df=plot_df,
		prefecture_shp=prefecture_shp,
		out_png=out_location_png,
		mag_col=None,
		origin_time_col='origin_time_hyp',
		lat_col='lat_deg_hyp',
		lon_col='lon_deg_hyp',
		depth_col='depth_km_hyp',
		markersize=30,
		lon_range=plot_params.lon_range,
		lat_range=plot_params.lat_range,
		depth_range=plot_params.depth_range,
		extras_xy=_build_plot_extras(plot_params.well_coord),
	)
	logger.info('rendered plots under: %s', img_dir)


def run_pipeline(
	config: JmaFpEventHypoinverseRunConfig,
	*,
	script_path: Path | None = None,
	config_path: Path | None = None,
) -> None:
	from das.picks_filter import filter_and_decimate_das_picks
	from hypo.hypoinverse_prt import load_hypoinverse_summary_from_prt
	from hypo.phase_ml import extract_ml_pick_phase_records
	from hypo.phase_ml_das import extract_das_phase_records

	run_dir = config.paths.run_dir
	run_dir.mkdir(parents=True, exist_ok=True)
	if script_path is not None:
		_write_script_snapshot(script_path, run_dir)
	if config_path is not None:
		_write_config_snapshot(config_path, run_dir)

	plot_params = _load_plot_params(
		config.paths.plot_config_yaml,
		config.plot.plot_setting,
	)
	eqt_df, initial_event_df, df_das_epic, df_das_meas = _load_inputs(config)

	initial_event_df = _filter_by_time_range(
		initial_event_df,
		time_col='origin_time',
		parsed_col='origin_dt',
		target_start=config.time_filter.target_start,
		target_end=config.time_filter.target_end,
	)
	df_das_epic = _filter_by_time_range(
		df_das_epic,
		time_col='event_time',
		parsed_col='event_time',
		target_start=config.time_filter.target_start,
		target_end=config.time_filter.target_end,
	)
	df_das_meas = _filter_by_time_range(
		df_das_meas,
		time_col='event_time_peak',
		parsed_col='event_time_peak',
		target_start=config.time_filter.target_start,
		target_end=config.time_filter.target_end,
	)

	df_das_meas_filtered = filter_and_decimate_das_picks(
		df_das_epic,
		df_das_meas,
		dt_sec=config.das_filter.dt_sec,
		fiber_spacing_m=config.das_filter.fiber_spacing_m,
		channel_start=config.das_filter.channel_start,
		win_half_samples=config.das_filter.win_half_samples,
		residual_thresh_s=config.das_filter.residual_thresh_s,
		spacing_m=config.das_filter.spacing_m,
	)

	phases_hinet = extract_ml_pick_phase_records(eqt_df)
	phases_das = extract_das_phase_records(
		initial_event_df,
		df_das_meas_filtered,
		max_dt_sec=config.das_phase.max_dt_sec,
	)
	phases_all = phases_hinet + phases_das

	arc_file = run_dir / 'hypoinverse_input.arc'
	_write_arc(
		initial_event_df,
		phases_all,
		station_with_das_csv=config.paths.station_with_das_csv,
		arc_file=arc_file,
		initial_event=config.initial_event,
	)

	cmd_run_file = _write_runtime_cmd(config.paths)
	_run_hypoinverse(config.paths.hypoinverse_exe, cmd_run_file, run_dir)

	prt_path = run_dir / 'hypoinverse_run.prt'
	prt_df = load_hypoinverse_summary_from_prt(prt_path)
	prt_plot_df = _filter_plot_df_by_quality(
		prt_df,
		max_erh_km=config.plot.max_erh_km,
		max_erz_km=config.plot.max_erz_km,
		max_origin_time_err_sec=config.plot.max_origin_time_err_sec,
	)
	_write_plot_filter_event_csvs(
		run_dir,
		initial_event_df=initial_event_df,
		prt_df=prt_df,
		prt_plot_df=prt_plot_df,
	)

	out_join_csv = run_dir / 'hypoinverse_events_jma_join.csv'
	joined_df = _build_joined_output(
		initial_event_df,
		prt_df,
		out_join_csv=out_join_csv,
	)
	_render_outputs(
		joined_df,
		prt_plot_df=prt_plot_df,
		img_dir=run_dir / 'img',
		prefecture_shp=config.paths.prefecture_shp,
		plot_params=plot_params,
	)
