from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	import pandas as pd

	from viz.plot_config import PlotConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JmaWithDasHypoinverseRunPaths:
	sta_file: Path
	station_csv: Path
	pcrh_file: Path
	scrh_file: Path
	hypoinverse_exe: Path
	cmd_template_file: Path
	epicenter_csv: Path
	measurement_csv: Path
	das_measurement_csv: Path
	das_epicenter_csv: Path
	prefecture_shp: Path
	plot_config_yaml: Path
	run_dir: Path


@dataclass(frozen=True)
class JmaWithDasHypoinverseTimeFilter:
	target_start: pd.Timestamp | str
	target_end: pd.Timestamp | str
	max_das_score: float | None


@dataclass(frozen=True)
class JmaWithDasHypoinverseInitialEvent:
	use_jma_flag: bool
	fix_depth: bool
	default_depth_km: float
	p_centroid_top_n: int
	origin_time_offset_sec: float


@dataclass(frozen=True)
class JmaWithDasHypoinverseDasFilter:
	dt_sec: float
	fiber_spacing_m: float
	channel_start: int
	win_half_samples: int
	residual_thresh_s: float
	decimation_base_spacing_m: float


@dataclass(frozen=True)
class JmaWithDasHypoinverseDasPhase:
	max_dt_sec: float


@dataclass(frozen=True)
class JmaWithDasHypoinverseSweep:
	das_total_weights: tuple[int | float, ...]
	use_das_channels: tuple[int, ...]


@dataclass(frozen=True)
class JmaWithDasHypoinversePlot:
	plot_setting: str


@dataclass(frozen=True)
class JmaWithDasHypoinverseRunConfig:
	paths: JmaWithDasHypoinverseRunPaths
	time_filter: JmaWithDasHypoinverseTimeFilter
	initial_event: JmaWithDasHypoinverseInitialEvent
	das_filter: JmaWithDasHypoinverseDasFilter
	das_phase: JmaWithDasHypoinverseDasPhase
	sweep: JmaWithDasHypoinverseSweep
	plot: JmaWithDasHypoinversePlot


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
	paths: JmaWithDasHypoinverseRunPaths,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	import pandas as pd

	_require_file(paths.epicenter_csv, label='epicenter_csv')
	_require_file(paths.measurement_csv, label='measurement_csv')
	_require_file(paths.das_epicenter_csv, label='das_epicenter_csv')
	_require_file(paths.das_measurement_csv, label='das_measurement_csv')

	df_epic = pd.read_csv(paths.epicenter_csv)
	df_meas = pd.read_csv(paths.measurement_csv)
	df_das_epic = pd.read_csv(paths.das_epicenter_csv)
	df_das_meas = pd.read_csv(paths.das_measurement_csv)
	logger.info(
		'loaded inputs: epicenters=%s measurements=%s das_epic=%s das_measurements=%s',
		len(df_epic),
		len(df_meas),
		len(df_das_epic),
		len(df_das_meas),
	)
	return df_epic, df_meas, df_das_epic, df_das_meas


def _filter_epicenter_and_measurements(
	df_epic: pd.DataFrame,
	df_meas: pd.DataFrame,
	*,
	target_start: pd.Timestamp,
	target_end: pd.Timestamp,
	max_das_score: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	import pandas as pd

	_require_columns(df_epic, ['event_id', 'origin_time'], label='epicenter_csv')
	_require_columns(df_meas, ['event_id'], label='measurement_csv')

	target_start_ts = pd.Timestamp(target_start)
	target_end_ts = pd.Timestamp(target_end)

	filtered_epic = df_epic.copy()
	filtered_meas = df_meas.copy()

	if 'das_score' not in filtered_epic.columns:
		filtered_epic['das_score'] = pd.NA

	filtered_epic['origin_dt'] = pd.to_datetime(filtered_epic['origin_time'])
	mask_time = (filtered_epic['origin_dt'] >= target_start_ts) & (
		filtered_epic['origin_dt'] < target_end_ts
	)

	if max_das_score is not None:
		filtered_epic['das_score'] = pd.to_numeric(
			filtered_epic['das_score'],
			errors='coerce',
		)
		mask_score = filtered_epic['das_score'] <= float(max_das_score)
	else:
		mask_score = pd.Series(True, index=filtered_epic.index)

	filtered_epic = filtered_epic[mask_time & mask_score].reset_index(drop=True)
	filtered_meas = filtered_meas[
		filtered_meas['event_id'].isin(filtered_epic['event_id'])
	].reset_index(drop=True)

	logger.info(
		'filtered jma inputs: epicenters=%s measurements=%s start=%s end=%s max_das_score=%s',
		len(filtered_epic),
		len(filtered_meas),
		target_start_ts,
		target_end_ts,
		max_das_score,
	)
	return filtered_epic, filtered_meas


def _filter_by_time_range(
	df: pd.DataFrame,
	*,
	time_col: str,
	parsed_col: str,
	target_start: pd.Timestamp,
	target_end: pd.Timestamp,
) -> pd.DataFrame:
	import pandas as pd

	_require_columns(df, [time_col], label=f'time range input {time_col}')
	target_start_ts = pd.Timestamp(target_start)
	target_end_ts = pd.Timestamp(target_end)

	out = df.copy()
	out[parsed_col] = pd.to_datetime(out[time_col])
	mask = (out[parsed_col] >= target_start_ts) & (out[parsed_col] < target_end_ts)
	return out.loc[mask].reset_index(drop=True)


def _patch_wet_line(lines: list[str], *, codeweight: int | float) -> list[str]:
	found = False
	patched: list[str] = []
	for line in lines:
		if line.strip().startswith('WET'):
			patched.append(f'WET 1.0 0.5 0.3 {codeweight}')
			found = True
			continue
		patched.append(line)

	if not found:
		raise ValueError('template cmd is missing a WET line to replace')

	return patched


def _build_runtime_cmd_with_wet(
	paths: JmaWithDasHypoinverseRunPaths,
	*,
	das_total_weight: int | float,
	use_das_channels: int,
) -> Path:
	from hypo.hypoinverse_cmd import patch_cmd_template_paths

	_require_file(paths.cmd_template_file, label='cmd_template_file')
	_require_file(paths.sta_file, label='sta_file')
	_require_file(paths.pcrh_file, label='pcrh_file')
	_require_file(paths.scrh_file, label='scrh_file')

	if use_das_channels <= 0:
		raise ValueError('use_das_channels must be >= 1')
	if float(das_total_weight) < 0:
		raise ValueError('das_total_weight must be >= 0')

	codeweight = float(das_total_weight) / float(use_das_channels)
	lines = paths.cmd_template_file.read_text(encoding='utf-8').splitlines()
	lines = patch_cmd_template_paths(
		lines,
		sta_file=str(paths.sta_file),
		pcrh_file=str(paths.pcrh_file),
		scrh_file=str(paths.scrh_file),
	)
	lines = _patch_wet_line(lines, codeweight=codeweight)

	cmd_run_file = paths.run_dir / 'hypoinverse_run.cmd'
	cmd_run_file.write_text(
		'\n'.join(lines) + '\n',
		encoding='utf-8',
		newline='\n',
	)
	logger.info(
		'wrote hypoinverse runtime cmd: %s codeweight=%s',
		cmd_run_file,
		codeweight,
	)
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
	df_epic: pd.DataFrame,
	df_meas: pd.DataFrame,
	*,
	prt_path: Path,
	out_join_csv: Path,
) -> pd.DataFrame:
	from hypo.join_jma_hypoinverse import build_joined_jma_hypo_csv

	_require_file(prt_path, label='prt_path')
	joined_df = build_joined_jma_hypo_csv(
		df_epic,
		df_meas,
		prt_path,
		out_join_csv,
	)
	logger.info('wrote joined catalog csv: %s rows=%s', out_join_csv, len(joined_df))
	return joined_df


def _render_outputs(
	df_join: pd.DataFrame,
	*,
	img_dir: Path,
	prefecture_shp: Path,
	plot_params: PlotConfig,
) -> None:
	from viz.events_map import plot_events_map_and_sections
	from viz.hypo.event_quality import plot_event_quality

	_require_file(prefecture_shp, label='prefecture_shp')
	img_dir.mkdir(parents=True, exist_ok=True)

	plot_event_quality(
		df_join,
		out_dir=img_dir,
		lat_col='lat_deg_jma',
		lon_col='lon_deg_jma',
		depth_col='depth_km_jma',
		hist_ranges={
			'RMS': (0.0, 1.5),
		},
	)

	extras = _build_plot_extras(plot_params.well_coord)
	out_location_png = img_dir / 'Hypoinv_event_location.png'
	out_jma_location_png = img_dir / 'jma_event_location.png'

	plot_events_map_and_sections(
		df=df_join,
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
		extras_xy=extras,
	)
	plot_events_map_and_sections(
		df=df_join,
		prefecture_shp=prefecture_shp,
		out_png=out_jma_location_png,
		mag_col='mag1_jma',
		origin_time_col='origin_time_jma',
		lat_col='lat_deg_jma',
		lon_col='lon_deg_jma',
		depth_col='depth_km_jma',
		markersize=10,
		lon_range=plot_params.lon_range,
		lat_range=plot_params.lat_range,
		depth_range=plot_params.depth_range,
		extras_xy=extras,
	)
	logger.info('rendered plots under: %s', img_dir)


def _build_run_dir(
	base_run_dir: Path,
	*,
	das_total_weight: int | float,
	use_das_channels: int,
) -> Path:
	return base_run_dir.parent / (
		base_run_dir.name
		+ f'_wet_{das_total_weight}'
		+ f'_ch_{use_das_channels}'
	)


def run_single_pipeline(
	config: JmaWithDasHypoinverseRunConfig,
	*,
	das_total_weight: int | float,
	use_das_channels: int,
	script_path: Path | None = None,
) -> None:
	from das.picks_filter import filter_and_decimate_das_picks
	from hypo.arc import write_hypoinverse_arc_from_phases
	from hypo.phase_jma import extract_phase_records
	from hypo.phase_ml_das import extract_das_phase_records

	if use_das_channels <= 0:
		raise ValueError('use_das_channels must be >= 1')
	if float(das_total_weight) < 0:
		raise ValueError('das_total_weight must be >= 0')

	run_dir = _build_run_dir(
		config.paths.run_dir,
		das_total_weight=das_total_weight,
		use_das_channels=use_das_channels,
	)
	run_dir.mkdir(parents=True, exist_ok=True)
	if script_path is not None:
		_write_script_snapshot(script_path, run_dir)

	run_paths = JmaWithDasHypoinverseRunPaths(
		sta_file=config.paths.sta_file,
		station_csv=config.paths.station_csv,
		pcrh_file=config.paths.pcrh_file,
		scrh_file=config.paths.scrh_file,
		hypoinverse_exe=config.paths.hypoinverse_exe,
		cmd_template_file=config.paths.cmd_template_file,
		epicenter_csv=config.paths.epicenter_csv,
		measurement_csv=config.paths.measurement_csv,
		das_measurement_csv=config.paths.das_measurement_csv,
		das_epicenter_csv=config.paths.das_epicenter_csv,
		prefecture_shp=config.paths.prefecture_shp,
		plot_config_yaml=config.paths.plot_config_yaml,
		run_dir=run_dir,
	)

	plot_params = _load_plot_params(
		run_paths.plot_config_yaml,
		config.plot.plot_setting,
	)
	df_epic, df_meas, df_das_epic, df_das_meas = _load_inputs(run_paths)
	df_epic, df_meas = _filter_epicenter_and_measurements(
		df_epic,
		df_meas,
		target_start=config.time_filter.target_start,
		target_end=config.time_filter.target_end,
		max_das_score=config.time_filter.max_das_score,
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

	spacing_m = config.das_filter.decimation_base_spacing_m / float(use_das_channels)
	df_das_meas_filtered = filter_and_decimate_das_picks(
		df_das_epic,
		df_das_meas,
		dt_sec=config.das_filter.dt_sec,
		fiber_spacing_m=config.das_filter.fiber_spacing_m,
		channel_start=config.das_filter.channel_start,
		win_half_samples=config.das_filter.win_half_samples,
		residual_thresh_s=config.das_filter.residual_thresh_s,
		spacing_m=spacing_m,
	)

	phases_hinet = extract_phase_records(df_meas)
	phases_das = extract_das_phase_records(
		df_epic,
		df_das_meas_filtered,
		max_dt_sec=config.das_phase.max_dt_sec,
	)
	phases_all = phases_hinet + phases_das

	arc_file = run_dir / 'hypoinverse_input.arc'
	write_hypoinverse_arc_from_phases(
		df_epic,
		phases_all,
		run_paths.station_csv,
		arc_file,
		default_depth_km=config.initial_event.default_depth_km,
		use_jma_flag=config.initial_event.use_jma_flag,
		p_centroid_top_n=config.initial_event.p_centroid_top_n,
		origin_time_offset_sec=config.initial_event.origin_time_offset_sec,
		fix_depth=config.initial_event.fix_depth,
	)

	cmd_run_file = _build_runtime_cmd_with_wet(
		run_paths,
		das_total_weight=das_total_weight,
		use_das_channels=use_das_channels,
	)
	_run_hypoinverse(run_paths.hypoinverse_exe, cmd_run_file, run_dir)

	prt_path = run_dir / 'hypoinverse_run.prt'
	out_join_csv = run_dir / 'hypoinverse_events_jma_join.csv'
	df_join = _build_joined_output(
		df_epic,
		df_meas,
		prt_path=prt_path,
		out_join_csv=out_join_csv,
	)
	_render_outputs(
		df_join,
		img_dir=run_dir / 'img',
		prefecture_shp=run_paths.prefecture_shp,
		plot_params=plot_params,
	)


def run_parameter_sweep(
	config: JmaWithDasHypoinverseRunConfig,
	*,
	script_path: Path | None = None,
) -> None:
	for das_total_weight in config.sweep.das_total_weights:
		for use_das_channels in config.sweep.use_das_channels:
			logger.info(
				'running sweep case: das_total_weight=%s use_das_channels=%s',
				das_total_weight,
				use_das_channels,
			)
			run_single_pipeline(
				config,
				das_total_weight=das_total_weight,
				use_das_channels=use_das_channels,
				script_path=script_path,
			)
