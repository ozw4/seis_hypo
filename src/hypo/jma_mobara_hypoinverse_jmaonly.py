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
class JmaOnlyHypoinversePaths:
	sta_file: Path
	station_csv: Path
	pcrh_file: Path
	scrh_file: Path
	exe_file: Path
	cmd_template_file: Path
	epicenter_csv: Path
	measurement_csv: Path
	prefecture_shp: Path
	plot_config_yaml: Path
	run_dir: Path


@dataclass(frozen=True)
class JmaOnlyHypoinverseTimeFilter:
	target_start: pd.Timestamp | str
	target_end: pd.Timestamp | str
	max_das_score: float | None


@dataclass(frozen=True)
class JmaOnlyHypoinverseInitialEvent:
	use_jma_flag: bool
	fix_depth: bool
	default_depth_km: float
	p_centroid_top_n: int
	origin_time_offset_sec: float


@dataclass(frozen=True)
class JmaOnlyHypoinverseRunConfig:
	paths: JmaOnlyHypoinversePaths
	time_filter: JmaOnlyHypoinverseTimeFilter
	initial_event: JmaOnlyHypoinverseInitialEvent
	plot_setting: str


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
	epicenter_csv: Path,
	measurement_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	import pandas as pd

	_require_file(epicenter_csv, label='epicenter_csv')
	_require_file(measurement_csv, label='measurement_csv')
	df_epic = pd.read_csv(epicenter_csv)
	df_meas = pd.read_csv(measurement_csv)
	logger.info(
		'loaded inputs: epicenters=%s measurements=%s',
		len(df_epic),
		len(df_meas),
	)
	return df_epic, df_meas


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
	mask_time_epic = (filtered_epic['origin_dt'] >= target_start_ts) & (
		filtered_epic['origin_dt'] < target_end_ts
	)

	if max_das_score is not None:
		filtered_epic['das_score'] = pd.to_numeric(
			filtered_epic['das_score'],
			errors='coerce',
		)
		mask_score_epic = filtered_epic['das_score'] <= float(max_das_score)
	else:
		mask_score_epic = pd.Series(True, index=filtered_epic.index)

	filtered_epic = filtered_epic[mask_time_epic & mask_score_epic].reset_index(
		drop=True
	)
	filtered_meas = filtered_meas[
		filtered_meas['event_id'].isin(filtered_epic['event_id'])
	].reset_index(drop=True)

	logger.info(
		'filtered inputs: epicenters=%s measurements=%s start=%s end=%s max_das_score=%s',
		len(filtered_epic),
		len(filtered_meas),
		target_start_ts,
		target_end_ts,
		max_das_score,
	)
	return filtered_epic, filtered_meas


def _write_arc(
	df_epic: pd.DataFrame,
	df_meas: pd.DataFrame,
	*,
	station_csv: Path,
	run_dir: Path,
	initial_event: JmaOnlyHypoinverseInitialEvent,
) -> Path:
	from hypo.arc import write_hypoinverse_arc

	_require_file(station_csv, label='station_csv')
	arc_file = run_dir / 'hypoinverse_input.arc'
	write_hypoinverse_arc(
		df_epic,
		df_meas,
		station_csv,
		arc_file,
		default_depth_km=initial_event.default_depth_km,
		use_jma_flag=initial_event.use_jma_flag,
		p_centroid_top_n=initial_event.p_centroid_top_n,
		origin_time_offset_sec=initial_event.origin_time_offset_sec,
		fix_depth=initial_event.fix_depth,
	)
	logger.info('wrote hypoinverse arc: %s', arc_file)
	return arc_file


def _write_runtime_cmd(paths: JmaOnlyHypoinversePaths) -> Path:
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
	exe_file: Path,
	cmd_run_file: Path,
	run_dir: Path,
) -> subprocess.CompletedProcess[str]:
	_require_file(exe_file, label='exe_file')
	_require_file(cmd_run_file, label='cmd_run_file')
	with cmd_run_file.open('rb') as stdin:
		result = subprocess.run(
			[str(exe_file)],
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


def run_pipeline(
	config: JmaOnlyHypoinverseRunConfig,
	*,
	script_path: Path | None = None,
) -> None:
	run_dir = config.paths.run_dir
	run_dir.mkdir(parents=True, exist_ok=True)
	if script_path is not None:
		_write_script_snapshot(script_path, run_dir)

	plot_params = _load_plot_params(
		config.paths.plot_config_yaml,
		config.plot_setting,
	)
	df_epic, df_meas = _load_inputs(
		config.paths.epicenter_csv,
		config.paths.measurement_csv,
	)
	df_epic, df_meas = _filter_epicenter_and_measurements(
		df_epic,
		df_meas,
		target_start=config.time_filter.target_start,
		target_end=config.time_filter.target_end,
		max_das_score=config.time_filter.max_das_score,
	)

	_write_arc(
		df_epic,
		df_meas,
		station_csv=config.paths.station_csv,
		run_dir=run_dir,
		initial_event=config.initial_event,
	)
	cmd_run_file = _write_runtime_cmd(config.paths)
	_run_hypoinverse(config.paths.exe_file, cmd_run_file, run_dir)

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
		prefecture_shp=config.paths.prefecture_shp,
		plot_params=plot_params,
	)
