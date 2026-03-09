from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from shutil import copy2

import numpy as np
import pandas as pd


def _find_src_dir(start: Path) -> Path:
	path = start.resolve()
	for directory in [path] + list(path.parents):
		src = directory / 'src'
		if (src / 'hypo' / 'jma_mobara_hypoinverse_config.py').is_file():
			return src
	raise FileNotFoundError(
		'could not locate repo src/ '
		'(expected src/hypo/jma_mobara_hypoinverse_config.py)'
	)


THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = _find_src_dir(THIS_DIR)
SRC_DIR_STR = str(SRC_DIR)
if SRC_DIR_STR not in sys.path:
	sys.path.insert(0, SRC_DIR_STR)


import sys

import numpy.core as numpy_core_compat

from common.load_config import load_config
from das.picks_filter import filter_and_decimate_das_picks
from hypo.arc import write_hypoinverse_arc_from_phases
from hypo.hypoinverse_cmd import write_cmd_template_paths
from hypo.hypoinverse_prt import load_hypoinverse_summary_from_prt
from hypo.initial_event_builder import build_initial_events_from_ml_picks
from hypo.jma_mobara_hypoinverse_config import (
	JmaMobaraHypoinverseConfig,
	load_jma_mobara_hypoinverse_config,
)
from hypo.phase_ml import extract_ml_pick_phase_records
from hypo.phase_ml_das import extract_das_phase_records
from viz.events_map import plot_events_map_and_sections
from viz.plot_config import PlotConfig

sys.modules.setdefault('numpy._core', numpy_core_compat)


def parse_args(argv: list[str]) -> argparse.Namespace:
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
		required=True,
		type=Path,
		help='pipeline config yaml path',
	)
	return parser.parse_args(argv)


def _write_script_snapshot(run_dir: Path) -> Path:
	script_path = Path(__file__).resolve()
	snapshot_path = run_dir / ('bak_' + script_path.name)
	snapshot_path.write_text(
		script_path.read_text(encoding='utf-8'),
		encoding='utf-8',
	)
	return snapshot_path


def _write_config_snapshot(config_path: Path, run_dir: Path) -> Path:
	out = run_dir / 'config_used.yaml'
	copy2(config_path, out)
	return out


def _load_station_df_from_pick_npz(pick_npz: Path) -> pd.DataFrame:
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

	return pd.DataFrame(
		{
			'station': sta_code,
			'lat': sta_lat,
			'lon': sta_lon,
		}
	)


def _filter_by_time_range(
	df: pd.DataFrame,
	*,
	time_col: str,
	parsed_col: str,
	target_start: pd.Timestamp,
	target_end: pd.Timestamp,
) -> pd.DataFrame:
	out = df.copy()
	out[parsed_col] = pd.to_datetime(out[time_col])
	mask = (out[parsed_col] >= target_start) & (out[parsed_col] < target_end)
	return out.loc[mask].reset_index(drop=True)


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


def _filter_plot_df_by_quality(
	df: pd.DataFrame,
	*,
	max_erh_km: float,
	max_erz_km: float,
) -> pd.DataFrame:
	required_cols = ['ERH', 'ERZ']
	missing = [col for col in required_cols if col not in df.columns]
	if missing:
		raise KeyError(f'plot quality filter requires columns: {missing}')

	count_before = len(df)
	print(
		'plot_quality_filter:',
		f'count_before={count_before}',
		f'max_erh_km={max_erh_km}',
		f'max_erz_km={max_erz_km}',
	)

	mask = (
		df['ERH'].notna()
		& df['ERZ'].notna()
		& (df['ERH'] <= max_erh_km)
		& (df['ERZ'] <= max_erz_km)
	)
	filtered_df = df.loc[mask].reset_index(drop=True)
	count_after = len(filtered_df)
	print('plot_quality_filter:', f'count_after={count_after}')

	if count_after == 0:
		raise RuntimeError(
			'no events remain after plot_quality_filter: '
			f'count_before={count_before}, '
			f'count_after={count_after}, '
			f'max_erh_km={max_erh_km}, '
			f'max_erz_km={max_erz_km}'
		)

	return filtered_df


def run_pipeline(
	config: JmaMobaraHypoinverseConfig,
	*,
	config_path: Path,
) -> None:
	run_dir = config.paths.run_dir
	run_dir.mkdir(parents=True, exist_ok=True)
	_write_script_snapshot(run_dir)
	_write_config_snapshot(config_path, run_dir)

	arc_file = run_dir / 'hypoinverse_input.arc'
	prt_path = run_dir / 'hypoinverse_run.prt'
	cmd_run_file = run_dir / 'hypoinverse_run.cmd'

	img_dir = run_dir / 'img'
	img_dir.mkdir(parents=True, exist_ok=True)
	out_location_png = img_dir / 'Hypoinv_event_location.png'

	plot_params = load_config(
		PlotConfig,
		config.paths.plot_config_yaml,
		config.plot.plot_setting,
	)

	station_df = _load_station_df_from_pick_npz(config.paths.pick_npz)
	eqt_df = pd.read_csv(config.paths.measurement_csv)
	df_epic = build_initial_events_from_ml_picks(eqt_df, station_df)

	df_das_epic = pd.read_csv(config.paths.das_epicenter_csv)
	df_das_meas = pd.read_csv(config.paths.das_measurement_csv)

	df_epic = _filter_by_time_range(
		df_epic,
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
		df_epic,
		df_das_meas_filtered,
		max_dt_sec=config.das_phase.max_dt_sec,
	)
	phases_all = phases_hinet + phases_das

	write_hypoinverse_arc_from_phases(
		df_epic,
		phases_all,
		config.paths.station_with_das_csv,
		arc_file,
		default_depth_km=config.initial_event.default_depth_km,
		use_jma_flag=config.initial_event.use_jma_flag,
		p_centroid_top_n=config.initial_event.p_centroid_top_n,
		origin_time_offset_sec=config.initial_event.origin_time_offset_sec,
		fix_depth=config.initial_event.fix_depth,
	)

	write_cmd_template_paths(
		config.paths.cmd_file,
		cmd_run_file,
		sta_file=str(config.paths.sta_file),
		pcrh_file=str(config.paths.pcrh_file),
		scrh_file=str(config.paths.scrh_file),
	)

	with cmd_run_file.open('rb') as stdin:
		result = subprocess.run(
			[str(config.paths.exe_file)],
			stdin=stdin,
			cwd=run_dir,
			capture_output=True,
			text=True,
			check=True,
		)

	print(result.stdout)
	print('returncode:', result.returncode)

	prt_df = load_hypoinverse_summary_from_prt(prt_path)
	prt_plot_df = _filter_plot_df_by_quality(
		prt_df,
		max_erh_km=config.plot_quality_filter.max_erh_km,
		max_erz_km=config.plot_quality_filter.max_erz_km,
	)
	plot_events_map_and_sections(
		df=prt_plot_df,
		prefecture_shp=str(config.paths.prefecture_shp),
		out_png=str(out_location_png),
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


def main(argv: list[str]) -> None:
	args = parse_args(argv)
	config_path = args.config.expanduser().resolve()
	config = load_jma_mobara_hypoinverse_config(config_path)
	run_pipeline(config, config_path=config_path)


if __name__ == '__main__':
	main(sys.argv[1:])
