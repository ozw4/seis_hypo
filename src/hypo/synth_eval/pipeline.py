from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from hypo.arc import write_hypoinverse_arc_from_phases
from hypo.crh import write_crh
from hypo.phase_jma import extract_phase_records
from hypo.phase_weights import override_phase_weight_by_station_prefix
from hypo.sta import write_hypoinverse_sta
from hypo.station_delays import add_p_and_s_delays_from_elevation

from .builders import (
	build_epic_df,
	build_meas_df,
	build_station_df,
	build_truth_df,
)
from .hypoinverse_runner import run_hypoinverse, write_cmd_from_template
from .metrics import evaluate
from .validation import (
	require_abs,
	require_dirname_only,
	require_filename_only,
)


@dataclass(frozen=True)
class Config:
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

	apply_station_elevation_delay: bool

	das_station_prefix: str
	das_phase_weight_code: int


@dataclass(frozen=True)
class SimParams:
	vp_kms: float
	vs_kms: float


def load_config(path: Path) -> Config:
	obj = yaml.safe_load(path.read_text(encoding='utf-8'))
	return Config(
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
		apply_station_elevation_delay=bool(
			obj.get('apply_station_elevation_delay', True)
		),
		das_station_prefix=str(obj.get('das_station_prefix', 'D')),
		das_phase_weight_code=int(obj.get('das_phase_weight_code', 3)),
	)


def _read_sim_yaml(sim_yaml: Path) -> SimParams:
	obj = yaml.safe_load(sim_yaml.read_text(encoding='utf-8'))
	model = obj['model']
	return SimParams(
		vp_kms=float(model['vp_mps']) / 1000.0,
		vs_kms=float(model['vs_mps']) / 1000.0,
	)


def run_synth_eval(
	config_path: Path, *, runs_root: Path
) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
	if not config_path.is_file():
		raise FileNotFoundError(f'config not found: {config_path}')

	cfg = load_config(config_path)

	dataset_dir = Path(cfg.dataset_dir)
	template_cmd = Path(cfg.template_cmd)
	hypoinverse_exe = Path(cfg.hypoinverse_exe)

	require_abs(dataset_dir, 'dataset_dir')
	require_abs(template_cmd, 'template_cmd')
	require_abs(hypoinverse_exe, 'hypoinverse_exe')

	require_filename_only(cfg.sim_yaml, 'sim_yaml')
	require_filename_only(cfg.receiver_geometry, 'receiver_geometry')
	require_dirname_only(cfg.outputs_dir, 'outputs_dir')

	sim_yaml = dataset_dir / cfg.sim_yaml
	receiver_geometry = dataset_dir / 'geometry' / cfg.receiver_geometry
	index_csv = dataset_dir / 'index.csv'
	events_dir = dataset_dir / 'events'

	if not sim_yaml.is_file():
		raise FileNotFoundError(f'missing: {sim_yaml}')
	if not receiver_geometry.is_file():
		raise FileNotFoundError(f'missing: {receiver_geometry}')
	if not index_csv.is_file():
		raise FileNotFoundError(f'missing: {index_csv}')
	if not events_dir.is_dir():
		raise FileNotFoundError(f'missing: {events_dir}')
	if not template_cmd.is_file():
		raise FileNotFoundError(f'missing: {template_cmd}')
	if not hypoinverse_exe.is_file():
		raise FileNotFoundError(f'missing: {hypoinverse_exe}')

	run_dir = runs_root / cfg.outputs_dir
	run_dir.mkdir(parents=True, exist_ok=True)

	station_csv = run_dir / 'station_synth.csv'
	sta_file = run_dir / 'stations_synth.sta'
	arc_file = run_dir / 'hypoinverse_input.arc'
	p_crh = run_dir / 'P.crh'
	s_crh = run_dir / 'S.crh'
	cmd_file = run_dir / 'synth.cmd'

	prt_file = run_dir / 'hypoinverse_run.prt'
	eval_csv = run_dir / 'eval_metrics.csv'
	eval_stats_csv = run_dir / 'eval_stats.csv'

	sim = _read_sim_yaml(sim_yaml)
	origin0 = pd.to_datetime(cfg.origin0)

	recv_xyz_m = np.load(receiver_geometry).astype(float)
	station_df = build_station_df(recv_xyz_m, cfg.station_set, cfg.lat0, cfg.lon0)
	if cfg.apply_station_elevation_delay:
		station_df = add_p_and_s_delays_from_elevation(
			station_df,
			vp_kms=float(sim.vp_kms),
			vs_kms=float(sim.vs_kms),
		)

	truth_df = build_truth_df(
		index_csv, cfg.lat0, cfg.lon0, origin0, cfg.dt_sec, cfg.max_events
	)
	epic_df = build_epic_df(truth_df, cfg.default_depth_km)
	meas_df = build_meas_df(events_dir, truth_df, station_df, cfg.station_set)

	station_df.to_csv(station_csv, index=False)
	write_hypoinverse_sta(station_csv, sta_file)

	phases = extract_phase_records(meas_df)
	phases = override_phase_weight_by_station_prefix(
		phases,
		station_prefix=str(cfg.das_station_prefix),
		weight=int(cfg.das_phase_weight_code),
	)

	write_hypoinverse_arc_from_phases(
		epic_df,
		phases,
		station_csv,
		arc_file,
		default_depth_km=float(cfg.default_depth_km),
		use_jma_flag=bool(cfg.arc_use_jma_flag),
		p_centroid_top_n=int(cfg.arc_p_centroid_top_n),
		origin_time_offset_sec=float(cfg.arc_origin_time_offset_sec),
		fix_depth=bool(cfg.fix_depth),
	)

	write_crh(p_crh, 'SYNTH_P', [(float(sim.vp_kms), 0.0)])
	write_crh(s_crh, 'SYNTH_S', [(float(sim.vs_kms), 0.0)])

	write_cmd_from_template(template_cmd, cmd_file)
	run_hypoinverse(hypoinverse_exe, cmd_file, run_dir)

	if not prt_file.is_file():
		raise FileNotFoundError(f'missing: {prt_file}')

	df_eval = evaluate(truth_df, prt_file, cfg.lat0, cfg.lon0)
	df_eval.to_csv(eval_csv, index=False)

	metrics_cols = ['horiz_m', 'dz_m', 'err3d_m', 'RMS', 'ERH', 'ERZ']
	missing = [c for c in metrics_cols if c not in df_eval.columns]
	if missing:
		raise ValueError(f'eval df missing columns: {missing}')

	stats = df_eval[metrics_cols].describe(percentiles=[0.5, 0.9, 0.95])
	stats.to_csv(eval_stats_csv)

	return run_dir, df_eval, stats
