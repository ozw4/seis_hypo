from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from subprocess import run

import pandas as pd

from common.config import (
	DEFAULT_LAYERS_OUT,
	DEFAULT_LOKI_HEADER_OUT,
	DEFAULT_NLL_MODEL_DIR,
	DEFAULT_NLL_RUN_DIR,
	TravelTimeBaseConfig,
)
from common.stations import normalize_station_rows
from jma.station_reader import stations_within_radius
from loki_tools.grid import GridSpec, propose_grid_from_stations, write_loki_header
from nonlinloc.control import write_nll_control_files_ps
from nonlinloc.vel1d import convert_1dvel_to_nll_layers


@dataclass(frozen=True)
class TravelTimePipelineResult:
	stations_df: pd.DataFrame
	grid: GridSpec
	layers_path: Path
	loki_header_path: Path
	control_p_path: Path
	control_s_path: Path
	loki_db_dir: Path


def run_traveltime_pipeline(cfg: TravelTimeBaseConfig) -> TravelTimePipelineResult:
	out_root = Path(cfg.output_dir)
	out_root.mkdir(parents=True, exist_ok=True)

	# 出力先を output_dir / DEFAULT_* に統一
	nll_run_dir = out_root / DEFAULT_NLL_RUN_DIR
	nll_model_dir = out_root / DEFAULT_NLL_MODEL_DIR
	db_dir = out_root / DEFAULT_LOKI_HEADER_OUT.parent

	for d in (
		nll_run_dir,
		nll_model_dir,
		db_dir,
		(out_root / DEFAULT_LAYERS_OUT).parent,
	):
		d.mkdir(parents=True, exist_ok=True)

	# --- 1) station rows ---
	stations_df = stations_within_radius(
		cfg.center_lat,
		cfg.center_lon,
		cfg.radius_km,
		cfg.channel_table_path,
		output='rows',
	)
	# stations_within_radius may return: list[str] | DataFrame | (list[str], DataFrame)
	# Normalize to a pandas.DataFrame so type checkers and downstream code are satisfied.
	if isinstance(stations_df, tuple):
		# (list[str], DataFrame) -> take the DataFrame
		_, stations_df = stations_df
	if isinstance(stations_df, list):
		# list[str] -> convert to a DataFrame with a single 'station' column
		stations_df = pd.DataFrame({'station': stations_df})

	# Normalize columns/types and enforce station-unique rows with elevation present.
	stations_df = normalize_station_rows(stations_df, require_elevation=True)

	# --- 2) grid proposal ---
	grid = propose_grid_from_stations(
		stations_df,
		dx_km=cfg.dx_km,
		dy_km=cfg.dy_km,
		dz_km=cfg.dz_km,
		pad_km=cfg.pad_km,
		xy_half_width_km=cfg.xy_half_width_km,
		z0_km=cfg.z0_km,
		zmax_km=cfg.zmax_km,
		center_mode=cfg.center_mode,
		lat0_deg=cfg.center_lat if cfg.center_mode == 'fixed' else None,
		lon0_deg=cfg.center_lon if cfg.center_mode == 'fixed' else None,
	)

	# --- 3) LOKI header ---（output_dir/db/header.hdr 固定）
	loki_header_path = write_loki_header(
		grid,
		stations_df,
		out_path=out_root / DEFAULT_LOKI_HEADER_OUT,
	)

	# --- 4) 1Dvel -> LAYER ---（output_dir/velocity/... 固定）
	layers_out = out_root / DEFAULT_LAYERS_OUT
	layers_path = Path(
		convert_1dvel_to_nll_layers(
			src=Path(cfg.vel1d_src),
			out=layers_out,
			strict=cfg.strict_1dvel,
		)
	)

	# --- 5) control files (P/S) ---
	# ★根本：LOKIのjoin前提に合わせて、.time.buf 出力先(gtout_dir)を db_dir に統一する
	#    => header.hdr と .time.buf が同じ output_dir/db に同居
	control_p_path, control_s_path = write_nll_control_files_ps(
		grid,
		stations_df,
		model_label=cfg.model_label,
		layers_path=layers_path,
		run_dir=nll_run_dir,
		vgout_dir=nll_model_dir,
		gtout_dir=db_dir,  # ★ここが重要
		quantity=cfg.quantity,
		gtmode=cfg.gtmode,
		depth_km_mode=cfg.depth_km_mode,
	)

	# --- 6) Execute NonLinLoc tools ---
	run(['Vel2Grid', str(control_p_path)], check=True)
	run(['Grid2Time', str(control_p_path)], check=True)

	run(['Vel2Grid', str(control_s_path)], check=True)
	run(['Grid2Time', str(control_s_path)], check=True)

	return TravelTimePipelineResult(
		stations_df=stations_df,
		grid=grid,
		layers_path=layers_path,
		loki_header_path=Path(loki_header_path),
		control_p_path=Path(control_p_path),
		control_s_path=Path(control_s_path),
		loki_db_dir=db_dir,
	)
