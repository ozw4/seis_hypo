from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from subprocess import run
from typing import Literal

import pandas as pd
from loki_tools.vel1d import convert_1dvel_to_nll_layers

from jma.station_reader import stations_within_radius
from loki_tools.grid import GridSpec, propose_grid_from_stations, write_loki_header
from nonlinloc.control import write_nll_control_files_ps


@dataclass(frozen=True)
class TravelTimePipelineConfig:
	# --- Station selection ---
	center_lat: float
	center_lon: float
	radius_km: float
	channel_table_path: str | Path = (
		'/workspace/proc/util/hinet_util/hinet_channelstbl_20251007'
	)

	# --- Grid proposal ---
	dx_km: float = 1.0
	dy_km: float = 1.0
	dz_km: float = 1.0
	pad_km: float = 10.0
	z0_km: float = -5.0
	zmax_km: float = 80.0
	center_mode: Literal['fixed', 'mean', 'median'] = 'fixed'

	# --- 1D velocity -> NLL LAYER ---
	vel1d_src: str | Path = 'velocity/vjma2001'
	layers_out: str | Path = 'velocity/jma2001.layers'
	strict_1dvel: bool = False

	# --- NonLinLoc control + outputs ---
	model_label: str = 'jma2001'
	nll_run_dir: str | Path = 'nll/run'
	nll_model_dir: str | Path = 'nll/model'
	nll_time_dir: str | Path = 'nll/time'
	quantity: str = 'SLOW_LEN'
	gtmode: str = 'GRID3D ANGLES_NO'
	depth_km_mode: Literal['zero', 'from_elevation'] = 'zero'

	# --- LOKI header output ---
	loki_header_out: str | Path = 'db/header.hdr'


@dataclass(frozen=True)
class TravelTimePipelineResult:
	stations_df: pd.DataFrame
	grid: GridSpec
	layers_path: Path
	loki_header_path: Path
	control_p_path: Path
	control_s_path: Path


def run_traveltime_pipeline(cfg: TravelTimePipelineConfig) -> TravelTimePipelineResult:
	"""1) 半径条件で station rows 抽出
	2) station 分布から LOKI用グリッド提案（center_mode='fixed' を推奨）
	3) LOKI header.hdr 作成
	4) 1D速度ファイル -> NonLinLoc LAYER ファイル生成
	5) NonLinLoc control(P/S) 自動生成
	6) Vel2Grid / Grid2Time を P→S の順に実行

	事前条件:
	- Vel2Grid と Grid2Time が PATH 上にあること
	- 上記 import 先の各ユーティリティがプロジェクトに配置済みであること
	"""
	# --- 1) station rows ---
	stations_df = stations_within_radius(
		cfg.center_lat,
		cfg.center_lon,
		cfg.radius_km,
		cfg.channel_table_path,
		output='rows',
	)

	# --- 2) grid proposal ---
	grid = propose_grid_from_stations(
		stations_df,
		dx_km=cfg.dx_km,
		dy_km=cfg.dy_km,
		dz_km=cfg.dz_km,
		pad_km=cfg.pad_km,
		z0_km=cfg.z0_km,
		zmax_km=cfg.zmax_km,
		center_mode=cfg.center_mode,
		lat0_deg=cfg.center_lat if cfg.center_mode == 'fixed' else None,
		lon0_deg=cfg.center_lon if cfg.center_mode == 'fixed' else None,
	)

	# --- 3) LOKI header ---
	loki_header_path = write_loki_header(
		grid,
		stations_df,
		out_path=cfg.loki_header_out,
	)

	# --- 4) 1Dvel -> LAYER ---
	layers_path = Path(
		convert_1dvel_to_nll_layers(
			src=Path(cfg.vel1d_src),
			out=Path(cfg.layers_out),
			strict=cfg.strict_1dvel,
		)
	)

	# --- 5) control files (P/S) ---
	control_p_path, control_s_path = write_nll_control_files_ps(
		grid,
		stations_df,
		model_label=cfg.model_label,
		layers_path=layers_path,
		run_dir=Path(cfg.nll_run_dir),
		vgout_dir=Path(cfg.nll_model_dir),
		gtout_dir=Path(cfg.nll_time_dir),
		quantity=cfg.quantity,
		gtmode=cfg.gtmode,
		depth_km_mode=cfg.depth_km_mode,
	)

	# --- 6) Execute NonLinLoc tools ---
	# P
	run(['Vel2Grid', str(control_p_path)], check=True)
	run(['Grid2Time', str(control_p_path)], check=True)

	# S
	run(['Vel2Grid', str(control_s_path)], check=True)
	run(['Grid2Time', str(control_s_path)], check=True)

	return TravelTimePipelineResult(
		stations_df=stations_df,
		grid=grid,
		layers_path=layers_path,
		loki_header_path=Path(loki_header_path),
		control_p_path=Path(control_p_path),
		control_s_path=Path(control_s_path),
	)


def run_traveltime_pipeline_default() -> TravelTimePipelineResult:
	"""コード直書き運用のためのデフォルト実行関数。
	必要ならここだけ座標や半径を変える。
	"""
	cfg = TravelTimePipelineConfig(
		center_lat=35.0,
		center_lon=138.0,
		radius_km=80.0,
		center_mode='fixed',
		model_label='jma2001',
	)
	return run_traveltime_pipeline(cfg)
