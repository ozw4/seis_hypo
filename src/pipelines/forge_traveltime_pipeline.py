from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import which
from subprocess import run

import pandas as pd

from common.stations import read_forge_stations_portal_depth
from loki_tools.grid import GridSpec, propose_grid_from_stations, write_loki_header
from nonlinloc.control import write_nll_control_files_ps
from nonlinloc.vel1d import nll_layers_text_from_1d_model, read_vs_model_44mod


@dataclass(frozen=True)
class ForgePipelineConfig:
	# inputs
	vs_model_path: Path
	stations_csv_path: Path

	# model
	vp_vs: float
	model_label: str

	# output root
	output_dir: Path

	# grid
	dx_km: float
	dy_km: float
	dz_km: float
	pad_km: float
	z0_km: float
	zmax_km: float
	center_mode: str
	xy_half_width_km: float | None

	# NLL control
	quantity: str
	gtmode: str
	depth_km_mode: str
	gt_plfd_eps: float
	gt_plfd_sweep: int

	# run tools
	run_nll_tools: bool = True


@dataclass(frozen=True)
class ForgePipelineResult:
	stations_df: pd.DataFrame
	grid: GridSpec
	layers_path: Path
	loki_header_path: Path
	control_p_path: Path
	control_s_path: Path
	nll_model_dir: Path
	loki_db_dir: Path


def _require_executable(name: str) -> None:
	if which(name) is None:
		raise FileNotFoundError(f'executable not found in PATH: {name}')


def run_forge44_traveltime_pipeline(cfg: ForgePipelineConfig) -> ForgePipelineResult:
	# ★ resolve() してから派生ディレクトリを作る（相対パス事故防止）
	out_root = Path(cfg.output_dir).expanduser().resolve()
	out_root.mkdir(parents=True, exist_ok=True)

	nll_run_dir = (out_root / 'nll' / 'run').resolve()
	nll_model_dir = (out_root / 'nll' / 'model').resolve()
	db_dir = (out_root / 'db').resolve()
	vel_dir = (out_root / 'velocity').resolve()

	for d in (nll_run_dir, nll_model_dir, db_dir, vel_dir):
		d.mkdir(parents=True, exist_ok=True)

	# inputsも resolve（意図しない cwd 依存を避ける）
	vs_model_path = Path(cfg.vs_model_path).expanduser().resolve()
	stations_csv_path = Path(cfg.stations_csv_path).expanduser().resolve()

	df_vs = read_vs_model_44mod(vs_model_path)
	layers_text = nll_layers_text_from_1d_model(
		z_m=df_vs['depth_m'].to_numpy(),
		vs_mps=df_vs['vs_mps'].to_numpy(),
		vp_over_vs=cfg.vp_vs,
	)

	layers_path = (vel_dir / f'{cfg.model_label}.layers').resolve()
	layers_path.write_text(layers_text, encoding='utf-8')

	stations_df = read_forge_stations_portal_depth(stations_csv_path)

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
		lat0_deg=None,
		lon0_deg=None,
	)

	loki_header_path = write_loki_header(
		grid,
		stations_df,
		out_path=(db_dir / 'header.hdr').resolve(),
	)

	control_p_path, control_s_path = write_nll_control_files_ps(
		grid,
		stations_df,
		model_label=cfg.model_label,
		layers_path=layers_path,
		run_dir=nll_run_dir,
		vgout_dir=nll_model_dir,
		gtout_dir=db_dir,
		quantity=cfg.quantity,
		gtmode=cfg.gtmode,
		depth_km_mode=cfg.depth_km_mode,
		gt_plfd_eps=cfg.gt_plfd_eps,
		gt_plfd_sweep=cfg.gt_plfd_sweep,
	)

	# writerの戻りが str でも Path でも、最終的にresolveして固定
	control_p_path = Path(control_p_path).expanduser().resolve()
	control_s_path = Path(control_s_path).expanduser().resolve()

	if cfg.run_nll_tools:
		_require_executable('Vel2Grid')
		_require_executable('Grid2Time')

		run(['Vel2Grid', str(control_p_path)], check=True)
		run(['Grid2Time', str(control_p_path)], check=True)
		run(['Vel2Grid', str(control_s_path)], check=True)
		run(['Grid2Time', str(control_s_path)], check=True)

	return ForgePipelineResult(
		stations_df=stations_df,
		grid=grid,
		layers_path=layers_path,
		loki_header_path=Path(loki_header_path),
		control_p_path=control_p_path,
		control_s_path=control_s_path,
		nll_model_dir=nll_model_dir,
		loki_db_dir=db_dir,
	)
