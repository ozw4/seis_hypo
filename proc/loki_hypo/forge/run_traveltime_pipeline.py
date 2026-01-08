# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import which
from subprocess import run

import pandas as pd

from loki_tools.grid import GridSpec, propose_grid_from_stations, write_loki_header
from nonlinloc.control import write_nll_control_files_ps


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


def _read_vs_model_44mod(path: Path) -> pd.DataFrame:
	if not path.is_file():
		raise FileNotFoundError(f'vs model not found: {path}')

	df = pd.read_csv(
		path,
		sep=r'\s+',
		header=None,
		names=['depth_m', 'vs_mps', 'sigma_mps'],
		comment='#',
		skiprows=1,
	)

	for c in ['depth_m', 'vs_mps', 'sigma_mps']:
		df[c] = pd.to_numeric(df[c], errors='coerce')

	df = df.dropna(subset=['depth_m', 'vs_mps']).copy()
	if df.empty:
		raise ValueError(f'no numeric rows found in: {path}')

	df = (
		df.sort_values('depth_m')
		.drop_duplicates(subset=['depth_m'], keep='first')
		.reset_index(drop=True)
	)

	if (df['depth_m'] < 0).any():
		raise ValueError('depth_m must be non-negative')
	if (df['vs_mps'] <= 0).any():
		raise ValueError('vs_mps must be positive')

	return df[['depth_m', 'vs_mps']].copy()


def _layers_text_from_vs(df_vs: pd.DataFrame, *, vp_vs: float) -> str:
	if vp_vs <= 0:
		raise ValueError(f'vp_vs must be positive. got {vp_vs}')

	depth_km = (df_vs['depth_m'].astype(float) / 1000.0).to_numpy()
	vs_km_s = (df_vs['vs_mps'].astype(float) / 1000.0).to_numpy()
	vp_km_s = vs_km_s * float(vp_vs)

	lines: list[str] = []
	for d, vp, vs in zip(depth_km.tolist(), vp_km_s.tolist(), vs_km_s.tolist()):
		lines.append(f'LAYER {d:.3f} {vp:.3f} 0.0 {vs:.3f} 0.0 0.0 0.0')

	text = '\n'.join(lines) + '\n'
	if not text.startswith('LAYER '):
		raise ValueError('layers text must start with LAYER')
	return text


def _read_forge_stations_portal_depth(path: Path) -> pd.DataFrame:
	if not path.is_file():
		raise FileNotFoundError(f'stations csv not found: {path}')

	df = pd.read_csv(path)
	if df.empty:
		raise ValueError(f'stations csv is empty: {path}')

	if 'station' not in df.columns:
		if 'station_id' in df.columns:
			df = df.copy()
			df['station'] = df['station_id'].astype(str)
		else:
			raise ValueError("stations csv must contain 'station' or 'station_id'")

	if 'lat' not in df.columns or 'lon' not in df.columns:
		raise ValueError("stations csv must contain 'lat' and 'lon'")

	if 'depth_m' not in df.columns:
		raise ValueError(
			"stations csv must contain 'depth_m' (portal-based, meters, positive downward)"
		)

	df = df.copy()
	df['station'] = df['station'].astype(str)
	if df['station'].isna().any():
		raise ValueError('station contains NaN')

	df['lat'] = df['lat'].astype(float)
	df['lon'] = df['lon'].astype(float)
	df['depth_m'] = pd.to_numeric(df['depth_m'], errors='coerce')

	if df['depth_m'].isna().any():
		raise ValueError('depth_m has NaN; fix station metadata')
	if (df['depth_m'] < 0).any():
		raise ValueError('depth_m must be non-negative (portal-based depth)')

	df['elevation_m'] = -df['depth_m'].astype(float)
	return df[['station', 'lat', 'lon', 'depth_m', 'elevation_m']].copy()


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

	df_vs = _read_vs_model_44mod(vs_model_path)
	layers_text = _layers_text_from_vs(df_vs, vp_vs=cfg.vp_vs)

	layers_path = (vel_dir / f'{cfg.model_label}.layers').resolve()
	layers_path.write_text(layers_text, encoding='utf-8')

	stations_df = _read_forge_stations_portal_depth(stations_csv_path)

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


def main() -> ForgePipelineResult:
	cfg = ForgePipelineConfig(
		vs_model_path=Path(
			'/workspace/data/velocity/forge/Vs_profiles/mod_profiles/44_mod.lst'
		),
		stations_csv_path=Path(
			'/workspace/data/station/forge/forge_das_station_metadata.csv'
		),
		vp_vs=1.75,
		model_label='forge44_vpvs1p75',
		output_dir=Path(
			'/workspace/data/velocity/forge/forge_tt_table/forge44_vpvs1p75'
		),
		dx_km=0.2,
		dy_km=0.2,
		dz_km=0.2,
		pad_km=5.0,
		z0_km=0.0,
		zmax_km=3.00,
		center_mode='median',
		xy_half_width_km=None,
		quantity='SLOW_LEN',
		gtmode='GRID3D ANGLES_NO',
		depth_km_mode='from_elevation',
		gt_plfd_eps=1.0e-3,
		gt_plfd_sweep=0,
		run_nll_tools=True,
	)
	return run_forge44_traveltime_pipeline(cfg)


if __name__ == '__main__':
	result = main()
	print('layers:', result.layers_path)
	print('header:', result.loki_header_path)
	print('P.in  :', result.control_p_path)
	print('S.in  :', result.control_s_path)
	print('db_dir:', result.loki_db_dir)
	print('model :', result.nll_model_dir)
