# %%
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from common.stations import normalize_station_rows
from loki_tools.grid import GridSpec

# ---- Project-default paths (adjust if your layout changes) ----
DEFAULT_LAYERS_PATH = Path('velocity/jma2001.layers')

DEFAULT_NLL_RUN_DIR = Path('nll/run')
DEFAULT_NLL_MODEL_DIR = Path('nll/model')
DEFAULT_NLL_TIME_DIR = Path('nll/time')


def _validate_grid(grid: GridSpec) -> None:
	if grid.nx <= 0 or grid.ny <= 0 or grid.nz <= 0:
		raise ValueError('nx, ny, nz must be positive')
	if grid.dx_km <= 0 or grid.dy_km <= 0 or grid.dz_km <= 0:
		raise ValueError('dx_km, dy_km, dz_km must be positive')


def _format_trans_simple(grid: GridSpec) -> str:
	"""NonLinLoc SIMPLE transform line."""
	return f'TRANS SIMPLE {grid.lat0_deg:.6f} {grid.lon0_deg:.6f} 0.0'


def _format_vggrid(grid: GridSpec, *, quantity: str = 'SLOW_LEN') -> str:
	"""NonLinLoc VGGRID line.
	quantity is commonly SLOW_LEN for Grid2Time workflows.
	"""
	return (
		f'VGGRID {grid.nx} {grid.ny} {grid.nz} '
		f'{grid.x0_km:.3f} {grid.y0_km:.3f} {grid.z0_km:.3f} '
		f'{grid.dx_km:.3f} {grid.dy_km:.3f} {grid.dz_km:.3f} '
		f'{quantity}'
	)


def write_nll_control_files_ps(
	grid: GridSpec,
	stations_df: pd.DataFrame,
	*,
	model_label: str = 'jma2001',
	layers_path: Path = Path('velocity/jma2001.layers'),
	run_dir: Path = Path('nll/run'),
	vgout_dir: Path = Path('nll/model'),
	gtout_dir: Path = Path('nll/time'),
	quantity: str = 'SLOW_LEN',
	gtmode: str = 'GRID3D ANGLES_NO',
	depth_km_mode: Literal['zero', 'from_elevation'] = 'zero',
	# ★ Grid2Time が落ちる原因だった “走時計算法指定” を渡すため追加
	gt_plfd_eps: float = 1.0e-3,
	gt_plfd_sweep: int = 0,
) -> tuple[Path, Path]:
	"""Write two control files:
		<run_dir>/<model_label>_P.in
		<run_dir>/<model_label>_S.in

	Notes
	-----
	- run_dir/ vgout_dir/ gtout_dir をここで必ず作成する。
	- Grid2Time 用の GT_PLFD パラメータを generate 側へ渡す。
	  generate_nll_control_text も同名引数を受け取り、
	  GTMODE の直後に `GT_PLFD {eps} {sweep}` を出力する実装にしておくこと。

	Returns
	-------
	(p_path, s_path)

	"""
	run_dir = Path(run_dir)
	vgout_dir = Path(vgout_dir)
	gtout_dir = Path(gtout_dir)

	run_dir.mkdir(parents=True, exist_ok=True)
	vgout_dir.mkdir(parents=True, exist_ok=True)
	gtout_dir.mkdir(parents=True, exist_ok=True)

	# LAYER 出力先がこの関数の責務外でも、存在しないと後続で詰むので親だけ作る
	Path(layers_path).parent.mkdir(parents=True, exist_ok=True)

	p_text = generate_nll_control_text(
		grid,
		stations_df,
		phase='P',
		model_label=model_label,
		layers_path=layers_path,
		vgout_dir=vgout_dir,
		gtout_dir=gtout_dir,
		quantity=quantity,
		gtmode=gtmode,
		depth_km_mode=depth_km_mode,
		gt_plfd_eps=gt_plfd_eps,
		gt_plfd_sweep=gt_plfd_sweep,
	)
	s_text = generate_nll_control_text(
		grid,
		stations_df,
		phase='S',
		model_label=model_label,
		layers_path=layers_path,
		vgout_dir=vgout_dir,
		gtout_dir=gtout_dir,
		quantity=quantity,
		gtmode=gtmode,
		depth_km_mode=depth_km_mode,
		gt_plfd_eps=gt_plfd_eps,
		gt_plfd_sweep=gt_plfd_sweep,
	)

	p_path = run_dir / f'{model_label}_P.in'
	s_path = run_dir / f'{model_label}_S.in'

	p_path.write_text(p_text)
	s_path.write_text(s_text)

	return p_path, s_path


def generate_nll_control_text(
	grid: GridSpec,
	stations_df: pd.DataFrame,
	*,
	phase: Literal['P', 'S'],
	model_label: str,
	layers_path: Path,
	vgout_dir: Path,
	gtout_dir: Path,
	quantity: str,
	gtmode: str,
	depth_km_mode: Literal['zero', 'from_elevation'],
	gt_plfd_eps: float = 1.0e-3,
	gt_plfd_sweep: int = 0,
) -> str:
	"""NonLinLoc Vel2Grid/Grid2Time 用 control text を生成する。

	必須修正ポイント:
	- GTMODE の直後に GT_PLFD を必ず出す。
	- depth_km_mode="from_elevation" のとき、elevation_m を深さ(下向き正)へ変換して
	  GTSRCE の zSrce に入れる（陸上は負、海底は正）。
	"""
	_validate_grid(grid)

	lines: list[str] = []

	lines.append('CONTROL 1 54321')
	lines.append(_format_trans_simple(grid))
	lines.append(f'VGOUT {Path(vgout_dir).as_posix()}/{model_label}')
	lines.append(f'VGTYPE {phase}')
	lines.append(_format_vggrid(grid, quantity=quantity))
	lines.append(f'INCLUDE {Path(layers_path).as_posix()}')

	lines.append(
		f'GTFILES {Path(vgout_dir).as_posix()}/{model_label} '
		f'{Path(gtout_dir).as_posix()}/{model_label} {phase}'
	)
	lines.append(f'GTMODE {gtmode}')
	lines.append(f'GT_PLFD {gt_plfd_eps:.1e} {gt_plfd_sweep}')

	require_elev = depth_km_mode == 'from_elevation'
	df = normalize_station_rows(stations_df, require_elevation=require_elev)

	for sta, lat, lon, elev_m in df[
		['station', 'lat', 'lon', 'elevation_m']
	].itertuples(index=False, name=None):
		if depth_km_mode == 'zero':
			dep_km = 0.0
		elif depth_km_mode == 'from_elevation':
			dep_km = -float(elev_m) / 1000.0
		else:
			raise ValueError(f'unsupported depth_km_mode: {depth_km_mode}')

		lines.append(
			f'GTSRCE {sta} LATLON {float(lat):.6f} {float(lon):.6f} {dep_km:.3f} 0'
		)

	return '\n'.join(lines) + '\n'


# ---- In-code usage example (non-CLI) ----
def build_controls_example(
	grid: GridSpec,
	stations_df: pd.DataFrame,
) -> tuple[Path, Path]:
	"""Your happy path:
		1) stations_within_radius(..., output="rows") -> stations_df
		2) propose_grid_from_stations(...) -> grid
		3) write_nll_control_files_ps(grid, stations_df)

	This helper just forwards to the writer.
	"""
	return write_nll_control_files_ps(
		grid,
		stations_df,
		model_label='jma2001',
		layers_path=DEFAULT_LAYERS_PATH,
		run_dir=DEFAULT_NLL_RUN_DIR,
		vgout_dir=DEFAULT_NLL_MODEL_DIR,
		gtout_dir=DEFAULT_NLL_TIME_DIR,
		quantity='SLOW_LEN',
		gtmode='GRID3D ANGLES_NO',
		depth_km_mode='zero',
	)
