# %%
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import pandas as pd

from loki_tool.grid import GridSpec

# ---- Project-default paths (adjust if your layout changes) ----
DEFAULT_LAYERS_PATH = Path('velocity/jma2001.layers')

DEFAULT_NLL_RUN_DIR = Path('nll/run')
DEFAULT_NLL_MODEL_DIR = Path('nll/model')
DEFAULT_NLL_TIME_DIR = Path('nll/time')


def _require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
	missing = [c for c in cols if c not in df.columns]
	if missing:
		raise ValueError(f'missing required columns in stations_df: {missing}')


def _validate_grid(grid: GridSpec) -> None:
	if grid.nx <= 0 or grid.ny <= 0 or grid.nz <= 0:
		raise ValueError('nx, ny, nz must be positive')
	if grid.dx_km <= 0 or grid.dy_km <= 0 or grid.dz_km <= 0:
		raise ValueError('dx_km, dy_km, dz_km must be positive')


def _station_rows_unique(stations_df: pd.DataFrame) -> pd.DataFrame:
	"""Expect station-unique rows.
	If duplicates exist, keep first to avoid ambiguous GTSRCE lines.
	"""
	_require_columns(stations_df, ['station', 'lat', 'lon'])

	df = stations_df.copy()
	if 'elevation_m' not in df.columns:
		df['elevation_m'] = 0.0

	df['lat'] = df['lat'].astype(float)
	df['lon'] = df['lon'].astype(float)
	df['elevation_m'] = df['elevation_m'].astype(float)

	df = (
		df.groupby('station', as_index=False)
		.agg({'lat': 'first', 'lon': 'first', 'elevation_m': 'first'})
		.sort_values('station')
		.reset_index(drop=True)
	)
	if df.empty:
		raise ValueError('stations_df is empty after station grouping')

	return df


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


def _format_gtsrce_lines(
	stations_df: pd.DataFrame,
	*,
	depth_km_mode: Literal['zero', 'from_elevation'] = 'zero',
) -> list[str]:
	"""Create GTSRCE lines.

	depth_km_mode:
		- "zero": depth=0.0 for all stations (safe default)
		- "from_elevation": use -elevation_m/1000 as depth_km

	Note:
		The exact GTSRCE syntax can be tuned later if you decide to
		encode station depth differently.

	"""
	df = _station_rows_unique(stations_df)

	lines: list[str] = []
	for sta, lat, lon, elev_m in df[
		['station', 'lat', 'lon', 'elevation_m']
	].itertuples(index=False, name=None):
		if depth_km_mode == 'zero':
			depth_km = 0.0
		elif depth_km_mode == 'from_elevation':
			depth_km = -float(elev_m) / 1000.0
		else:
			raise ValueError(f'unsupported depth_km_mode: {depth_km_mode}')

		# Conservative, commonly-seen pattern:
		# GTSRCE <label> LATLON <lat> <lon> <depth_km> 0
		lines.append(
			f'GTSRCE {sta} LATLON {float(lat):.6f} {float(lon):.6f} {depth_km:.3f} 0'
		)

	return lines


def generate_nll_control_text(
	grid: GridSpec,
	stations_df: pd.DataFrame,
	*,
	phase: Literal['P', 'S'],
	model_label: str = 'jma2001',
	layers_path: Path = DEFAULT_LAYERS_PATH,
	vgout_dir: Path = DEFAULT_NLL_MODEL_DIR,
	gtout_dir: Path = DEFAULT_NLL_TIME_DIR,
	control_id: int = 1,
	random_seed: int = 54321,
	quantity: str = 'SLOW_LEN',
	gtmode: str = 'GRID3D ANGLES_NO',
	depth_km_mode: Literal['zero', 'from_elevation'] = 'zero',
) -> str:
	"""Generate a single NonLinLoc control file text for a given phase.

	This text is intended to be used for:
		Vel2Grid <control>
		Grid2Time <control>

	The output root names are derived from:
		vgout_dir / model_label
		gtout_dir / model_label

	Adjust gtmode or quantity if your NonLinLoc build expects different tokens.
	"""
	_validate_grid(grid)
	_require_columns(stations_df, ['station', 'lat', 'lon'])

	vg_root = (Path(vgout_dir) / model_label).as_posix()
	gt_root = (Path(gtout_dir) / model_label).as_posix()
	layers_inc = Path(layers_path).as_posix()

	lines: list[str] = []

	# ---- Generic ----
	lines.append(f'CONTROL {control_id} {random_seed}')
	lines.append(_format_trans_simple(grid))

	# ---- Vel2Grid ----
	lines.append(f'VGOUT {vg_root}')
	lines.append(f'VGTYPE {phase}')
	lines.append(_format_vggrid(grid, quantity=quantity))
	lines.append(f'INCLUDE {layers_inc}')

	# ---- Grid2Time ----
	# Typical pattern: GTFILES <vg_root> <gt_root> <phase>
	lines.append(f'GTFILES {vg_root} {gt_root} {phase}')
	lines.append(f'GTMODE {gtmode}')

	# ---- Stations as sources ----
	lines.extend(_format_gtsrce_lines(stations_df, depth_km_mode=depth_km_mode))

	return '\n'.join(lines) + '\n'


def write_nll_control_files_ps(
	grid: GridSpec,
	stations_df: pd.DataFrame,
	*,
	model_label: str = 'jma2001',
	layers_path: Path = DEFAULT_LAYERS_PATH,
	run_dir: Path = DEFAULT_NLL_RUN_DIR,
	vgout_dir: Path = DEFAULT_NLL_MODEL_DIR,
	gtout_dir: Path = DEFAULT_NLL_TIME_DIR,
	quantity: str = 'SLOW_LEN',
	gtmode: str = 'GRID3D ANGLES_NO',
	depth_km_mode: Literal['zero', 'from_elevation'] = 'zero',
) -> tuple[Path, Path]:
	"""Write two control files:
		<run_dir>/<model_label>_P.in
		<run_dir>/<model_label>_S.in

	Returns:
		(p_path, s_path)

	"""
	run_dir = Path(run_dir)
	run_dir.mkdir(parents=True, exist_ok=True)

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
	)

	p_path = run_dir / f'{model_label}_P.in'
	s_path = run_dir / f'{model_label}_S.in'

	p_path.write_text(p_text)
	s_path.write_text(s_text)

	return p_path, s_path


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
