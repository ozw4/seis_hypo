# file: proc/prepare_data/forge/build_nll_layers_and_controls_from_44mod.py
# %%
from __future__ import annotations

from pathlib import Path

import pandas as pd

from loki_tools.grid import propose_grid_from_stations, write_loki_header
from nonlinloc.control import write_nll_control_files_ps

# =========================
# User parameters (edit here)
# =========================

# Inputs
VS_MODEL_PATH = Path(
	'/workspace/data/velocity/forge/Vs_profiles/mod_profiles/44_mod.lst'
)  # 44_mod.lst (1st line is meaningless -> ignored)
STATIONS_CSV_PATH = Path(
	'/workspace/data/station/forge/forge_das_station_metadata.csv'
)  # station meta (must contain depth_m)

# Vp/Vs (easy to swap later)
VP_VS = 1.75

# Outputs
MODEL_LABEL = 'forge44_vpvs1p75'
OUT_PATH = Path('/workspace/data/velocity/forge')
LAYER_PATH = OUT_PATH / '44_mod_vpvs1p75.layers'

# LOKI header (optional but handy for later) -> DO NOT write into .layers
WRITE_LOKI_HEADER = True
LOKI_HEADER_PATH = OUT_PATH / f'{MODEL_LABEL}.loki_header.txt'

# Grid proposal (tune for Forge geometry)
GRID_DX_KM = 0.10
GRID_DY_KM = 0.10
GRID_DZ_KM = 0.10
GRID_PAD_KM = 0.50
GRID_Z0_KM = -0.50
GRID_ZMAX_KM = 3.00
GRID_CENTER_MODE = 'median'  # 'fixed'|'mean'|'median'
GRID_XY_HALF_WIDTH_KM = (
	None  # set float to force symmetric box; None to auto from stations+pad
)

# Depth handling for GTSRCE lines in .in
# NOTE: we feed "elevation_m = -depth_m(portal)" so that "from_elevation" produces depth correctly.
DEPTH_KM_MODE = 'from_elevation'  # 'zero'|'from_elevation'

# Grid2Time stability knobs
GT_PLFD_EPS = 1.0e-3
GT_PLFD_SWEEP = 0

# =========================
# Implementation
# =========================


def _read_vs_model(path: Path) -> pd.DataFrame:
	if not path.is_file():
		raise FileNotFoundError(f'vs model not found: {path}')

	# 1st line is meaningless -> skiprows=1
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

	df = df.sort_values('depth_m').reset_index(drop=True)
	df = df.drop_duplicates(subset=['depth_m'], keep='first').reset_index(drop=True)

	if (df['depth_m'] < 0).any():
		raise ValueError('depth_m must be non-negative')

	if (df['vs_mps'] <= 0).any():
		raise ValueError('vs_mps must be positive')

	return df[['depth_m', 'vs_mps', 'sigma_mps']]


def _to_nll_layers_text(df_vs: pd.DataFrame, *, vp_vs: float) -> str:
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


def _read_stations(path: Path) -> pd.DataFrame:
	if not path.is_file():
		raise FileNotFoundError(f'stations csv not found: {path}')

	df = pd.read_csv(path)
	if df.empty:
		raise ValueError(f'stations csv is empty: {path}')

	# station id
	if 'station' not in df.columns:
		if 'station_id' in df.columns:
			df = df.copy()
			df['station'] = df['station_id'].astype(str)
		else:
			raise ValueError("stations csv must contain 'station' or 'station_id'")

	# lat/lon
	if 'lat' not in df.columns or 'lon' not in df.columns:
		raise ValueError("stations csv must contain 'lat' and 'lon'")

	# depth_m (portal-based, positive downward) is REQUIRED
	if 'depth_m' not in df.columns:
		raise ValueError(
			"stations csv must contain 'depth_m' (portal-based depth in meters)"
		)

	df = df.copy()
	df['station'] = df['station'].astype(str)
	df['lat'] = df['lat'].astype(float)
	df['lon'] = df['lon'].astype(float)
	df['depth_m'] = pd.to_numeric(df['depth_m'], errors='coerce')

	if df['depth_m'].isna().any():
		raise ValueError('depth_m has NaN; fix station metadata')

	if (df['depth_m'] < 0).any():
		raise ValueError('depth_m must be non-negative (portal-based depth)')

	# NonLinLoc helper expects elevation-like sign in some paths:
	# elevation_m positive upward, so below the portal becomes negative.
	df['elevation_m'] = -df['depth_m'].astype(float)

	return df[['station', 'lat', 'lon', 'depth_m', 'elevation_m']].copy()


def main() -> tuple[Path, Path, Path]:
	df_vs = _read_vs_model(VS_MODEL_PATH)
	layers_text = _to_nll_layers_text(df_vs, vp_vs=VP_VS)

	OUT_PATH.mkdir(parents=True, exist_ok=True)
	LAYER_PATH.write_text(layers_text, encoding='utf-8')

	stations_df = _read_stations(STATIONS_CSV_PATH)

	grid = propose_grid_from_stations(
		stations_df,
		dx_km=GRID_DX_KM,
		dy_km=GRID_DY_KM,
		dz_km=GRID_DZ_KM,
		pad_km=GRID_PAD_KM,
		xy_half_width_km=GRID_XY_HALF_WIDTH_KM,
		z0_km=GRID_Z0_KM,
		zmax_km=GRID_ZMAX_KM,
		center_mode=GRID_CENTER_MODE,
		lat0_deg=None,
		lon0_deg=None,
	)

	if WRITE_LOKI_HEADER:
		LOKI_HEADER_PATH.parent.mkdir(parents=True, exist_ok=True)
		write_loki_header(grid, stations_df, out_path=LOKI_HEADER_PATH)

	p_in, s_in = write_nll_control_files_ps(
		grid,
		stations_df,
		model_label=MODEL_LABEL,
		layers_path=LAYER_PATH,
		run_dir=OUT_PATH,
		vgout_dir=OUT_PATH,
		gtout_dir=OUT_PATH,
		quantity='SLOW_LEN',
		gtmode='GRID3D ANGLES_NO',
		depth_km_mode=DEPTH_KM_MODE,
		gt_plfd_eps=GT_PLFD_EPS,
		gt_plfd_sweep=GT_PLFD_SWEEP,
	)

	return LAYER_PATH, p_in, s_in


if __name__ == '__main__':
	layers_path, p_in_path, s_in_path = main()
	print('layers:', layers_path)
	print('P.in  :', p_in_path)
	print('S.in  :', s_in_path)
	if WRITE_LOKI_HEADER:
		print('loki :', LOKI_HEADER_PATH)
