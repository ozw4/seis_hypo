# file: proc/prepare_data/forge/build_nll_layers_and_controls_from_44mod.py
# %%
from __future__ import annotations

from pathlib import Path

from common.stations import read_forge_stations_portal_depth
from loki_tools.grid import propose_grid_from_stations, write_loki_header
from nonlinloc.control import write_nll_control_files_ps
from nonlinloc.vel1d import nll_layers_text_from_1d_model, read_vs_model_44mod

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

def main() -> tuple[Path, Path, Path]:
	df_vs = read_vs_model_44mod(VS_MODEL_PATH)
	layers_text = nll_layers_text_from_1d_model(
		z_m=df_vs['depth_m'].to_numpy(),
		vs_mps=df_vs['vs_mps'].to_numpy(),
		vp_over_vs=VP_VS,
	)

	OUT_PATH.mkdir(parents=True, exist_ok=True)
	LAYER_PATH.write_text(layers_text, encoding='utf-8')

	stations_df = read_forge_stations_portal_depth(STATIONS_CSV_PATH)

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
