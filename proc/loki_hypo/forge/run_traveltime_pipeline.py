# %%
from __future__ import annotations

from pathlib import Path

from pipelines.forge_traveltime_pipeline import (
	ForgePipelineConfig,
	run_forge44_traveltime_pipeline,
)

VS_MODEL_PATH = Path(
	'/workspace/data/velocity/forge/Vs_profiles/mod_profiles/44_mod.lst'
)
STATIONS_CSV_PATH = Path('/workspace/data/station/forge/forge_das_station_metadata.csv')
VP_VS = 1.75
MODEL_LABEL = 'forge44_vpvs1p75'
OUTPUT_DIR = Path('/workspace/data/velocity/forge/forge_tt_table/forge44_vpvs1p75')
DX_KM = 0.2
DY_KM = 0.2
DZ_KM = 0.2
PAD_KM = 5.0
Z0_KM = 0.0
ZMAX_KM = 3.00
CENTER_MODE = 'median'
XY_HALF_WIDTH_KM = None
QUANTITY = 'SLOW_LEN'
GTMODE = 'GRID3D ANGLES_NO'
DEPTH_KM_MODE = 'from_elevation'
GT_PLFD_EPS = 1.0e-3
GT_PLFD_SWEEP = 0
RUN_NLL_TOOLS = True


def main():
	cfg = ForgePipelineConfig(
		vs_model_path=VS_MODEL_PATH,
		stations_csv_path=STATIONS_CSV_PATH,
		vp_vs=VP_VS,
		model_label=MODEL_LABEL,
		output_dir=OUTPUT_DIR,
		dx_km=DX_KM,
		dy_km=DY_KM,
		dz_km=DZ_KM,
		pad_km=PAD_KM,
		z0_km=Z0_KM,
		zmax_km=ZMAX_KM,
		center_mode=CENTER_MODE,
		xy_half_width_km=XY_HALF_WIDTH_KM,
		quantity=QUANTITY,
		gtmode=GTMODE,
		depth_km_mode=DEPTH_KM_MODE,
		gt_plfd_eps=GT_PLFD_EPS,
		gt_plfd_sweep=GT_PLFD_SWEEP,
		run_nll_tools=RUN_NLL_TOOLS,
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
