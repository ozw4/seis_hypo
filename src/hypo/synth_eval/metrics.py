from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common.geo import latlon_to_local_xy_km
from hypo.hypoinverse_prt import load_hypoinverse_summary_from_prt


def evaluate(
	truth_df: pd.DataFrame, prt_path: Path, lat0: float, lon0: float
) -> pd.DataFrame:
	df_hyp = (
		load_hypoinverse_summary_from_prt(prt_path)
		.sort_values('seq')
		.reset_index(drop=True)
	)

	tr = truth_df.sort_values('event_id').reset_index(drop=True)
	tr['seq'] = np.arange(1, len(tr) + 1, dtype=int)

	df = df_hyp.merge(tr, on='seq', how='inner')

	x_km, y_km = latlon_to_local_xy_km(
		df['lat_deg_hyp'].to_numpy(float),
		df['lon_deg_hyp'].to_numpy(float),
		lat0_deg=lat0,
		lon0_deg=lon0,
	)
	df['x_m_hyp'] = x_km * 1000.0
	df['y_m_hyp'] = y_km * 1000.0
	df['z_m_hyp'] = df['depth_km_hyp'].to_numpy(float) * 1000.0

	df['dx_m'] = df['x_m_hyp'] - df['x_m_true']
	df['dy_m'] = df['y_m_hyp'] - df['y_m_true']
	df['dz_m'] = df['z_m_hyp'] - df['z_m_true']

	df['horiz_m'] = np.sqrt(df['dx_m'] ** 2 + df['dy_m'] ** 2)
	df['err3d_m'] = np.sqrt(df['dx_m'] ** 2 + df['dy_m'] ** 2 + df['dz_m'] ** 2)
	return df
