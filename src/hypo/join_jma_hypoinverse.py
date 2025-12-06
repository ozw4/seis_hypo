from pathlib import Path

import pandas as pd

from hypo.arc_alignment import build_arc_event_map
from hypo.hypoinverse_prt import load_hypoinverse_summary_from_prt


def build_joined_jma_hypo_csv(
	epic_df: pd.DataFrame,
	meas_df: pd.DataFrame,
	prt_path: str | Path,
	out_csv: str | Path,
) -> pd.DataFrame:
	"""epic_df / meas_df / .prt から
	JMA 情報 + hypoinverse 情報が 1 行に揃った CSV を 1 本だけ作る。

	出力カラム例:
	seq, event_id,
	origin_time_jma, lat_deg_jma, lon_deg_jma, depth_km_jma,
	origin_time_hyp, lat_deg_hyp, lon_deg_hyp, depth_km_hyp,
	RMS, ERH, ERZ, NSTA, NPHS, DMIN, MODEL, GAP, ITR, NFM, NWR, NWS, NVR
	"""
	prt_path = Path(prt_path)
	out_csv = Path(out_csv)

	df_map = build_arc_event_map(epic_df, meas_df)
	df_hyp = load_hypoinverse_summary_from_prt(prt_path)

	df_join = df_map.merge(df_hyp, on='seq', how='inner')
	if df_join.empty:
		raise RuntimeError('JMA と hypoinverse の対応行が 1 件もありません')

	out_csv.parent.mkdir(parents=True, exist_ok=True)
	df_join.to_csv(out_csv, index=False)
	return df_join
