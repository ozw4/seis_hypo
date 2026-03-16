from __future__ import annotations

import pandas as pd


_INITIAL_REQUIRED_COLS = [
	'event_id',
	'origin_time',
	'latitude_deg',
	'longitude_deg',
	'depth_km',
]


def build_hypoinverse_event_export_df(
	initial_event_df: pd.DataFrame,
	hyp_df: pd.DataFrame,
) -> pd.DataFrame:
	"""初期イベント表と HypoInverse 出力を event_id ベースで結合する。"""
	missing_initial = [
		col for col in _INITIAL_REQUIRED_COLS if col not in initial_event_df.columns
	]
	if missing_initial:
		raise ValueError(
			f'initial_event_df is missing required columns: {missing_initial}'
		)
	if 'id_no_prt' not in hyp_df.columns:
		raise ValueError("hyp_df is missing required column: ['id_no_prt']")

	initial_df = initial_event_df.loc[:, _INITIAL_REQUIRED_COLS].copy()
	initial_df = initial_df.rename(
		columns={
			'origin_time': 'origin_time_init',
			'latitude_deg': 'lat_deg_init',
			'longitude_deg': 'lon_deg_init',
			'depth_km': 'depth_km_init',
		}
	)
	initial_df['event_id'] = pd.to_numeric(
		initial_df['event_id'],
		errors='raise',
	).astype('Int64')

	out = hyp_df.copy()
	out['event_id'] = pd.to_numeric(out['id_no_prt'], errors='coerce').astype('Int64')
	if out['event_id'].isna().any():
		raise ValueError('hyp_df.id_no_prt contains non-numeric values')

	out = out.merge(initial_df, on='event_id', how='left')

	front_cols = [
		'event_id',
		'sequence_no_prt',
		'id_no_prt',
		'seq',
		'origin_time_init',
		'lat_deg_init',
		'lon_deg_init',
		'depth_km_init',
		'origin_time_hyp',
		'lat_deg_hyp',
		'lon_deg_hyp',
		'depth_km_hyp',
		'RMS',
		'ERH',
		'ERZ',
		'origin_time_err_sec',
		'NSTA',
		'NPHS',
		'DMIN',
		'MODEL',
		'GAP',
		'ITR',
		'NFM',
		'NWR',
		'NWS',
		'NVR',
	]
	ordered_cols = [col for col in front_cols if col in out.columns]
	ordered_cols += [col for col in out.columns if col not in ordered_cols]
	return out.loc[:, ordered_cols]
