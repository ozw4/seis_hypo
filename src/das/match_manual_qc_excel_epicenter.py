# src/catalog/match_excel_epicenters.py
from __future__ import annotations

import numpy as np
import pandas as pd

from common.core import validate_columns

REQUIRED_XL_COLS = [
	'EQseq',
	'LON',
	'LAT',
	'Dep',
	'Mag',
	'origin_time_local',
	'das_score',
]

REQUIRED_EP_COLS = [
	'event_id',
	'origin_time',
	'latitude_deg',
	'longitude_deg',
	'depth_km',
	'mag1',
]


def match_excel_to_epicenters_with_das(
	df_xl: pd.DataFrame,
	df_ep: pd.DataFrame,
	time_tol_sec: float = 1.0,
	lat_tol_deg: float = 0.1,
	lon_tol_deg: float = 0.1,
	dep_tol_km: float = 0.1,
	verbose: bool = True,
) -> tuple[set[int], dict[int, float]]:
	"""Excel と epicenters をマッチさせて、
	対応が取れた event_id の集合と event_id -> das_score 対応を返す。
	"""
	validate_columns(df_xl, REQUIRED_XL_COLS, 'Excel')
	validate_columns(df_ep, REQUIRED_EP_COLS, 'epicenters CSV')

	df_xl = df_xl.copy()
	df_ep = df_ep.copy()

	df_xl['origin_time_local'] = pd.to_datetime(df_xl['origin_time_local'])
	df_ep['origin_time_dt'] = pd.to_datetime(df_ep['origin_time'])

	t_ep = df_ep['origin_time_dt'].to_numpy()
	lat_ep = df_ep['latitude_deg'].to_numpy(dtype=float)
	lon_ep = df_ep['longitude_deg'].to_numpy(dtype=float)
	dep_ep = df_ep['depth_km'].to_numpy(dtype=float)
	event_id_ep = df_ep['event_id'].to_numpy()

	time_tol = pd.Timedelta(seconds=float(time_tol_sec))
	lat_tol = float(lat_tol_deg)
	lon_tol = float(lon_tol_deg)
	dep_tol = float(dep_tol_km)

	matched_event_ids: set[int] = set()
	das_by_event_id: dict[int, float] = {}

	n_unique = 0
	n_ambig = 0
	n_no_time = 0
	n_no_space = 0

	for _, row in df_xl.iterrows():
		eqseq = row['EQseq']
		lon = float(row['LON'])
		lat = float(row['LAT'])
		dep = float(row['Dep'])
		das_score = float(row['das_score'])
		t_ts = pd.to_datetime(row['origin_time_local'])

		t_min = t_ts - time_tol
		t_max = t_ts + time_tol
		mask_time = (t_ep >= t_min) & (t_ep <= t_max)
		idx_time = np.nonzero(mask_time)[0]

		if idx_time.size == 0:
			n_no_time += 1
			if verbose:
				print(
					f'[no_time_match] EQseq={eqseq} '
					f'origin_time={t_ts} '
					f'Excel(LON={lon:.3f}, LAT={lat:.3f}, Dep={dep:.1f})'
				)
			continue

		cand_lat = lat_ep[idx_time]
		cand_lon = lon_ep[idx_time]
		cand_dep = dep_ep[idx_time]
		cand_event_id = event_id_ep[idx_time]
		cand_t = t_ep[idx_time]

		cand_t_series = pd.to_datetime(pd.Series(cand_t))
		time_diff_s_all = np.abs((cand_t_series - t_ts).dt.total_seconds().to_numpy())

		dlat_all = cand_lat - lat
		dlon_all = cand_lon - lon
		ddep_all = cand_dep - dep

		best_idx_time = int(np.argmin(time_diff_s_all))

		best_event_id = int(cand_event_id[best_idx_time])
		best_time_diff_s = float(time_diff_s_all[best_idx_time])
		best_dlat = float(dlat_all[best_idx_time])
		best_dlon = float(dlon_all[best_idx_time])
		best_ddep = float(ddep_all[best_idx_time])
		jma_lon = float(cand_lon[best_idx_time])
		jma_lat = float(cand_lat[best_idx_time])
		jma_dep = float(cand_dep[best_idx_time])

		if (
			abs(best_dlat) <= lat_tol
			and abs(best_dlon) <= lon_tol
			and abs(best_ddep) <= dep_tol
		):
			matched_event_ids.add(best_event_id)

			if best_event_id in das_by_event_id:
				if not np.isclose(das_by_event_id[best_event_id], das_score):
					msg = f'event_id={best_event_id} に複数の異なる das_score が対応しています'
					raise ValueError(msg)
			else:
				das_by_event_id[best_event_id] = das_score

			mask_space = (
				(np.abs(dlat_all) <= lat_tol)
				& (np.abs(dlon_all) <= lon_tol)
				& (np.abs(ddep_all) <= dep_tol)
			)
			n_space = int(mask_space.sum())

			if n_space > 1:
				n_ambig += 1
				if verbose:
					print(
						f'[ambiguous_match] EQseq={eqseq} '
						f'best_event_id={best_event_id} '
						f'best_time_diff_s={best_time_diff_s:.3f} '
						f'delta_lat_deg={best_dlat:.4f} '
						f'delta_lon_deg={best_dlon:.4f} '
						f'delta_depth_km={best_ddep:.2f} '
						f'num_spatial_candidates={n_space}'
					)
			else:
				n_unique += 1
		else:
			n_no_space += 1
			if verbose:
				print(
					'[no_spatial_match] '
					f'EQseq={eqseq} '
					f'origin_time={t_ts} '
					f'Excel(LON={lon:.3f}, LAT={lat:.3f}, Dep={dep:.1f}) '
					f'JMA(LON={jma_lon:.3f}, LAT={jma_lat:.3f}, Dep={jma_dep:.1f}) '
					f'dLON={best_dlon:.3f} dLAT={best_dlat:.3f} dDep={best_ddep:.1f} '
					f'num_time_candidates={int(idx_time.size)} '
					f'min_time_diff_s={best_time_diff_s:.3f}'
				)

	total_rows = len(df_xl)
	if verbose:
		print(
			f'[summary] total_excel_rows={total_rows} '
			f'unique_matches={n_unique} '
			f'ambiguous_matches={n_ambig} '
			f'no_time_match={n_no_time} '
			f'no_spatial_match={n_no_space}'
		)

	if not matched_event_ids:
		raise RuntimeError('対応が取れた event_id が 1 件もありません。')

	return matched_event_ids, das_by_event_id


def subset_by_event_ids(df: pd.DataFrame, event_ids: set[int]) -> pd.DataFrame:
	"""任意の DataFrame を event_id で絞り込む。epicenters / measurements 共通で使う想定。"""
	if 'event_id' not in df.columns:
		raise ValueError('event_id 列がありません。')
	df_sub = df[df['event_id'].isin(event_ids)].copy()
	df_sub = df_sub.sort_values('event_id').reset_index(drop=True)
	return df_sub


def attach_das_score(
	df_ep_subset: pd.DataFrame,
	das_by_event_id: dict[int, float],
) -> pd.DataFrame:
	"""Epicenters サブセットに das_score を付与する。"""
	df_das = pd.DataFrame(
		{
			'event_id': list(das_by_event_id.keys()),
			'das_score': list(das_by_event_id.values()),
		}
	)
	df = df_ep_subset.merge(df_das, on='event_id', how='left')
	return df
