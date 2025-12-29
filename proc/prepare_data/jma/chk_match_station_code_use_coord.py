# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# =========================
# 設定（ここだけ直書き）
# =========================
STA_FILE = '/workspace/data/station/jma/station.csv'
CH_FILE = '/workspace/proc/prepare_data/jma/snapshots/monthly/monthly_presence.csv'

STA_CODE_COL = 'station_code'
STA_NUM_COL = 'station_number'
STA_LAT_COL = 'Latitude_deg'
STA_LON_COL = 'Longitude_deg'

CH_CODE_COL = 'station'  # monthly_presence.csv の station 列
CH_LAT_COL = 'lat'
CH_LON_COL = 'lon'

RADIUS_KM = 0.5
MAX_CANDIDATES = 5
CHUNK_SIZE = 300

OUT_DIR = Path('./station_code_match')
# =========================


def _prep_unique_points(
	df: pd.DataFrame,
	code_col: str,
	lat_col: str,
	lon_col: str,
	extra_cols: list[str],
) -> pd.DataFrame:
	need = [code_col, lat_col, lon_col] + extra_cols
	miss = [c for c in need if c not in df.columns]
	if miss:
		raise KeyError(f'missing columns: {miss} / columns={list(df.columns)}')

	x = df[need].copy()
	x[code_col] = x[code_col].dropna().astype(str).str.strip()
	x = x[x[code_col] != '']
	x[lat_col] = pd.to_numeric(x[lat_col], errors='raise')
	x[lon_col] = pd.to_numeric(x[lon_col], errors='raise')
	x = x.dropna(subset=[code_col, lat_col, lon_col])

	agg = {lat_col: 'mean', lon_col: 'mean'}
	for c in extra_cols:
		agg[c] = 'first'

	x = x.groupby(code_col, as_index=False).agg(agg)
	x = x.rename(columns={code_col: 'code', lat_col: 'lat', lon_col: 'lon'})
	return x


def _haversine_km_matrix(
	lat1_deg: np.ndarray,
	lon1_deg: np.ndarray,
	lat2_deg: np.ndarray,
	lon2_deg: np.ndarray,
) -> np.ndarray:
	R = 6371.0088
	lat1 = np.deg2rad(lat1_deg)
	lon1 = np.deg2rad(lon1_deg)
	lat2 = np.deg2rad(lat2_deg)
	lon2 = np.deg2rad(lon2_deg)

	dlat = lat2 - lat1
	dlon = lon2 - lon1

	a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
		np.sin(dlon / 2.0) ** 2
	)
	c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
	return R * c


@dataclass(frozen=True)
class CoordMatch:
	unique: pd.DataFrame
	ambiguous: pd.DataFrame
	unmatched: pd.DataFrame


def match_by_coords(
	src: pd.DataFrame,
	dst: pd.DataFrame,
	*,
	radius_km: float,
	max_candidates: int,
	chunk_size: int,
) -> CoordMatch:
	if radius_km <= 0:
		raise ValueError('radius_km must be positive')
	if chunk_size <= 0:
		raise ValueError('chunk_size must be positive')

	src_lat = src['lat'].to_numpy()
	src_lon = src['lon'].to_numpy()
	dst_lat = dst['lat'].to_numpy()
	dst_lon = dst['lon'].to_numpy()
	dst_code = dst['code'].to_numpy()

	unique_rows: list[dict[str, object]] = []
	amb_rows: list[dict[str, object]] = []
	unmatch_rows: list[dict[str, object]] = []

	n = len(src)
	for start in range(0, n, chunk_size):
		end = min(start + chunk_size, n)
		lat1 = src_lat[start:end].reshape(-1, 1)
		lon1 = src_lon[start:end].reshape(-1, 1)
		lat2 = dst_lat.reshape(1, -1)
		lon2 = dst_lon.reshape(1, -1)

		D = _haversine_km_matrix(lat1, lon1, lat2, lon2)

		for i in range(end - start):
			dist = D[i]
			within = np.where(dist <= radius_km)[0]

			src_code = src.iloc[start + i]['code']
			src_lat_i = float(src.iloc[start + i]['lat'])
			src_lon_i = float(src.iloc[start + i]['lon'])

			if within.size == 0:
				j = int(dist.argmin())
				unmatch_rows.append(
					{
						'src_code': src_code,
						'src_lat': src_lat_i,
						'src_lon': src_lon_i,
						'nearest_dst_code': str(dst_code[j]),
						'nearest_km': float(dist[j]),
					}
				)
				continue

			order = within[np.argsort(dist[within])]
			if order.size == 1:
				j = int(order[0])
				unique_rows.append(
					{
						'src_code': src_code,
						'dst_code': str(dst_code[j]),
						'km': float(dist[j]),
						'src_lat': src_lat_i,
						'src_lon': src_lon_i,
						'dst_lat': float(dst.iloc[j]['lat']),
						'dst_lon': float(dst.iloc[j]['lon']),
					}
				)
			else:
				top = order[:max_candidates]
				cands = [f'{dst_code[j]}:{dist[j]:.3f}km' for j in top]
				amb_rows.append(
					{
						'src_code': src_code,
						'src_lat': src_lat_i,
						'src_lon': src_lon_i,
						'n_candidates': int(order.size),
						'candidates_top': ' | '.join(cands),
					}
				)

	unique = pd.DataFrame(unique_rows).sort_values(
		['km', 'src_code'], ascending=[True, True]
	)
	ambiguous = pd.DataFrame(amb_rows).sort_values(
		['n_candidates', 'src_code'], ascending=[False, True]
	)
	unmatched = pd.DataFrame(unmatch_rows).sort_values(
		['nearest_km', 'src_code'], ascending=[True, True]
	)
	return CoordMatch(unique=unique, ambiguous=ambiguous, unmatched=unmatched)


# ---- load（必要列だけ読む）----
sta_raw = pd.read_csv(
	STA_FILE,
	usecols=[STA_CODE_COL, STA_NUM_COL, STA_LAT_COL, STA_LON_COL],
)
ch_raw = pd.read_csv(
	CH_FILE,
	usecols=[CH_CODE_COL, CH_LAT_COL, CH_LON_COL],
	low_memory=False,
)

sta = _prep_unique_points(
	sta_raw, STA_CODE_COL, STA_LAT_COL, STA_LON_COL, [STA_NUM_COL]
)
ch = _prep_unique_points(ch_raw, CH_CODE_COL, CH_LAT_COL, CH_LON_COL, [])

sta_codes = set(sta['code'])
ch_codes = set(ch['code'])

both = sorted(sta_codes & ch_codes)
only_ch = sorted(ch_codes - sta_codes)
only_sta = sorted(sta_codes - ch_codes)

print('=== code set counts ===')
print('sta unique codes:', len(sta_codes))
print('ch  unique codes:', len(ch_codes))
print('both:', len(both))
print('only_ch:', len(only_ch))
print('only_sta:', len(only_sta))

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- 1) exact match（コード完全一致）----
exact = pd.DataFrame({'ch_code': both})
exact['sta_code'] = exact['ch_code']
exact = exact.merge(
	sta[['code', STA_NUM_COL]].rename(columns={'code': 'sta_code'}),
	on='sta_code',
	how='left',
)
exact['match_type'] = 'exact'
exact['km'] = 0.0
exact.to_csv(OUT_DIR / 'match_exact.csv', index=False)

# ---- 2) coord match for only_ch（chにしかないコードを座標でstaへ）----
ch_only_df = ch[ch['code'].isin(only_ch)].reset_index(drop=True)
cm_ch_to_sta = match_by_coords(
	ch_only_df,
	sta,
	radius_km=RADIUS_KM,
	max_candidates=MAX_CANDIDATES,
	chunk_size=CHUNK_SIZE,
)
ch2sta_unique = cm_ch_to_sta.unique.rename(
	columns={'src_code': 'ch_code', 'dst_code': 'sta_code'}
)
ch2sta_unique = ch2sta_unique.merge(
	sta[['code', STA_NUM_COL]].rename(columns={'code': 'sta_code'}),
	on='sta_code',
	how='left',
)
ch2sta_unique['match_type'] = f'coord_unique_{RADIUS_KM}km'
ch2sta_unique.to_csv(OUT_DIR / 'match_coord_unique_ch_to_sta.csv', index=False)

cm_ch_to_sta.ambiguous.rename(columns={'src_code': 'ch_code'}).to_csv(
	OUT_DIR / 'match_coord_ambiguous_ch_to_sta.csv',
	index=False,
)
cm_ch_to_sta.unmatched.rename(columns={'src_code': 'ch_code'}).to_csv(
	OUT_DIR / 'match_coord_unmatched_ch_to_sta.csv',
	index=False,
)

print('\n=== coord match (ch -> sta) ===')
print('unique:', len(ch2sta_unique))
print('ambiguous:', len(cm_ch_to_sta.ambiguous))
print('unmatched:', len(cm_ch_to_sta.unmatched))

# ---- 3) reverse view（staにしかないコードがchの別コードに吸われてないか）----
sta_only_df = sta[sta['code'].isin(only_sta)].reset_index(drop=True)
cm_sta_to_ch = match_by_coords(
	sta_only_df,
	ch,
	radius_km=RADIUS_KM,
	max_candidates=MAX_CANDIDATES,
	chunk_size=CHUNK_SIZE,
)
sta2ch_unique = cm_sta_to_ch.unique.rename(
	columns={'src_code': 'sta_code', 'dst_code': 'ch_code'}
)
sta2ch_unique = sta2ch_unique.merge(
	sta[['code', STA_NUM_COL]].rename(columns={'code': 'sta_code'}),
	on='sta_code',
	how='left',
)
sta2ch_unique['match_type'] = f'coord_unique_{RADIUS_KM}km'
sta2ch_unique.to_csv(OUT_DIR / 'match_coord_unique_sta_to_ch.csv', index=False)

cm_sta_to_ch.ambiguous.rename(columns={'src_code': 'sta_code'}).to_csv(
	OUT_DIR / 'match_coord_ambiguous_sta_to_ch.csv',
	index=False,
)
cm_sta_to_ch.unmatched.rename(columns={'src_code': 'sta_code'}).to_csv(
	OUT_DIR / 'match_coord_unmatched_sta_to_ch.csv',
	index=False,
)

print('\n=== coord match (sta -> ch) ===')
print('unique:', len(sta2ch_unique))
print('ambiguous:', len(cm_sta_to_ch.ambiguous))
print('unmatched:', len(cm_sta_to_ch.unmatched))

# ---- 4) ch->sta の「採用候補」マップ（exact + coord_unique）----
mapping = pd.concat(
	[
		exact[['ch_code', 'sta_code', STA_NUM_COL, 'match_type', 'km']],
		ch2sta_unique[['ch_code', 'sta_code', STA_NUM_COL, 'match_type', 'km']],
	],
	ignore_index=True,
).drop_duplicates(subset=['ch_code'], keep='first')

mapping.to_csv(OUT_DIR / 'mapping_ch_to_sta.csv', index=False)

# 同じ sta_code に複数 ch_code がぶら下がるもの（確認用）
coll = mapping.groupby('sta_code', as_index=False).agg(
	n_ch=('ch_code', 'count'),
	ch_codes=('ch_code', lambda s: '|'.join(sorted(s.astype(str).tolist()))),
)
coll = coll[coll['n_ch'] > 1].sort_values(['n_ch', 'sta_code'], ascending=[False, True])
coll.to_csv(OUT_DIR / 'mapping_collisions_sta_to_many_ch.csv', index=False)

print('\n=== mapping summary (ch -> sta) ===')
print('mapped ch codes:', len(mapping))
print('unmapped ch codes:', len(ch_codes) - len(mapping))
print('collisions (sta_code with multiple ch_code):', len(coll))
print(f'\nWrote CSVs to: {OUT_DIR.resolve()}')
