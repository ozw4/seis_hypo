# %%
from pathlib import Path

import pandas as pd

# ===== 設定（ここだけ直せばOK） =====
PICKS_CSV = Path(
	'/workspace/proc/pick_continuous/forge/out/das_eqt_picks_woconvert.csv'
)  # EQT picks（channelはindex）
STATIONS_CSV = Path(
	'forge_das_station_metadata.csv'
)  # さっき作ったstation表（index列あり）

Z_COL = 'z_depth_km'

OUT_PICKS = Path('gamma_picks_with_stations.csv')
OUT_STATIONS = Path('gamma_stations.csv')
# ====================================

picks = pd.read_csv(PICKS_CSV)
stations = pd.read_csv(STATIONS_CSV)

# picksのchannelは「物理CH」じゃなく「配列index」なので名前を明確化
picks = picks.rename(columns={'channel': 'index'})

# 必須列チェック（無ければ即エラーで落とす）
need_picks = {'index', 'phase', 'pick_time_utc_ms', 'pick_time_utc_iso', 'prob'}
need_stas = {'index', 'station_id', 'well', 'channel', 'x_km', 'y_km', Z_COL}
missing_p = need_picks - set(picks.columns)
missing_s = need_stas - set(stations.columns)
assert not missing_p, f'picksに必要列がない: {sorted(missing_p)}'
assert not missing_s, f'stationsに必要列がない: {sorted(missing_s)}'

# JOIN（indexで結合）
m = picks.merge(
	stations[['index', 'station_id', 'well', 'channel', 'x_km', 'y_km', Z_COL]],
	on='index',
	how='left',
)

# 結合できないpickがあったら即落とす（座標が無いとGaMMAで詰む）
n_bad = int(m['station_id'].isna().sum())
assert n_bad == 0, f'station_idが引けないpickが {n_bad} 行ある（index対応を確認）'

# GaMMA向けに列を整形（使いやすい名前に）
m = m.rename(
	columns={
		'channel': 'channel_phys',  # 物理チャンネル番号（70..1080, 1196..2400）
		Z_COL: 'z_km',
	}
)

# GaMMAでよく使う形：station_id / phase / timestamp / prob
# （timestampはmsとISO両方残しておく：後でどっちでも使える）
cols_front = [
	'station_id',
	'well',
	'index',
	'channel_phys',
	'phase',
	'pick_time_utc_ms',
	'pick_time_utc_iso',
	'prob',
	'x_km',
	'y_km',
	'z_km',
]
keep = [c for c in cols_front if c in m.columns] + [
	c for c in m.columns if c not in cols_front
]
m = m[keep]

m.to_csv(OUT_PICKS, index=False)

# stations側もGaMMA用に最小列で出す（必要なら列足してOK）
s = stations[['station_id', 'well', 'index', 'channel', 'x_km', 'y_km', Z_COL]].copy()
s = s.rename(columns={'channel': 'channel_phys', Z_COL: 'z_km'})
s.to_csv(OUT_STATIONS, index=False)

print(f'Wrote: {OUT_PICKS.resolve()}')
print(f'Wrote: {OUT_STATIONS.resolve()}')

# ついでに、78A/78Bのpick数を確認
print(m.groupby('well')['station_id'].count())
