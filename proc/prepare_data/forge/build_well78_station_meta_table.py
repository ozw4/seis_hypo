# %%
from pathlib import Path

import numpy as np
import pandas as pd

# ====== パラメータ（ここだけ編集すればOK） ======
# Tap-testで決まった「井口(深度0)」と「深部端(深度max)」のキャリブレーション（単位: m）
W78A = dict(
	well='78A-32',
	E_m=335780.84,
	N_m=4262991.99,
	elev_head_m=1701.92,
	depth_bottom_m=989.90,  # 深部端の深度
	ch_shallow=1080,  # 深度0(井口)のチャンネル
	ch_deep=70,  # 深部端(深い側)のチャンネル
)

W78B = dict(
	well='78B-32',
	E_m=335865.45,
	N_m=4262983.53,
	elev_head_m=1705.62,
	depth_bottom_m=1193.42,
	ch_shallow=1196,  # 深度0(井口)のチャンネル
	ch_deep=2400,  # 深部端(深い側)のチャンネル（あなたの確定）
)

# ローカル座標の原点（おすすめ: 78Bの井口）
ORIGIN_E_m = W78B['E_m']
ORIGIN_N_m = W78B['N_m']

# 出力
OUT_DIR = Path()
OUT_CSV = OUT_DIR / 'forge_das_station_metadata.csv'
OUT_MEMO = OUT_DIR / 'forge_das_station_metadata_README.md'
# ===============================================


def make_well_df(cfg: dict, origin_E_m: float, origin_N_m: float) -> pd.DataFrame:
	ch0 = int(cfg['ch_shallow'])
	ch1 = int(cfg['ch_deep'])
	depth1 = float(cfg['depth_bottom_m'])

	ch_min = min(ch0, ch1)
	ch_max = max(ch0, ch1)
	ch = np.arange(ch_min, ch_max + 1, dtype=int)

	# 深度0 at ch_shallow, 深度depth_bottom at ch_deep を仮定した線形内挿
	# depth(ch) = (ch - ch_shallow) * depth_bottom / (ch_deep - ch_shallow)
	depth_m = (ch - ch0) * (depth1 / (ch1 - ch0))
	elev_m = float(cfg['elev_head_m']) - depth_m

	E_m = np.full(ch.shape, float(cfg['E_m']), dtype=float)
	N_m = np.full(ch.shape, float(cfg['N_m']), dtype=float)

	# GaMMA向けに km 化（ローカル）
	x_km = (E_m - origin_E_m) / 1000.0
	y_km = (N_m - origin_N_m) / 1000.0

	# z は2種類出しておく（どっちを使うかはGaMMA設定に合わせて選ぶ）
	z_depth_km = depth_m / 1000.0  # 深さ[km] 下向き正（1D速度モデルのzと相性良い）
	z_elev_km = -elev_m / 1000.0  # -標高[km]（GaMMA例でよく見る）

	# あなたの波形配列 index への対応をそのまま入れる
	if cfg['well'].startswith('78A'):
		# index[0] -> ch 70 ; index[1010] -> ch 1080
		index = ch - 70
		station_id = [f'DAS78A_CH{c:04d}' for c in ch]
	else:
		# index[1011] -> ch 1196 ; index[2215] -> ch 2400 （N=2216想定）
		index = (ch - 1196) + 1011
		station_id = [f'DAS78B_CH{c:04d}' for c in ch]

	df = (
		pd.DataFrame(
			{
				'station_id': station_id,
				'well': cfg['well'],
				'channel': ch,
				'index': index.astype(int),
				'E_m': E_m,
				'N_m': N_m,
				'elev_m': elev_m.astype(float),
				'depth_m': depth_m.astype(float),
				'x_km': x_km.astype(float),
				'y_km': y_km.astype(float),
				'z_depth_km': z_depth_km.astype(float),
				'z_elev_km': z_elev_km.astype(float),
			}
		)
		.sort_values('channel')
		.reset_index(drop=True)
	)

	return df


df_a = make_well_df(W78A, ORIGIN_E_m, ORIGIN_N_m)
df_b = make_well_df(W78B, ORIGIN_E_m, ORIGIN_N_m)

df = (
	pd.concat([df_a, df_b], ignore_index=True)
	.sort_values(['well', 'channel'])
	.reset_index(drop=True)
)

df.to_csv(OUT_CSV, index=False)

memo = f"""# FORGE DAS station metadata (78A-32 / 78B-32)

## What this is
A GaMMA-ready station table created from tap-test calibrated endpoints (Tables 1 & 2).
Assumptions:
- Wells are treated as **vertical**
- Receiver positions are assigned by **linear interpolation** between wellhead and well-bottom endpoints

## Horizontal coordinates (meters)
- 78A-32: E=335780.84, N=4262991.99
- 78B-32: E=335865.45, N=4262983.53
For each well, E/N are constant for all channels (vertical-well approximation).

## Local coordinates (km)
Origin is set to the 78B wellhead:
- E0={ORIGIN_E_m} m, N0={ORIGIN_N_m} m
x_km = (E - E0) / 1000
y_km = (N - N0) / 1000

## Channel → depth (linear interpolation)

### 78A-32 (deep -> surface)
Endpoints:
- depth=0.00 m at ch=1080, elev=1701.92 m
- depth=989.90 m at ch=70,   elev=712.02 m
Formulas:
- depth_m(ch) = (ch - 1080) * (989.90 / (70 - 1080))
- elev_m(ch)  = 1701.92 - depth_m(ch)

### 78B-32 (surface -> deep)
Endpoints:
- depth=0.00 m at ch=1196, elev=1705.62 m
- depth=1193.42 m at ch=2400, elev=511.58 m
Formulas:
- depth_m(ch) = (ch - 1196) * (1193.42 / (2400 - 1196))
- elev_m(ch)  = 1705.62 - depth_m(ch)

## z conventions included
- z_depth_km = depth_m / 1000   (depth positive downward)
- z_elev_km  = -elev_m / 1000   (negative elevation)

Use one consistently in GaMMA.

## Array index mapping included
Per your mapping:
- 78A: index = channel - 70
- 78B: index = (channel - 1196) + 1011

## Output files
- {OUT_CSV.name}
- {OUT_MEMO.name}
"""
OUT_MEMO.write_text(memo, encoding='utf-8')

print(f'Wrote: {OUT_CSV.resolve()}')
print(f'Wrote: {OUT_MEMO.resolve()}')
print(df.head(2))
print(df[df['well'] == '78B-32'].tail(2))

# %%
