# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ====== 設定（ここだけ触ればOK） ======
MODEL_DIR = Path('/workspace/data/velocity/forge/Vs_profiles/mod_profiles')
MODEL_NAME = '44_mod.lst'  # maybe most close to well position
VP_VS_RATIO = 1.75  # 迷ったら 1.73〜1.80 の範囲で固定（GaMMA例は 1.73/1.75 が多い）
DZ_M = 25.0  # 深さ方向の間引き（m）。25〜50mくらいが扱いやすい
OUT_CSV = 'gamma_vel_model44.csv'
OUT_JSON = 'gamma_vel_model44.json'
# =====================================


# 3列（depth_m, Vs_mps, sigma_mps想定）を読む
df = pd.read_csv(
	MODEL_DIR / MODEL_NAME,
	sep=r'\s+',
	header=None,
	names=['depth_m', 'vs_mps', 'sigma_mps'],
)

# 先頭行だけ深さが飛んでる（例: 100mが先に出る）パターンを弾く
if len(df) >= 2 and df.loc[0, 'depth_m'] > df.loc[1, 'depth_m']:
	df = df.iloc[1:].reset_index(drop=True)

# 念のため昇順に
df = df.sort_values('depth_m').reset_index(drop=True)

# 深さを等間隔にリサンプル（線形補間）
z_m = df['depth_m'].to_numpy()
vs_mps = df['vs_mps'].to_numpy()

z_grid_m = np.arange(z_m.min(), z_m.max() + 1e-9, DZ_M)
vs_grid_mps = np.interp(z_grid_m, z_m, vs_mps)

# GaMMA eikonal 用に km / km/s に変換
z_km = (z_grid_m / 1000.0).astype(float)
vs_kms = (vs_grid_mps / 1000.0).astype(float)
vp_kms = (vs_kms * VP_VS_RATIO).astype(float)

vel = {'z': z_km.tolist(), 'p': vp_kms.tolist(), 's': vs_kms.tolist()}

# 保存（確認しやすいようにCSVとJSONの両方）
pd.DataFrame({'z_km': z_km, 'vp_kms': vp_kms, 'vs_kms': vs_kms}).to_csv(
	OUT_CSV, index=False
)

with open(OUT_JSON, 'w', encoding='utf-8') as f:
	json.dump(vel, f, indent=2)

print(f'Wrote: {MODEL_DIR / OUT_CSV}')
print(f'Wrote: {MODEL_DIR / OUT_JSON}')
print('vel dict preview:', {k: vel[k][:5] for k in ['z', 'p', 's']})
