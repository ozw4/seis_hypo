# %%
"""Run GaMMA phase association for WIN32 (Hi-net) continuous picks."""

# proc/run_continuous/association/win32/run_gamma_win32.py
#
# NOTE:
# - Prompt1で作成した前処理スクリプトの出力
#   (build_gamma_picks_from_win32_eqt_csv.py / build_gamma_stations_from_ch.py)
#   を入力にする前提の GaMMA association 実行スクリプト。

from __future__ import annotations

import os
from pathlib import Path

from gamma_workflow.run import run_gamma_from_csvs

# =========================
# 設定（ここだけ触ればOK）
# =========================
PICKS_CSV = Path('/workspace/proc/run_continuous/association/jma/out/gamma_picks.csv')
STATIONS_CSV = Path(
	'/workspace/proc/run_continuous/association/jma/out/gamma_stations.csv'
)
VEL_MODEL_JSON = Path('/workspace/proc/prepare_data/forge/gamma_vel_model44.json')
OUT_DIR = Path('/workspace/proc/run_continuous/association/win32/out')

METHOD = 'BGMM'  # "BGMM" or "GMM"
USE_DBSCAN = True

# ---- DBSCAN / eps周り ----
# dbscan_eps の基準値（秒）。Noneにすると estimate_eps() で推定（stationsとvp依存）
DBSCAN_EPS_SEC: float | None = None

# 推定epsのときに使うパラメータ（estimate_eps(stations, vp, sigma=...) の sigma）
DBSCAN_EPS_SIGMA = 2.0

# 固定eps/推定epsのどちらにも掛ける倍率（少し強め/弱めたい時に便利）
DBSCAN_EPS_MULT = 1.0

# DBSCANコア点の密度条件
DBSCAN_MIN_SAMPLES = 3

# hierarchical_dbscan_clustering() の分割対象閾値
DBSCAN_MIN_CLUSTER_SIZE = 100

# “時間的に長すぎるクラスタだけ割る”判定（大きいほど割りにくい＝分割が減る）
DBSCAN_MAX_TIME_SPACE_RATIO = 10.0

USE_AMPLITUDE = False
OVERSAMPLE_FACTOR_BGMM = 5

# 速度モデル
USE_EIKONAL_1D = True
EIKONAL_H_KM = 1.0

# 探索範囲（km）
XY_MARGIN_KM = 2.0
Z_RANGE_KM = (0.0, 5.0)

# 後段フィルタ（WIN32向けに forge より低め）
MIN_PICKS_PER_EQ = 15
MIN_P_PICKS_PER_EQ = 0
MIN_S_PICKS_PER_EQ = 0
MAX_SIGMA11_SEC = 3.0
MAX_SIGMA22_LOG10_MS = 1.0
MAX_SIGMA12_COV = 1.0

# CPU
NCPU = max(1, (os.cpu_count() or 1) - 1)


def main() -> None:
	"""Run GaMMA with top-of-file constants and write 3 output files."""
	result = run_gamma_from_csvs(
		picks_csv=PICKS_CSV,
		stations_csv=STATIONS_CSV,
		vel_json=VEL_MODEL_JSON,
		out_dir=OUT_DIR,
		method=METHOD,
		use_dbscan=USE_DBSCAN,
		use_amplitude=USE_AMPLITUDE,
		oversample_factor_bgmm=OVERSAMPLE_FACTOR_BGMM,
		use_eikonal_1d=USE_EIKONAL_1D,
		eikonal_h_km=EIKONAL_H_KM,
		xy_margin_km=XY_MARGIN_KM,
		z_range_km=Z_RANGE_KM,
		dbscan_eps_sec=DBSCAN_EPS_SEC,
		dbscan_eps_sigma=DBSCAN_EPS_SIGMA,
		dbscan_eps_mult=DBSCAN_EPS_MULT,
		dbscan_min_samples=DBSCAN_MIN_SAMPLES,
		dbscan_min_cluster_size=DBSCAN_MIN_CLUSTER_SIZE,
		dbscan_max_time_space_ratio=DBSCAN_MAX_TIME_SPACE_RATIO,
		ncpu=NCPU,
		min_picks_per_eq=MIN_PICKS_PER_EQ,
		min_p_picks_per_eq=MIN_P_PICKS_PER_EQ,
		min_s_picks_per_eq=MIN_S_PICKS_PER_EQ,
		max_sigma11_sec=MAX_SIGMA11_SEC,
		max_sigma22_log10_ms=MAX_SIGMA22_LOG10_MS,
		max_sigma12_cov=MAX_SIGMA12_COV,
	)

	if result['events_count'] == 0:
		print('GaMMA returned 0 events.')

	print(f"stations: {result['stations_count']}")
	print(f"picks: {result['picks_count']} (assigned: {result['assigned_count']})")
	print(f"events: {result['events_count']}")
	print(f"wrote: {result['events_path']}")
	print(f"wrote: {result['picks_path']}")
	print(f"wrote: {result['config_path']}")
	if USE_DBSCAN:
		print(
			'dbscan:',
			f"eps0={result['config'].get('dbscan_eps'):.3f}s, "
			f"min_samples={result['config'].get('dbscan_min_samples')}, "
			f"min_cluster_size={result['config'].get('dbscan_min_cluster_size')}, "
			f"max_time_space_ratio={result['config'].get('dbscan_max_time_space_ratio')}",
		)


if __name__ == '__main__':
	main()

# 実行例:
# export PYTHONPATH="$PWD/src"
# python proc/run_continuous/association/win32/run_gamma_win32.py
