# %%
# proc/locate/forge/run_gamma_forge.py
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from gamma.utils import association, estimate_eps

from common.core import validate_columns
# =========================
# 設定（ここだけ触ればOK）
# =========================
PICKS_CSV = Path('/workspace/proc/prepare_data/forge/forge_gamma_picks.csv')
STATIONS_CSV = Path('/workspace/data/station/forge/forge_das_station_metadata.csv')
VEL_MODEL_JSON = Path('/workspace/proc/prepare_data/forge/gamma_vel_model44.json')
OUT_DIR = Path('results/gamma_forge')

METHOD = 'BGMM'  # "BGMM" or "GMM"
USE_DBSCAN = True

# ---- DBSCAN / eps周り（ここが今回の主題） ----
# dbscan_eps の基準値（秒）。Noneにすると estimate_eps() で推定（stationsとvp依存）
DBSCAN_EPS_SEC: float | None = 15.0

# 推定epsのときに使うパラメータ（estimate_eps(stations, vp, sigma=...) の sigma）
DBSCAN_EPS_SIGMA = 2.0

# 固定eps/推定epsのどちらにも掛ける倍率（少し強め/弱めたい時に便利）
DBSCAN_EPS_MULT = 1.0

# DBSCANコア点の密度条件
DBSCAN_MIN_SAMPLES = 3

# hierarchical_dbscan_clustering() の分割対象閾値（DASなら上げると軽くなる）
DBSCAN_MIN_CLUSTER_SIZE = 2000

# “時間的に長すぎるクラスタだけ割る”判定（大きいほど割りにくい＝分割が減る）
DBSCAN_MAX_TIME_SPACE_RATIO = 10.0

# NOTE:
# gamma.utils 内部では ratio/=1.2 を繰り返して eps_eff = dbscan_eps * ratio を段階的に下げます。
# なので「最小epsを直接指定」はこのAPIではできず、分割されなくなったところで止まる仕様です。

USE_AMPLITUDE = False
OVERSAMPLE_FACTOR_BGMM = 5

# 速度モデル
USE_EIKONAL_1D = True
EIKONAL_H_KM = 1.0

# 探索範囲（km）
XY_MARGIN_KM = 2.0
Z_RANGE_KM = (0.0, 5.0)

# 後段フィルタ
MIN_PICKS_PER_EQ = 100
MIN_P_PICKS_PER_EQ = 0
MIN_S_PICKS_PER_EQ = 0
MAX_SIGMA11_SEC = 3.0
MAX_SIGMA22_LOG10_MS = 1.0
MAX_SIGMA12_COV = 1.0

# CPU
NCPU = max(1, (os.cpu_count() or 1) - 1)


def _normalize_picks(picks_raw: pd.DataFrame) -> pd.DataFrame:
	picks = picks_raw.copy()

	if 'phase_time' in picks.columns:
		picks['phase_time'] = pd.to_datetime(
			picks['phase_time'], utc=True, errors='raise'
		)
		rename_map = {
			'station_id': 'id',
			'phase_time': 'timestamp',
			'phase_type': 'type',
			'phase_score': 'prob',
			'phase_amplitude': 'amp',
		}
		for k, v in rename_map.items():
			if k in picks.columns:
				picks.rename(columns={k: v}, inplace=True)

	validate_columns(picks, ['id', 'timestamp', 'type'], 'picks')

	if 'prob' not in picks.columns:
		picks['prob'] = 1.0
	if 'amp' not in picks.columns:
		picks['amp'] = -1.0

	picks['type'] = picks['type'].astype(str).str.upper()
	picks = picks[picks['type'].isin(['P', 'S'])].reset_index(drop=True)

	return picks[['id', 'timestamp', 'type', 'prob', 'amp']]


def _normalize_stations(sta_raw: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
	sta = sta_raw.copy()

	if 'station_id' in sta.columns:
		sta.rename(columns={'station_id': 'id'}, inplace=True)

	if 'x(km)' not in sta.columns:
		if 'x_km' in sta.columns:
			sta['x(km)'] = sta['x_km'].astype(float)
		else:
			raise ValueError('stations needs x(km) or x_km')
	if 'y(km)' not in sta.columns:
		if 'y_km' in sta.columns:
			sta['y(km)'] = sta['y_km'].astype(float)
		else:
			raise ValueError('stations needs y(km) or y_km')
	if 'z(km)' not in sta.columns:
		if 'z_depth_km' in sta.columns:
			sta['z(km)'] = sta['z_depth_km'].astype(float)
		else:
			raise ValueError('stations needs z(km) or z_depth_km')

	validate_columns(sta, ['id', 'x(km)', 'y(km)', 'z(km)'], 'stations')

	if 'E_m' in sta.columns and 'N_m' in sta.columns:
		origin_E_m = float(
			np.median(
				sta['E_m'].to_numpy(dtype=float)
				- sta['x(km)'].to_numpy(dtype=float) * 1000.0
			)
		)
		origin_N_m = float(
			np.median(
				sta['N_m'].to_numpy(dtype=float)
				- sta['y(km)'].to_numpy(dtype=float) * 1000.0
			)
		)
	else:
		origin_E_m = float('nan')
		origin_N_m = float('nan')

	sta = (
		sta[['id', 'x(km)', 'y(km)', 'z(km)']]
		.drop_duplicates('id')
		.reset_index(drop=True)
	)
	return sta, origin_E_m, origin_N_m


def _load_vel_json(path: Path) -> dict:
	vel = json.loads(path.read_text(encoding='utf-8'))
	for k in ['z', 'p', 's']:
		if k not in vel:
			raise ValueError(f"velocity json must have key '{k}'")
	return vel


def _build_config(stations: pd.DataFrame) -> dict:
	x_min = float(stations['x(km)'].min()) - float(XY_MARGIN_KM)
	x_max = float(stations['x(km)'].max()) + float(XY_MARGIN_KM)
	y_min = float(stations['y(km)'].min()) - float(XY_MARGIN_KM)
	y_max = float(stations['y(km)'].max()) + float(XY_MARGIN_KM)

	cfg: dict = {}
	cfg['use_dbscan'] = bool(USE_DBSCAN)
	cfg['use_amplitude'] = bool(USE_AMPLITUDE)
	cfg['method'] = str(METHOD)
	cfg['oversample_factor'] = (
		int(OVERSAMPLE_FACTOR_BGMM) if cfg['method'] == 'BGMM' else 1
	)

	cfg['dims'] = ['x(km)', 'y(km)', 'z(km)']
	cfg['x(km)'] = (x_min, x_max)
	cfg['y(km)'] = (y_min, y_max)
	cfg['z(km)'] = (float(Z_RANGE_KM[0]), float(Z_RANGE_KM[1]))

	cfg['bfgs_bounds'] = (
		(x_min, x_max),
		(y_min, y_max),
		(cfg['z(km)'][0], cfg['z(km)'][1]),
		(None, None),
	)

	# ---- velocity / eikonal first (vp is needed for estimate_eps) ----
	if USE_EIKONAL_1D:
		if not VEL_MODEL_JSON.exists():
			raise FileNotFoundError(f'VEL_MODEL_JSON not found: {VEL_MODEL_JSON}')
		vel = _load_vel_json(VEL_MODEL_JSON)
		vp0 = float(np.asarray(vel['p'], dtype=float)[0])
		vs0 = float(np.asarray(vel['s'], dtype=float)[0])
		cfg['vel'] = {'p': vp0, 's': vs0}
		cfg['eikonal'] = {
			'vel': vel,
			'h': float(EIKONAL_H_KM),
			'xlim': cfg['x(km)'],
			'ylim': cfg['y(km)'],
			'zlim': cfg['z(km)'],
		}
	else:
		cfg['vel'] = {'p': 6.0, 's': 6.0 / 1.75}
		cfg['eikonal'] = None

	# ---- DBSCAN params ----
	if cfg['use_dbscan']:
		vp_for_eps = float(cfg['vel']['p'])
		if DBSCAN_EPS_SEC is None:
			eps0 = float(
				estimate_eps(stations, vp_for_eps, sigma=float(DBSCAN_EPS_SIGMA))
			)
		else:
			eps0 = float(DBSCAN_EPS_SEC)

		eps0 *= float(DBSCAN_EPS_MULT)
		if not np.isfinite(eps0) or eps0 <= 0.0:
			raise ValueError(f'Bad dbscan_eps after scaling: {eps0}')

		cfg['dbscan_eps'] = eps0
		cfg['dbscan_min_samples'] = int(DBSCAN_MIN_SAMPLES)
		cfg['dbscan_min_cluster_size'] = int(DBSCAN_MIN_CLUSTER_SIZE)
		cfg['dbscan_max_time_space_ratio'] = float(DBSCAN_MAX_TIME_SPACE_RATIO)

	cfg['ncpu'] = int(NCPU)

	cfg['min_picks_per_eq'] = int(MIN_PICKS_PER_EQ)
	cfg['min_p_picks_per_eq'] = int(MIN_P_PICKS_PER_EQ)
	cfg['min_s_picks_per_eq'] = int(MIN_S_PICKS_PER_EQ)
	cfg['max_sigma11'] = float(MAX_SIGMA11_SEC)
	cfg['max_sigma22'] = float(MAX_SIGMA22_LOG10_MS)
	cfg['max_sigma12'] = float(MAX_SIGMA12_COV)

	return cfg


def main() -> None:
	if not PICKS_CSV.exists():
		raise FileNotFoundError(f'PICKS_CSV not found: {PICKS_CSV}')
	if not STATIONS_CSV.exists():
		raise FileNotFoundError(f'STATIONS_CSV not found: {STATIONS_CSV}')

	OUT_DIR.mkdir(parents=True, exist_ok=True)

	picks_raw = pd.read_csv(PICKS_CSV)
	stations_raw = pd.read_csv(STATIONS_CSV)

	picks = _normalize_picks(picks_raw)
	stations, origin_E_m, origin_N_m = _normalize_stations(stations_raw)

	picks = picks[picks['id'].isin(set(stations['id']))].reset_index(drop=True)
	if USE_AMPLITUDE:
		picks = picks[picks['amp'] != -1].reset_index(drop=True)

	config = _build_config(stations)
	(OUT_DIR / 'gamma_config.json').write_text(
		json.dumps(config, indent=2), encoding='utf-8'
	)

	event_idx0 = 0
	events, assignments = association(
		picks, stations, config, event_idx0, config['method']
	)

	events_df = pd.DataFrame(events)
	if len(events_df) == 0:
		print('GaMMA returned 0 events.')
		(OUT_DIR / 'gamma_events.csv').write_text('', encoding='utf-8')
	else:
		if not np.isnan(origin_E_m) and not np.isnan(origin_N_m):
			events_df['E_m'] = (
				origin_E_m + events_df['x(km)'].to_numpy(dtype=float) * 1000.0
			)
			events_df['N_m'] = (
				origin_N_m + events_df['y(km)'].to_numpy(dtype=float) * 1000.0
			)
		events_df.to_csv(
			OUT_DIR / 'gamma_events.csv',
			index=False,
			float_format='%.6f',
			date_format='%Y-%m-%dT%H:%M:%S.%f',
		)

	assign_df = pd.DataFrame(
		assignments, columns=['pick_index', 'event_index', 'gamma_score']
	)
	picks_out = picks.copy()
	picks_out = picks_out.join(assign_df.set_index('pick_index')).fillna(-1)
	picks_out['event_index'] = picks_out['event_index'].astype(int)

	picks_out.rename(
		columns={
			'id': 'station_id',
			'timestamp': 'phase_time',
			'type': 'phase_type',
			'prob': 'phase_score',
			'amp': 'phase_amplitude',
		},
		inplace=True,
	)

	picks_out.to_csv(
		OUT_DIR / 'gamma_picks.csv',
		index=False,
		date_format='%Y-%m-%dT%H:%M:%S.%f',
	)

	n_assigned = int((picks_out['event_index'] >= 0).sum())
	print(f'stations: {len(stations)}')
	print(f'picks: {len(picks_out)} (assigned: {n_assigned})')
	print(f'events: {len(events_df)}')
	print(f'wrote: {OUT_DIR / "gamma_events.csv"}')
	print(f'wrote: {OUT_DIR / "gamma_picks.csv"}')
	print(f'wrote: {OUT_DIR / "gamma_config.json"}')
	if USE_DBSCAN:
		print(
			'dbscan:',
			f'eps0={config.get("dbscan_eps"):.3f}s, '
			f'min_samples={config.get("dbscan_min_samples")}, '
			f'min_cluster_size={config.get("dbscan_min_cluster_size")}, '
			f'max_time_space_ratio={config.get("dbscan_max_time_space_ratio")}',
		)


if __name__ == '__main__':
	main()
