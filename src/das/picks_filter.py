from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common.core import validate_columns  # 旧 util.validate_columns 相当


def filter_and_decimate_das_picks(
	df_events: pd.DataFrame,
	df_picks: pd.DataFrame,
	*,
	dt_sec: float,
	fiber_spacing_m: float = 1.0,
	channel_start: int = 200,
	win_half_samples: int = 500,
	residual_thresh_s: float = 0.05,
	spacing_m: float = 100.0,
) -> pd.DataFrame:
	"""RANSAC で求めた P 波到達カーブと DAS ピックを組み合わせてフィルタ+間引きを行う.

	処理内容
	--------
	1. events_summary の直線モデル t = a + s x (slowness/intercept) と比較し、
	|t_obs - t_pred| <= residual_thresh_s のピックのみ残す。
	2. offset_m を spacing_m ごとにビニングし、各 (event_id, bin) で
	w_conf が最大の 1 チャネルだけ残す。

	Parameters
	----------
	df_events : pd.DataFrame
		RANSAC まとめ (events_summary_*.csv 相当)。
		必須列: ['event_id', 'slowness_s_per_m', 'intercept_s']。
	df_picks : pd.DataFrame
		DAS ピック (das_picks_*.csv 相当)。
		必須列: ['event_id', 'peak_index', 'channel', 'sample_index']。
		'w_conf' が無い場合は 1.0 で埋める。
	dt_sec : float
		サンプリング間隔 [s] (例: 0.01)。
	fiber_spacing_m : float, default 1.0
		1 チャネルあたりの DAS 空間間隔 [m]。
	channel_start : int, default 200
		offset=0 m とみなすチャネル番号 (use_ch_range.start に合わせる)。
	win_half_samples : int, default 500
		検出窓のハーフ幅 [samples]。元コードの idx-500:idx+500 に対応。
	residual_thresh_s : float, default 0.05
		直線モデルからの許容残差 [s]。
	spacing_m : float, default 100.0
		この距離ごとに 1 チャネルに間引くビン幅 [m]。

	Returns
	-------
	pd.DataFrame
		フィルタ+間引き後の DAS ピック。
		入力 df_picks の列に加えて以下の列を持つ:

		- offset_m   : float, channel_start を原点としたオフセット [m]
		- t_obs_sec  : float, 窓内の観測時刻 [s]
		- t_pred_sec : float, 直線モデルからの予測時刻 [s]
		- resid_sec  : float, t_obs_sec - t_pred_sec
		- fit_bin    : int, spacing_m ごとのビン番号

	"""
	validate_columns(
		df_events,
		['event_id', 'slowness_s_per_m', 'intercept_s'],
		'df_events',
	)
	validate_columns(
		df_picks,
		['event_id', 'peak_index', 'channel', 'sample_index'],
		'df_picks',
	)

	df_picks = df_picks.copy()

	if 'w_conf' not in df_picks.columns:
		df_picks['w_conf'] = 1.0

	# --- events_summary の RANSAC パラメータ (slowness/intercept) を付与 ---
	ev_cols = ['event_id', 'slowness_s_per_m', 'intercept_s']
	df = df_picks.merge(df_events[ev_cols], on='event_id', how='inner')

	# --- チャネル番号 → オフセット[m]（channel_start を原点とする） ---
	df['offset_m'] = (df['channel'].astype(int) - int(channel_start)) * float(
		fiber_spacing_m
	)

	# --- 検出窓内の観測時刻 t_obs_sec を復元 ---
	#   窓開始サンプル= peak_index - win_half_samples
	#   t_obs_sec = (sample_index - 窓開始) * dt
	peak_idx = df['peak_index'].astype(int)
	sample_idx = df['sample_index'].astype(int)
	win_start = peak_idx - int(win_half_samples)
	df['t_obs_sec'] = (sample_idx - win_start) * float(dt_sec)

	# --- RANSAC 直線 t = a + s x からの予測と残差 ---
	s = df['slowness_s_per_m'].astype(float)
	a = df['intercept_s'].astype(float)
	x = df['offset_m'].astype(float)

	df['t_pred_sec'] = a + s * x
	df['resid_sec'] = df['t_obs_sec'] - df['t_pred_sec']

	# 1) 残差フィルタリング
	m_resid = df['resid_sec'].abs() <= float(residual_thresh_s)
	df_filt = df[m_resid].copy()
	if df_filt.empty:
		return df_filt

	# 2) spacing_m ごとにビニングし、(event_id, fit_bin) ごとに w_conf 最大 1 本に間引き
	spacing_m = float(spacing_m)
	df_filt['fit_bin'] = np.round(df_filt['offset_m'] / spacing_m).astype(int)

	grp = df_filt.groupby(['event_id', 'fit_bin'], sort=False)['w_conf']
	idx_best = grp.idxmax()

	df_out = df_filt.loc[idx_best].sort_values(
		['event_id', 'offset_m'],
		ignore_index=True,
	)

	return df_out


if __name__ == '__main__':
	events_csv = Path('events_summary_20200215_20200301.csv')
	picks_csv = Path('das_picks_20200215_20200301.csv')

	df_events = pd.read_csv(events_csv)
	df_picks = pd.read_csv(picks_csv)

	df_filtered = filter_and_decimate_das_picks(
		df_events,
		df_picks,
		dt_sec=0.01,
		fiber_spacing_m=1.0,
		channel_start=200,  # use_ch_range.start と合わせる
		win_half_samples=500,  # idx-500:idx+500 に合わせる
		residual_thresh_s=0.05,  # ±0.05s 以内を「整合的」とみなす
		spacing_m=100.0,  # 100m ごとに 1 チャネルに間引き
	)

	df_filtered.to_csv(
		'das_picks_filtered_decimated.csv',
		index=False,
	)
	print(f'filtered+decimated DAS picks: {len(df_filtered)} rows')
