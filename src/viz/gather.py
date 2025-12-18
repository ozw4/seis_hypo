from __future__ import annotations

import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import detrend as sp_detrend
from scipy.signal.windows import tukey

from common.geo import compute_station_order


# ---- ギャザー描画（塗りつぶしwiggle）----
def plot_gather(
	data: np.ndarray,
	station_df: pd.DataFrame | None = None,
	scaling: str = 'zscore',
	amp: float = 4.0,
	title: str | None = None,
	p_idx: np.ndarray | None = None,
	s_idx: np.ndarray | None = None,
	order_mode: str = 'pca',
	azimuth_deg: float | None = None,
	ax: plt.Axes | None = None,
	decim: int = 1,
	detrend: str | None = None,  # 'constant' | 'linear' | None
	taper_frac: float = 0.02,  # 端部ターパー（片側比）
	y_time: str = 'samples',  # ← 'samples' | 'absolute' | 'relative'
	fs: float | None = None,  # ← y_time≠'samples'なら必須(Hz)
	t_start: dt.datetime | None = None,  # ← 窓の開始(絶対時刻, JST想定)
	event_time: dt.datetime | None = None,  # ← 相対表示で0にしたい時刻
):
	assert data.ndim == 2
	n_ch, n_t = data.shape

	# --- 並び替え ---
	if station_df is not None and {'station', 'lat', 'lon'}.issubset(
		station_df.columns
	):
		order = compute_station_order(
			station_df.iloc[:n_ch], mode=order_mode, azimuth_deg=azimuth_deg
		)
		data = data[order]
		station_df = station_df.iloc[order].reset_index(drop=True)
		if p_idx is not None:
			p_idx = np.asarray(p_idx)[order]
		if s_idx is not None:
			s_idx = np.asarray(s_idx)[order]

	n_ch, n_t = data.shape  # 念のため更新
	x = data.astype(float, copy=False)

	# --- Detrend → Taper ---
	if detrend in ('constant', 'linear'):
		x = sp_detrend(x, axis=1, type=detrend)
	elif detrend is not None:
		raise ValueError("detrend must be 'constant'|'linear'|None")

	if taper_frac > 0.0:
		w = tukey(n_t, alpha=2 * taper_frac)
		x *= w

	# --- 正規化 ---
	if scaling == 'zscore':
		m = x.mean(axis=1, keepdims=True)
		s = x.std(axis=1, keepdims=True) + 1e-12
		x = (x - m) / s
	elif scaling == 'max':
		m = np.max(np.abs(x), axis=1, keepdims=True) + 1e-12
		x = x / m
	elif scaling == 'none':
		pass
	else:
		raise ValueError(f'unknown scaling: {scaling}')

	# --- 間引き（描画用） ---
	if decim > 1:
		x = x[:, ::decim]
	n_t_dec = x.shape[1]
	y = np.arange(n_t_dec) * decim  # 縦軸は元のサンプル番号で揃える

	# --- 横配置（ampは横振れ幅スケール） ---
	x_for_layout = (
		x
		if scaling == 'none'
		else x / (np.max(np.abs(x), axis=1, keepdims=True) + 1e-12)
	)
	centers = np.arange(n_ch)[:, None]
	xs = centers + amp * x_for_layout

	# --- 描画：塗りつぶしwiggle（正側だけ塗る） ---
	if ax is None:
		fig, ax = plt.subplots(figsize=(max(8.0, 0.12 * n_ch), 8))

	for i in range(n_ch):
		xi = xs[i]
		base = float(i)
		ax.plot(xi, y, lw=0.5, c='k', zorder=2)  # 輪郭線
		ax.fill_betweenx(
			y, base, xi, where=(xi >= base), linewidth=0, alpha=0.7, zorder=1, color='k'
		)

	# 軸と範囲
	ax.set_xlim(float(xs.min()) - 0.1, float(xs.max()) + 0.1)
	ax.set_ylim(y[-1], y[0] if len(y) > 0 else 0)
	ax.set_xlabel('station')
	if y_time == 'samples':
		ax.set_ylabel('sample')
	else:
		if fs is None or t_start is None:
			raise ValueError("y_time≠'samples' の場合は fs と t_start が必要です")
		total_s = (n_t_dec - 1) * (decim / fs)
		# 目盛り間隔（だいたい5〜8本出るように）
		cand = np.array([0.5, 1, 2, 5, 10, 20, 30], float)
		step_s = cand[np.argmin(np.abs(total_s / cand - 6))]
		t_grid = np.arange(0.0, total_s + 1e-9, step_s)
		yticks = (t_grid * fs / decim).astype(int)
		if y_time == 'absolute':
			labels = [
				(t_start + dt.timedelta(seconds=float(ts))).strftime('%H:%M:%S.%f')[:-3]
				for ts in t_grid
			]
			ax.set_ylabel('time (JST)')
		elif y_time == 'relative':
			if event_time is None:
				raise ValueError("y_time='relative' には event_time が必要です")
			offset = (event_time - t_start).total_seconds()
			labels = [f'{(ts - offset):+0.1f}s' for ts in t_grid]
			ax.set_ylabel('time from event')
		else:
			raise ValueError("y_time は 'samples' | 'absolute' | 'relative'")
		ax.set_yticks(yticks)
		ax.set_yticklabels(labels)
	ax.set_xticks(np.arange(n_ch))
	if station_df is not None and 'station' in station_df.columns:
		ax.set_xticklabels(station_df['station'].to_numpy()[:n_ch], rotation=90)
	else:
		ax.set_xticklabels([str(i) for i in range(n_ch)])
	if title:
		ax.set_title(title)

	# Picks
	if p_idx is not None:
		m = np.isfinite(p_idx)
		ax.scatter(
			np.arange(n_ch)[m],
			np.asarray(p_idx)[m],
			s=50,
			marker='_',
			c='b',
			linewidths=5,
			zorder=3,
		)
	if s_idx is not None:
		m = np.isfinite(s_idx)
		ax.scatter(
			np.arange(n_ch)[m],
			np.asarray(s_idx)[m],
			s=50,
			marker='_',
			c='r',
			linewidths=5,
			zorder=3,
		)

	plt.tight_layout()
	return ax
