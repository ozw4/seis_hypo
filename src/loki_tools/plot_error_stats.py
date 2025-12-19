# %%
#!/usr/bin/env python3
# proc/loki_hypo/run_plot_error_stats.py
#
# compare_df の dh_km / dz_km を
# - ヒストグラム
# - 箱ひげ
# - coherence(cmax) のビン別箱ひげ
# で可視化し、外れ値イベントもCSVに吐く。

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def iqr_outliers(df: pd.DataFrame, col: str, *, k: float = 1.5) -> pd.DataFrame:
	x = df[col].astype(float)
	q1 = float(x.quantile(0.25))
	q3 = float(x.quantile(0.75))
	iqr = q3 - q1
	lo = q1 - k * iqr
	hi = q3 + k * iqr
	out = df[(x < lo) | (x > hi)].copy()
	out[f'{col}_lo'] = lo
	out[f'{col}_hi'] = hi
	return out


def make_coherence_bins(
	df: pd.DataFrame,
	coh_col: str,
	*,
	n_bins: int = 4,
) -> pd.Series:
	"""coherence(cmax等) を分位ビンに分ける。
	- 件数が少ない(1件など)/有効値が少ない場合は 'all' の単一ビンを返す
	- qcut は境界重複で落ちることがあるため duplicates='drop' を使い、labels は後付けする
	- coh_col が NaN の行は 'missing' を返す
	"""
	if df.empty:
		return pd.Series(dtype=str, index=df.index)

	coh = df[coh_col].astype(float)
	valid = coh.notna()
	n_valid = int(valid.sum())

	# 1件(または有効値1件)では分位が作れないので単一ビン
	if n_valid < 2 or n_bins < 2:
		return pd.Series(['all'] * len(df), index=df.index, dtype=str)

	# 同値が多くても単調増加にして分位を切れるようにする
	r = coh.rank(method='first')

	q = min(int(n_bins), n_valid)
	if q < 2:
		return pd.Series(['all'] * len(df), index=df.index, dtype=str)

	cut = pd.qcut(r[valid], q=q, duplicates='drop')  # labelsは後で付ける
	cat = pd.Categorical(cut)
	m = len(cat.categories)

	# qcutが落ちないようにしているが、念のためビンができなければ単一ビン
	if m < 1:
		return pd.Series(['all'] * len(df), index=df.index, dtype=str)

	labels = [f'Q{i + 1}' for i in range(m)]

	out = pd.Series(['missing'] * len(df), index=df.index, dtype=str)
	out.loc[valid] = [labels[i] for i in cat.codes]
	return out


def plot_hist(
	values: np.ndarray, *, title: str, xlabel: str, out_png: Path, bins: int = 20
) -> None:
	fig = plt.figure(figsize=(8, 5))
	ax = fig.add_subplot(111)
	ax.hist(values, bins=bins)
	ax.set_title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel('count')
	fig.tight_layout()
	out_png.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_png, dpi=200)
	plt.close(fig)


def plot_box(
	values: list[np.ndarray],
	labels: list[str],
	*,
	title: str,
	ylabel: str,
	out_png: Path,
) -> None:
	fig = plt.figure(figsize=(9, 5))
	ax = fig.add_subplot(111)
	ax.boxplot(values, labels=labels, showfliers=True)
	ax.set_title(title)
	ax.set_ylabel(ylabel)
	fig.tight_layout()
	out_png.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_png, dpi=200)
	plt.close(fig)


def pick_mag_column(df: pd.DataFrame) -> str:
	cands = [
		'mag1_jma',
		'mag_jma',
		'mag1',
		'magnitude',
		'mag',
	]
	for c in cands:
		if c in df.columns:
			return c
	raise ValueError(f'no JMA magnitude column found in compare_df. tried={cands}')


def make_mag_bins(mag: pd.Series) -> tuple[pd.Series, list[str]]:
	# 固定の 0.5 刻み（小規模地震向け）
	m = mag.astype(float).to_numpy()
	mmin = float(np.nanmin(m))
	mmax = float(np.nanmax(m))
	if not np.isfinite(mmin) or not np.isfinite(mmax):
		raise ValueError('magnitude contains no finite values')

	lo = np.floor(mmin * 2.0) / 2.0
	hi = np.ceil(mmax * 2.0) / 2.0
	if hi <= lo:
		hi = lo + 0.5

	edges = np.arange(lo, hi + 0.5, 0.5)
	if len(edges) < 3:
		edges = np.array([lo, lo + 0.5, lo + 1.0], dtype=float)

	cut = pd.cut(mag.astype(float), bins=edges, right=False, include_lowest=True)
	labels = [str(iv) for iv in cut.cat.categories]
	return cut.astype(str), labels
