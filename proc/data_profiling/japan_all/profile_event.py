# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_events(csv_path: str | Path) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	df['origin_time'] = pd.to_datetime(df['origin_time'])
	return df


def plot_station_count_vs_hypocenter_flag(
	df: pd.DataFrame,
	max_y: float | None = None,
	clip_quantile: float | None = 0.99,
	show_fliers: bool = False,
) -> plt.Figure:
	"""station_count と hypocenter_flag の関係の可視化（箱ひげ図）
	・max_y: y軸の上限を直接指定したい場合に使う
	・clip_quantile: max_y が未指定のとき、全体のこのパーセンタイルを y 上限にする
	  例）0.99 → station_count の 99% 点で上限を切る
	・show_fliers: 外れ値の点を描画するかどうか
	"""
	subset = df[['station_count', 'hypocenter_flag']].dropna()

	if max_y is None and clip_quantile is not None:
		max_y = float(subset['station_count'].quantile(clip_quantile))

	grouped = subset.groupby('hypocenter_flag')
	data = [g['station_count'].to_numpy() for _, g in grouped]
	labels = list(grouped.groups.keys())

	fig, ax = plt.subplots()
	ax.boxplot(data, labels=labels, showfliers=show_fliers)

	ax.set_xlabel('hypocenter_flag')
	ax.set_ylabel('station_count')
	ax.set_title('station_count vs hypocenter_flag')

	if max_y is not None:
		ax.set_ylim(0, max_y)

	fig.tight_layout()
	return fig


def plot_monthly_event_hist(df: pd.DataFrame) -> plt.Figure:
	"""各「年月(YYYY-MM)」ごとの地震数を棒グラフで可視化する。
	20年分なら 20×12 本の棒が並ぶイメージ。
	"""
	months = df['origin_time'].dt.to_period('M')
	counts = months.value_counts().sort_index()

	x = np.arange(len(counts))

	fig, ax = plt.subplots(figsize=(12, 6))
	ax.bar(x, counts.to_numpy(), width=0.9)

	ax.set_ylabel('Number of events')
	ax.set_title('Number of earthquakes per month')

	labels = counts.index.astype(str)  # 'YYYY-MM'
	step = max(1, len(labels) // 12)  # だいたい 12 個程度ラベルを表示
	ax.set_xticks(x[::step])
	ax.set_xticklabels(labels[::step], rotation=45, ha='right')

	fig.tight_layout()
	return fig


if __name__ == '__main__':
	# ▼ここを20年分のCSVファイルパスに変更
	csv_path = '/workspace/data/arrivetime/arrivetime_epicenters.csv'

	df = load_events(csv_path)

	# 1. 各月に発生した地震数のヒストグラム
	fig_month = plot_monthly_event_hist(df)
	plt.show()

	# 2. station_count と hypocenter_flag の関係
	fig_station = plot_station_count_vs_hypocenter_flag(df)
	plt.show()

	# 3. 任意の期間 & mag1_type を指定した mag1 ヒストグラム
	#   例: 2002-01-01〜2002-12-31 のうち mag1_type="Mj" だけ
	fig_mag = plot_mag1_hist(
		df,
		start='2002-06-01',
		end='2002-07-01',
		mag1_type=None,  # ここを "Mj" などに変えるとタイプを絞れる
		# bins=np.arange(0.0, 7.1, 0.1),  # ビン幅を変える場合は指定
	)
	plt.show()
