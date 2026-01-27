# plot_true_pred_xy.py
# CSVに (x_true, y_true, x_pred, y_pred) 列がある前提で、
# 正解(真値)と予測を散布図に描き、対応ペアを点線で結びます。

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd


def plot_true_vs_pred_xy(
	df: pd.DataFrame,
	true_x: str = 'x_true',
	true_y: str = 'y_true',
	pred_x: str = 'x_pred',
	pred_y: str = 'y_pred',
	title: str = 'True vs Predicted (XY)',
) -> None:
	required = [true_x, true_y, pred_x, pred_y]
	missing = [c for c in required if c not in df.columns]
	if missing:
		raise ValueError(f'Missing columns: {missing}. Existing: {list(df.columns)}')

	d = df[required].copy()
	d = d.apply(pd.to_numeric, errors='raise')

	tx, ty, px, py = (
		d[true_x].to_numpy(),
		d[true_y].to_numpy(),
		d[pred_x].to_numpy(),
		d[pred_y].to_numpy(),
	)

	fig, ax = plt.subplots()
	ax.scatter(tx, ty, label='True')
	ax.scatter(px, py, label='Pred')

	# ペアを点線でリンク
	for i in range(len(d)):
		ax.plot([tx[i], px[i]], [ty[i], py[i]], linestyle='--', linewidth=1)

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_title(title)
	ax.set_aspect('equal', adjustable='box')
	ax.grid(True)
	ax.legend()
	plt.tight_layout()
	plt.show()


def main() -> None:
	p = argparse.ArgumentParser()
	p.add_argument('csv', help='Input CSV path')
	p.add_argument('--true-x', default='x_true')
	p.add_argument('--true-y', default='y_true')
	p.add_argument('--pred-x', default='x_pred')
	p.add_argument('--pred-y', default='y_pred')
	p.add_argument('--title', default='True vs Predicted (XY)')
	args = p.parse_args()

	df = pd.read_csv(args.csv)
	plot_true_vs_pred_xy(
		df,
		true_x=args.true_x,
		true_y=args.true_y,
		pred_x=args.pred_x,
		pred_y=args.pred_y,
		title=args.title,
	)


if __name__ == '__main__':
	main()
