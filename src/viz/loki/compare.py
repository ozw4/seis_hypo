from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _finite_1d(x: np.ndarray) -> np.ndarray:
	a = np.asarray(x, dtype=float).ravel()
	return a[np.isfinite(a)]


def plot_hist_overlay(
	series: list[np.ndarray],
	labels: list[str],
	*,
	title: str,
	xlabel: str,
	out_png: Path,
	bins: int,
	density: bool = True,
) -> Path:
	if len(series) == 0:
		raise ValueError('series must be non-empty')
	if len(series) != len(labels):
		raise ValueError('len(series) must match len(labels)')

	series_f = [_finite_1d(s) for s in series]
	all_vals = np.concatenate([s for s in series_f if s.size > 0], axis=0)
	if all_vals.size == 0:
		raise RuntimeError('no finite values to plot')

	edges = np.histogram_bin_edges(all_vals, bins=int(bins))

	fig = plt.figure(figsize=(9, 4.8))
	ax = fig.add_subplot(111)

	for s, lbl in zip(series_f, labels, strict=True):
		if s.size == 0:
			continue
		ax.hist(
			s,
			bins=edges,
			histtype='step',
			density=bool(density),
			label=f'{lbl} (n={s.size})',
			linewidth=2.0,
		)

	ax.set_title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel('Density' if density else 'Count')
	ax.legend()
	fig.tight_layout()

	out_png = Path(out_png)
	out_png.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_png, dpi=200)
	plt.close(fig)
	return out_png


def save_scatter(
	x: np.ndarray,
	y: np.ndarray,
	*,
	title: str,
	xlabel: str,
	ylabel: str,
	out_png: Path,
	dpi: int = 200,
) -> Path:
	x = np.asarray(x, dtype=float).ravel()
	y = np.asarray(y, dtype=float).ravel()
	if x.size != y.size:
		raise ValueError('x and y length mismatch')

	mask = np.isfinite(x) & np.isfinite(y)
	x2 = x[mask]
	y2 = y[mask]

	out_png = Path(out_png)
	out_png.parent.mkdir(parents=True, exist_ok=True)

	fig = plt.figure(figsize=(9, 4.8))
	ax = fig.add_subplot(111)
	ax.scatter(x2, y2)
	ax.set_title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	fig.tight_layout()
	fig.savefig(out_png, dpi=int(dpi))
	plt.close(fig)
	return out_png
