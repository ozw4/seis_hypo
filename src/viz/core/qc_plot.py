from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def require_file(path: Path, *, hint: str | None = None) -> None:
	path = Path(path)
	if not path.exists():
		msg = f'Input file not found: {path}'
		if hint:
			msg += f'\n{hint}'
		raise FileNotFoundError(msg)


def require_any_files(paths: list[Path], *, hint: str | None = None) -> None:
	if not paths:
		msg = 'No input files matched.'
		if hint:
			msg += f'\n{hint}'
		raise FileNotFoundError(msg)


def binned_stats(x: np.ndarray, y: np.ndarray, bins: np.ndarray) -> pd.DataFrame:
	"""Binned median and p16/p84 stats for y over x-bins.

	Returns a row for every bin (even empty ones) with columns:
	bin_left, bin_right, bin_center, n, median, p16, p84
	"""
	x = np.asarray(x, dtype=float)
	y = np.asarray(y, dtype=float)

	idx = np.digitize(x, bins) - 1
	ok = (idx >= 0) & (idx < len(bins) - 1) & np.isfinite(x) & np.isfinite(y)
	idx = idx[ok]
	y = y[ok]

	if y.size == 0:
		return pd.DataFrame(
			columns=['bin_left', 'bin_right', 'bin_center', 'n', 'median', 'p16', 'p84']
		)

	df = pd.DataFrame({'bin': idx, 'y': y})
	g = df.groupby('bin', sort=True)

	out = pd.DataFrame(
		{
			'bin_left': bins[:-1],
			'bin_right': bins[1:],
			'bin_center': (bins[:-1] + bins[1:]) / 2.0,
			'n': g.size().reindex(range(len(bins) - 1), fill_value=0).to_numpy(),
		}
	)

	def q(v: pd.Series, p: float) -> float:
		return float(np.quantile(v.to_numpy(dtype=float), p))

	med = g['y'].median()
	p16 = g['y'].apply(lambda v: q(v, 0.16))
	p84 = g['y'].apply(lambda v: q(v, 0.84))

	out['median'] = med.reindex(range(len(bins) - 1)).to_numpy()
	out['p16'] = p16.reindex(range(len(bins) - 1)).to_numpy()
	out['p84'] = p84.reindex(range(len(bins) - 1)).to_numpy()

	return out


def binned_mean(x: np.ndarray, y: np.ndarray, bins: np.ndarray) -> pd.DataFrame:
	"""Binned mean for y over x-bins.

	Use this for rates (0/1) such as good_0p10.
	Returns: bin_left, bin_right, bin_center, n, mean
	"""
	x = np.asarray(x, dtype=float)
	y = np.asarray(y, dtype=float)

	idx = np.digitize(x, bins) - 1
	ok = (idx >= 0) & (idx < len(bins) - 1) & np.isfinite(x) & np.isfinite(y)
	idx = idx[ok]
	y = y[ok]

	df = pd.DataFrame({'bin': idx, 'y': y})
	g = df.groupby('bin', sort=True)

	out = pd.DataFrame(
		{
			'bin_left': bins[:-1],
			'bin_right': bins[1:],
			'bin_center': (bins[:-1] + bins[1:]) / 2.0,
			'n': g.size().reindex(range(len(bins) - 1), fill_value=0).to_numpy(),
		}
	)

	out['mean'] = g['y'].mean().reindex(range(len(bins) - 1)).to_numpy()
	return out


def hexbin_with_color_mode(
	fig,
	ax,
	*,
	x: np.ndarray,
	y: np.ndarray,
	gridsize: int,
	mincnt: int,
	cmap: str,
	mode: str,
	xedges_for_xnorm: np.ndarray | None = None,
	xbin_min_colsum: float = 0.0,
	vmin: float | None = None,
	vmax: float | None = None,
	colorbar_label: str | None = None,
) -> None:
	"""Matplotlib hexbin with three color modes.

	    mode:
	- 'counts'     : raw counts (linear)
	    - 'log_counts' : log10(counts)
	    - 'xnorm'      : fraction within x-bin (p(y|x-bin) style)
	"""
	mode2 = str(mode).strip().lower()
	if mode2 not in {'xnorm', 'log_counts', 'counts'}:
		raise ValueError(f'unknown mode: {mode} (use xnorm/log_counts/counts)')

	if mode2 == 'xnorm':
		if xedges_for_xnorm is None:
			raise ValueError('xedges_for_xnorm must be provided when mode=xnorm')

		hb = ax.hexbin(x, y, gridsize=int(gridsize), mincnt=int(mincnt), cmap=str(cmap))
		offsets = np.asarray(hb.get_offsets(), dtype=float)
		xc = offsets[:, 0]
		counts = np.asarray(hb.get_array(), dtype=float)
		if counts.size == 0:
			cbar = fig.colorbar(hb, ax=ax)
			cbar.set_label(colorbar_label or 'fraction within x-bin')
			return
		if np.any(counts < 0):
			raise ValueError('hexbin counts should be non-negative')

		ix = np.digitize(xc, xedges_for_xnorm) - 1
		ok = (ix >= 0) & (ix < len(xedges_for_xnorm) - 1)
		if not np.any(ok):
			raise ValueError('no hex centers fall into xedges_for_xnorm range')

		ix2 = ix[ok]
		c2 = counts[ok]
		colsum = np.bincount(
			ix2, weights=c2, minlength=len(xedges_for_xnorm) - 1
		).astype(float)
		den = colsum[ix2]
		if np.any(den <= 0):
			raise ValueError(
				'found x-bin with non-positive colsum (check mincnt/xedges)'
			)

		frac = c2 / den
		arr = np.full_like(counts, np.nan, dtype=float)
		if float(xbin_min_colsum) > 0:
			mask = den >= float(xbin_min_colsum)
			arr[ok] = np.where(mask, frac, np.nan)
		else:
			arr[ok] = frac
		hb.set_array(arr)
		hb.autoscale()
		if vmin is not None or vmax is not None:
			hb.set_clim(vmin=vmin, vmax=vmax)
		cbar = fig.colorbar(hb, ax=ax)
		cbar.set_label(colorbar_label or 'fraction within x-bin')
		return

	if mode2 == 'log_counts':
		hb = ax.hexbin(
			x, y, gridsize=int(gridsize), mincnt=int(mincnt), bins='log', cmap=str(cmap)
		)
		if vmin is not None or vmax is not None:
			hb.set_clim(vmin=vmin, vmax=vmax)
		cbar = fig.colorbar(hb, ax=ax)
		cbar.set_label(colorbar_label or 'log10(counts)')
		return

	hb = ax.hexbin(x, y, gridsize=int(gridsize), mincnt=int(mincnt), cmap=str(cmap))
	if vmin is not None or vmax is not None:
		hb.set_clim(vmin=vmin, vmax=vmax)
	cbar = fig.colorbar(hb, ax=ax)
	cbar.set_label(colorbar_label or 'counts')
