from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

DEFAULT_DPI = 200


def ensure_parent(out_png: str | Path) -> Path:
	"""出力先ディレクトリを作って Path を返す。"""
	p = Path(out_png)
	p.parent.mkdir(parents=True, exist_ok=True)
	return p


def save_figure(
	fig: Figure,
	out_png: str | Path,
	*,
	dpi: int = DEFAULT_DPI,
	bbox_inches: str | None = None,
	pad_inches: float | None = None,
	tight_layout: bool = False,
	close: bool = True,
) -> Path:
	"""Figure を保存する共通関数。

	- dpi: 既存の多くが200なのでデフォ200
	- bbox_inches: events_map系は 'tight' を渡したい（デフォNone）
	- close: close=True をデフォ（メモリリーク防止）
	- tight_layout: 既存挙動を変えないためデフォFalse（必要なときだけ True）
	"""
	out_png = ensure_parent(out_png)

	if tight_layout:
		fig.tight_layout()

	kwargs: dict[str, Any] = {'dpi': int(dpi)}
	if bbox_inches is not None:
		kwargs['bbox_inches'] = bbox_inches
	if pad_inches is not None:
		kwargs['pad_inches'] = float(pad_inches)

	fig.savefig(out_png, **kwargs)

	if close:
		plt.close(fig)

	return out_png


def save_axes_figure(
	ax: Any,
	out_png: str | Path,
	*,
	dpi: int = DEFAULT_DPI,
	bbox_inches: str | None = None,
	pad_inches: float | None = None,
	tight_layout: bool = False,
	close: bool = True,
) -> Path:
	"""Axes から fig を辿って保存する薄いラッパ。"""
	return save_figure(
		ax.figure,
		out_png,
		dpi=dpi,
		bbox_inches=bbox_inches,
		pad_inches=pad_inches,
		tight_layout=tight_layout,
		close=close,
	)


def save_current_figure(
	out_png: str | Path,
	*,
	dpi: int = DEFAULT_DPI,
	bbox_inches: str | None = None,
	pad_inches: float | None = None,
	tight_layout: bool = False,
	close: bool = True,
) -> Path:
	"""plt.* ベースの既存コード救済用（将来は save_figure に寄せる想定）。"""
	import matplotlib.pyplot as plt

	fig = plt.gcf()
	return save_figure(
		fig,
		out_png,
		dpi=dpi,
		bbox_inches=bbox_inches,
		pad_inches=pad_inches,
		tight_layout=tight_layout,
		close=close,
	)
