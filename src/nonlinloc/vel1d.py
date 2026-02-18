# %%
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---- Hard-coded paths for your workflow ----
DEFAULT_1DVEL_SRC = Path('/workspace/data/velocity/jma/vjma2001')
DEFAULT_NLL_LAYERS_OUT = Path('/workspace/data/velocity/jma_tt_table/jma2001.layers')


@dataclass(frozen=True, order=True)
class Vel1DPoint:
	"""A single depth node of a 1D velocity table.

	Assumed columns:
	    Vp  Vs  Depth
	Units (assumed):
	    km/s, km/s, km
	"""

	depth_km: float
	vp_km_s: float
	vs_km_s: float


def parse_1dvel_lines(
	lines: Iterable[str],
	*,
	strict: bool = False,
) -> list[Vel1DPoint]:
	"""Parse generic 1D velocity table lines.

	Expected whitespace-separated columns:
	    Vp  Vs  Depth
	- Extra columns are ignored.
	- Blank lines are ignored.

	strict=False:
	    - Ignore non-blank lines with fewer than 3 tokens.
	strict=True:
	    - Raise ValueError on malformed non-blank lines.
	"""
	rows: list[Vel1DPoint] = []

	for idx, raw in enumerate(lines, start=1):
		ln = raw.strip()
		if not ln:
			continue

		parts = ln.split()
		if len(parts) < 3:
			if strict:
				raise ValueError(f'Line {idx} has fewer than 3 columns: {raw!r}')
			continue

		vp = float(parts[0])
		vs = float(parts[1])
		depth = float(parts[2])

		rows.append(Vel1DPoint(depth_km=depth, vp_km_s=vp, vs_km_s=vs))

	rows.sort()
	return rows


def read_1dvel(
	src: Path = DEFAULT_1DVEL_SRC,
	*,
	strict: bool = False,
) -> list[Vel1DPoint]:
	"""Read a generic 1D velocity file from disk.

	File format assumption:
	    Vp Vs Depth  (first 3 columns)
	"""
	lines = src.read_text().splitlines()
	return parse_1dvel_lines(lines, strict=strict)


def read_vs_model_44mod(path: Path) -> pd.DataFrame:
	if not path.is_file():
		raise FileNotFoundError(f'vs model not found: {path}')

	df = pd.read_csv(
		path,
		sep=r'\s+',
		header=None,
		names=['depth_m', 'vs_mps', 'sigma_mps'],
		comment='#',
		skiprows=1,
	)

	for c in ['depth_m', 'vs_mps', 'sigma_mps']:
		df[c] = pd.to_numeric(df[c], errors='coerce')

	df = df.dropna(subset=['depth_m', 'vs_mps']).copy()
	if df.empty:
		raise ValueError(f'no numeric rows found in: {path}')

	df = (
		df.sort_values('depth_m')
		.drop_duplicates(subset=['depth_m'], keep='first')
		.reset_index(drop=True)
	)

	if (df['depth_m'] < 0).any():
		raise ValueError('depth_m must be non-negative')
	if (df['vs_mps'] <= 0).any():
		raise ValueError('vs_mps must be positive')

	return df[['depth_m', 'vs_mps', 'sigma_mps']].copy()


def to_nll_layer_lines(
	rows: Sequence[Vel1DPoint],
	*,
	vp_grad: float = 0.0,
	vs_grad: float = 0.0,
	rho_top: float = 0.0,
	rho_grad: float = 0.0,
) -> list[str]:
	"""Convert 1D velocity points to NonLinLoc LAYER lines.

	NonLinLoc LAYER format:
	    LAYER depth Vp_top Vp_grad Vs_top Vs_grad rho_top rho_grad

	Defaults:
	    - Gradients are zero (piecewise-constant layers).
	    - Density terms are zeroed.
	"""
	out: list[str] = []
	for r in rows:
		out.append(
			'LAYER '
			f'{r.depth_km:.3f} '
			f'{r.vp_km_s:.3f} {vp_grad:.1f} '
			f'{r.vs_km_s:.3f} {vs_grad:.1f} '
			f'{rho_top:.1f} {rho_grad:.1f}'
		)
	return out


def nll_layers_text_from_1d_model(
	*,
	z_m: np.ndarray,
	vs_mps: np.ndarray,
	vp_over_vs: float,
	rho_gcc: float | None = None,
) -> str:
	if vp_over_vs <= 0:
		raise ValueError(f'vp_over_vs must be positive. got {vp_over_vs}')

	depth_m = np.asarray(z_m, dtype=float)
	vs_mps = np.asarray(vs_mps, dtype=float)

	if depth_m.ndim != 1 or vs_mps.ndim != 1:
		raise ValueError('z_m and vs_mps must be 1D arrays')
	if depth_m.size == 0:
		raise ValueError('z_m and vs_mps must be non-empty')
	if depth_m.size != vs_mps.size:
		raise ValueError('z_m and vs_mps must have the same length')
	if np.any(np.diff(depth_m) < 0):
		raise ValueError('z_m must be monotonically non-decreasing')

	depth_km = depth_m / 1000.0
	vs_km_s = vs_mps / 1000.0
	vp_km_s = vs_km_s * float(vp_over_vs)
	rho_top = 0.0 if rho_gcc is None else float(rho_gcc)

	lines: list[str] = []
	for d, vp, vs in zip(
		depth_km.tolist(),
		vp_km_s.tolist(),
		vs_km_s.tolist(),
	):
		lines.append(f'LAYER {d:.3f} {vp:.3f} 0.0 {vs:.3f} 0.0 {rho_top:.1f} 0.0')

	text = '\n'.join(lines) + '\n'
	if not text.startswith('LAYER '):
		raise ValueError('layers text must start with LAYER')
	return text


def write_layers(
	out_path: Path = DEFAULT_NLL_LAYERS_OUT,
	layer_lines: Sequence[str] = (),
) -> Path:
	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text('\n'.join(layer_lines) + '\n')
	return out_path


def convert_1dvel_to_nll_layers(
	*,
	src: Path = DEFAULT_1DVEL_SRC,
	out: Path = DEFAULT_NLL_LAYERS_OUT,
	strict: bool = False,
) -> Path:
	rows = read_1dvel(src, strict=strict)
	layer_lines = to_nll_layer_lines(rows)
	return write_layers(out, layer_lines)


def main() -> Path:
	"""Non-CLI default entry point.

	Adjust DEFAULT_1DVEL_SRC and DEFAULT_NLL_LAYERS_OUT if needed.
	"""
	return convert_1dvel_to_nll_layers(
		src=DEFAULT_1DVEL_SRC,
		out=DEFAULT_NLL_LAYERS_OUT,
		strict=False,
	)


if __name__ == '__main__':
	result = main()
	print(result)
