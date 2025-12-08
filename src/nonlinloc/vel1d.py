# %%
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

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
