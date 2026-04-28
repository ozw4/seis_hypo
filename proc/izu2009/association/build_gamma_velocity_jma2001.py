# %%
"""Build GaMMA velocity-model JSON from the existing JMA2001 NLL LAYER file."""

# file: proc/izu2009/association/build_gamma_velocity_jma2001.py
#
# Purpose:
# - Convert the existing NonLinLoc-style JMA2001 LAYER velocity model to the
#   GaMMA eikonal JSON format expected by gamma_workflow.velocity.load_velocity_json().
# - Preserve the original depth sampling and Vp/Vs values.

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _REPO_ROOT / 'src'

for _path in (_REPO_ROOT, _SRC_DIR):
	_path_str = str(_path)
	if _path_str not in sys.path:
		sys.path.insert(0, _path_str)

from common.json_io import write_json
from gamma_workflow.velocity import load_velocity_json

IN_NLL_LAYERS = (
	_REPO_ROOT / 'proc/loki_hypo/mobara/mobara_traveltime/velocity/jma2001.layers'
)
OUT_DIR = _REPO_ROOT / 'proc/izu2009/association/in'
OUT_GAMMA_VELOCITY_JSON = OUT_DIR / 'gamma_vel_jma2001.json'
OUT_GAMMA_VELOCITY_CSV = OUT_DIR / 'gamma_vel_jma2001.csv'

REQUIRED_OUTPUT_KEYS = ['z', 'p', 's']
OUT_CSV_COLUMNS = ['z_km', 'vp_km_s', 'vs_km_s']


def _require_finite(value: float, *, column: str, line_no: int, path: Path) -> None:
	if not math.isfinite(value):
		raise ValueError(f'{path}: line {line_no}: non-finite {column}: {value}')


def _parse_nll_layer_rows(layers_path: Path) -> list[dict[str, float]]:
	if not layers_path.is_file():
		raise FileNotFoundError(f'NLL LAYER file not found: {layers_path}')

	rows: list[dict[str, float]] = []
	lines = layers_path.read_text(encoding='utf-8').splitlines()

	for line_no, raw_line in enumerate(lines, start=1):
		line = raw_line.strip()
		if not line or line.startswith('#'):
			continue

		parts = line.split()
		if len(parts) != 8:
			raise ValueError(
				f'{layers_path}: line {line_no} must have 8 columns. '
				f'got {len(parts)}: {raw_line!r}'
			)
		if parts[0] != 'LAYER':
			raise ValueError(
				f'{layers_path}: line {line_no} must start with LAYER: {raw_line!r}'
			)

		z_km = float(parts[1])
		vp_km_s = float(parts[2])
		vp_grad = float(parts[3])
		vs_km_s = float(parts[4])
		vs_grad = float(parts[5])
		rho = float(parts[6])
		rho_grad = float(parts[7])

		for column, value in [
			('z_km', z_km),
			('vp_km_s', vp_km_s),
			('vp_grad', vp_grad),
			('vs_km_s', vs_km_s),
			('vs_grad', vs_grad),
			('rho', rho),
			('rho_grad', rho_grad),
		]:
			_require_finite(value, column=column, line_no=line_no, path=layers_path)

		rows.append(
			{
				'z_km': z_km,
				'vp_km_s': vp_km_s,
				'vs_km_s': vs_km_s,
			}
		)

	if not rows:
		raise ValueError(f'no LAYER rows found: {layers_path}')

	return rows


def _validate_gamma_velocity_rows(rows: list[dict[str, float]], label: str) -> None:
	previous_z: float | None = None

	for idx, row in enumerate(rows):
		z_km = row['z_km']
		vp_km_s = row['vp_km_s']
		vs_km_s = row['vs_km_s']

		if z_km < 0.0:
			raise ValueError(f'{label}: row {idx}: negative z_km: {z_km}')
		if vp_km_s <= 0.0:
			raise ValueError(f'{label}: row {idx}: non-positive vp_km_s: {vp_km_s}')
		if vs_km_s <= 0.0:
			raise ValueError(f'{label}: row {idx}: non-positive vs_km_s: {vs_km_s}')
		if vp_km_s <= vs_km_s:
			raise ValueError(
				f'{label}: row {idx}: vp_km_s must be greater than vs_km_s. '
				f'vp={vp_km_s}, vs={vs_km_s}'
			)
		if previous_z is not None and z_km <= previous_z:
			raise ValueError(
				f'{label}: row {idx}: z_km must be strictly increasing. '
				f'previous={previous_z}, current={z_km}'
			)

		previous_z = z_km

	if rows[0]['z_km'] != 0.0:
		raise ValueError(f'{label}: first z_km must be 0.0, got {rows[0]["z_km"]}')


def _rows_to_velocity_dict(rows: list[dict[str, float]]) -> dict[str, list[float]]:
	vel = {
		'z': [row['z_km'] for row in rows],
		'p': [row['vp_km_s'] for row in rows],
		's': [row['vs_km_s'] for row in rows],
	}

	missing = [key for key in REQUIRED_OUTPUT_KEYS if key not in vel]
	if missing:
		raise ValueError(f'output velocity dict missing keys: {missing}')

	return vel


def build_gamma_velocity_jma2001(
	layers_path: Path = IN_NLL_LAYERS,
) -> tuple[dict[str, list[float]], list[dict[str, float]]]:
	"""Create a GaMMA velocity dictionary from a NonLinLoc LAYER file."""
	rows = _parse_nll_layer_rows(Path(layers_path))
	_validate_gamma_velocity_rows(rows, str(layers_path))
	return _rows_to_velocity_dict(rows), rows


def _write_velocity_csv(path: Path, rows: list[dict[str, float]]) -> None:
	with path.open('w', encoding='utf-8', newline='') as f:
		writer = csv.DictWriter(f, fieldnames=OUT_CSV_COLUMNS)
		writer.writeheader()
		writer.writerows(rows)


def main() -> None:
	"""Run JMA2001 LAYER-to-GaMMA velocity conversion."""
	OUT_DIR.mkdir(parents=True, exist_ok=True)

	vel, rows = build_gamma_velocity_jma2001(IN_NLL_LAYERS)
	_write_velocity_csv(OUT_GAMMA_VELOCITY_CSV, rows)
	write_json(OUT_GAMMA_VELOCITY_JSON, vel, ensure_ascii=False, indent=2)
	load_velocity_json(OUT_GAMMA_VELOCITY_JSON)

	z_values = [row['z_km'] for row in rows]
	vp_values = [row['vp_km_s'] for row in rows]
	vs_values = [row['vs_km_s'] for row in rows]

	print('Wrote:', OUT_GAMMA_VELOCITY_JSON)
	print('Wrote:', OUT_GAMMA_VELOCITY_CSV)
	print('Rows:', len(rows))
	print('Depth range km:', min(z_values), '->', max(z_values))
	print('Vp range km/s:', min(vp_values), '->', max(vp_values))
	print('Vs range km/s:', min(vs_values), '->', max(vs_values))


if __name__ == '__main__':
	main()

# 実行例:
# python proc/izu2009/association/build_gamma_velocity_jma2001.py
