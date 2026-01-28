from __future__ import annotations

from pathlib import Path

import pandas as pd

from common.core import validate_columns
from hypo.crh import write_crh


def compute_reference_elevation_km(
	station_df: pd.DataFrame,
	*,
	elevation_col: str = 'Elevation_m',
	margin_m: float = 0.0,
) -> float:
	"""Compute reference elevation (km) for HypoInverse CRE.

	Definition:
		ref_elev_km = (max(Elevation_m) + margin_m) / 1000

	For borehole-only arrays where max(Elevation_m) <= 0, a negative reference
	elevation is rarely useful. To keep the depth datum sane, this function clamps
	the reference elevation to be non-negative (>= 0 m).
	"""
	if station_df is None:
		raise ValueError('station_df must not be None')

	validate_columns(station_df, [elevation_col], 'station DataFrame')
	if station_df.empty:
		raise ValueError('station_df is empty')

	s = station_df[elevation_col].dropna()
	if s.empty:
		raise ValueError(f'{elevation_col} has no valid values')

	max_m = float(s.max())
	ref_m = max_m + float(margin_m)
	if ref_m < 0.0:
		ref_m = 0.0

	return ref_m / 1000.0


def compute_typical_station_elevation_km(*, explicit_m: float | None) -> float:
	"""Compute typical station elevation (km).

	- If explicit_m is provided: typical_elev_km = explicit_m / 1000
	- If explicit_m is None: typical_elev_km = 0.0
	"""
	if explicit_m is None:
		return 0.0
	return float(explicit_m) / 1000.0


def compute_cre_layer_top_shift_km(ref_elev_km: float, typical_elev_km: float) -> float:
	"""Compute layer-top shift (km) to convert CRH-style depths to CRE datum."""
	return float(ref_elev_km) - float(typical_elev_km)


def apply_layer_top_shift_km(
	layer_tops_km: list[float], shift_km: float
) -> list[float]:
	"""Apply a layer-top shift (km) while forcing the first top to 0.0."""
	if layer_tops_km is None or len(layer_tops_km) == 0:
		raise ValueError('layer_tops_km must not be empty')

	tops = [float(z) for z in layer_tops_km]
	tops[0] = 0.0

	for i in range(1, len(tops)):
		if tops[i] <= tops[i - 1]:
			raise ValueError('layer_tops_km must be strictly increasing')

	s = float(shift_km)
	for i in range(1, len(tops)):
		tops[i] = float(tops[i]) + s

	for i in range(1, len(tops)):
		if tops[i] <= tops[i - 1]:
			raise ValueError('shifted layer_tops_km must be strictly increasing')

	return tops


def _write_scalar(path: Path, value: float) -> None:
	# Use a stable, round-trippable representation.
	text = format(float(value), '.15g')
	path.write_text(text + '\n', encoding='utf-8')


def write_cre_meta(
	run_dir: Path,
	*,
	ref_elev_km: float,
	typical_elev_km: float,
	shift_km: float,
) -> None:
	"""Write CRE parameter metadata to run_dir.

	Outputs:
	- cre_ref_elev_km.txt
	- cre_typical_station_elev_km.txt
	- cre_layer_top_shift_km.txt
	"""
	d = Path(run_dir)
	d.mkdir(parents=True, exist_ok=True)

	_write_scalar(d / 'cre_ref_elev_km.txt', ref_elev_km)
	_write_scalar(d / 'cre_typical_station_elev_km.txt', typical_elev_km)
	_write_scalar(d / 'cre_layer_top_shift_km.txt', shift_km)


def write_cre_from_layer_tops(
	run_dir: Path,
	*,
	vp_kms: float,
	vs_kms: float,
	layer_tops_km: list[float],
	shift_km: float,
) -> tuple[Path, Path]:
	"""Write P/S CRE model files (P.cre / S.cre) from arbitrary layer tops.

	This function only converts (tops -> shifted tops) and writes files.
	It does not generate synthetic tops.
	"""
	d = Path(run_dir)
	d.mkdir(parents=True, exist_ok=True)

	tops_km = apply_layer_top_shift_km(layer_tops_km, shift_km=float(shift_km))

	p_cre = d / 'P.cre'
	s_cre = d / 'S.cre'

	write_crh(p_cre, 'CRE_P', [(float(vp_kms), float(z)) for z in tops_km])
	write_crh(s_cre, 'CRE_S', [(float(vs_kms), float(z)) for z in tops_km])

	return p_cre, s_cre
