from __future__ import annotations

from pathlib import Path

import pandas as pd

from common.core import validate_columns


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
