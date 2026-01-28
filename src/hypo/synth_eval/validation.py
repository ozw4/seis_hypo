from __future__ import annotations

from pathlib import Path


def require_abs(p: Path, key: str) -> None:
	if not p.is_absolute():
		raise ValueError(f'{key} must be an absolute path: {p}')


def require_filename_only(name: str, key: str) -> None:
	if '/' in name or '\\' in name:
		raise ValueError(f'{key} must be filename only (no directory): {name}')


def require_dirname_only(name: str, key: str) -> None:
	if '/' in name or '\\' in name:
		raise ValueError(
			f'{key} must be directory name only (no path separators): {name}'
		)
	if name.strip() == '':
		raise ValueError(f'{key} must be non-empty')


def validate_elevation_correction_config(
	*,
	model_type: str,
	use_station_elev: bool,
	apply_station_elevation_delay: bool,
) -> None:
	mt = str(model_type).strip().upper()
	if mt not in ('CRE', 'CRH'):
		raise ValueError(f"model_type must be 'CRE' or 'CRH', got: {model_type!r}")

	if mt != 'CRE' and bool(use_station_elev):
		raise ValueError("use_station_elev is only supported when model_type='CRE'")

	if mt == 'CRE' and bool(use_station_elev) and bool(apply_station_elevation_delay):
		raise ValueError(
			'apply_station_elevation_delay must be False when use_station_elev=True'
		)
