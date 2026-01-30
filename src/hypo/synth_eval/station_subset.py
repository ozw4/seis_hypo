from __future__ import annotations

import numpy as np


def _parse_indices_field(value: object, *, label: str) -> tuple[bool, list[int]]:
	if isinstance(value, str):
		if value.strip().lower() != 'all':
			raise ValueError(
				f'station_subset.{label} must be "all" or list[int], got: {value!r}'
			)
		return True, []
	if isinstance(value, list):
		seen: set[int] = set()
		indices: list[int] = []
		for item in value:
			if isinstance(item, bool):
				raise ValueError(
					f'station_subset.{label} entries must be int, got bool'
				)
			if not isinstance(item, int):
				raise ValueError(
					f'station_subset.{label} entries must be int, got {type(item).__name__}'
				)
			if item in seen:
				raise ValueError(
					f'station_subset.{label} has duplicate indices: {value!r}'
				)
			seen.add(item)
			indices.append(item)
		return False, indices
	raise ValueError(
		f'station_subset.{label} must be "all" or list[int], got: {value!r}'
	)


def _validate_station_subset_value(key: str, value: object) -> None:
	_parse_indices_field(value, label=key)


def validate_station_subset_schema(station_subset: object) -> dict[str, object]:
	if not isinstance(station_subset, dict):
		raise ValueError(
			'station_subset must be a dict with keys surface_indices/das_indices'
		)

	allowed = {'surface_indices', 'das_indices'}
	extra = set(station_subset.keys()) - allowed
	if extra:
		raise ValueError(f'station_subset has unexpected keys: {sorted(extra)}')
	if not station_subset:
		raise ValueError('station_subset must include surface_indices or das_indices')

	for key, value in station_subset.items():
		_validate_station_subset_value(str(key), value)

	return dict(station_subset)


def _resolve_indices(
	global_indices: np.ndarray,
	*,
	use_all: bool,
	indices: list[int],
	label: str,
) -> np.ndarray:
	if use_all:
		return global_indices
	total = len(global_indices)
	for item in indices:
		if item < 0 or item >= total:
			raise ValueError(
				f'station_subset.{label} index {item} out of range (0 <= i < {total})'
			)
	if not indices:
		return global_indices[:0]
	return global_indices[np.array(indices, dtype=int)]


def normalize_station_subset(
	station_subset: dict[str, object],
	*,
	codes: list[str] | np.ndarray,
	expected_len: int | None = None,
	min_points: int = 4,
) -> np.ndarray:
	"""Return global receiver_indices (unique, sorted).

	Optionally validate codes length against expected_len.
	"""
	station_subset = validate_station_subset_schema(station_subset)

	codes_arr = np.asarray(codes)
	if codes_arr.ndim != 1:
		raise ValueError('codes must be a 1D sequence of station_code')
	if expected_len is not None:
		if isinstance(expected_len, bool) or not isinstance(expected_len, int):
			raise ValueError('expected_len must be an int')
		if expected_len < 0:
			raise ValueError('expected_len must be non-negative')
		if codes_arr.size != expected_len:
			raise ValueError(
				f'codes length {codes_arr.size} does not match expected_len={expected_len}'
			)

	codes_upper = np.char.upper(codes_arr.astype(str))
	stations_is_das = np.char.startswith(codes_upper, 'D')
	surface_global = np.where(~stations_is_das)[0]
	das_global = np.where(stations_is_das)[0]

	surface_all = False
	surface_indices: list[int] = []
	if 'surface_indices' in station_subset:
		surface_all, surface_indices = _parse_indices_field(
			station_subset['surface_indices'],
			label='surface_indices',
		)

	das_all = False
	das_indices: list[int] = []
	if 'das_indices' in station_subset:
		das_all, das_indices = _parse_indices_field(
			station_subset['das_indices'],
			label='das_indices',
		)

	surface_selected = _resolve_indices(
		surface_global,
		use_all=surface_all,
		indices=surface_indices,
		label='surface_indices',
	)
	das_selected = _resolve_indices(
		das_global,
		use_all=das_all,
		indices=das_indices,
		label='das_indices',
	)

	if surface_selected.size == 0 and das_selected.size == 0:
		raise ValueError('station_subset selects no stations')

	receiver_indices = np.unique(np.concatenate([surface_selected, das_selected]))
	if receiver_indices.size < min_points:
		raise ValueError(
			f'station_subset selects {receiver_indices.size} stations, '
			f'fewer than min_points={min_points}'
		)
	return receiver_indices
