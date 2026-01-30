from __future__ import annotations


def _validate_station_subset_value(key: str, value: object) -> None:
	if isinstance(value, str):
		if value.strip().lower() != 'all':
			raise ValueError(
				f'station_subset.{key} must be "all" or list[int], got: {value!r}'
			)
		return
	if isinstance(value, list):
		seen: set[int] = set()
		for item in value:
			if isinstance(item, bool):
				raise ValueError(f'station_subset.{key} entries must be int, got bool')
			if not isinstance(item, int):
				raise ValueError(
					f'station_subset.{key} entries must be int, got {type(item).__name__}'
				)
			if item in seen:
				raise ValueError(
					f'station_subset.{key} has duplicate indices: {value!r}'
				)
			seen.add(item)
		return
	raise ValueError(
		f'station_subset.{key} must be "all" or list[int], got: {value!r}'
	)


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
