from __future__ import annotations

import numpy as np
import pandas as pd


def parse_event_subsample_3ints(
	value: object,
	*,
	key: str,
	min_value: int,
	field_prefix: str = 'event_subsample',
) -> tuple[int, int, int]:
	if value is None:
		raise ValueError(f'{field_prefix}.{key} must be provided')
	if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
		raise ValueError(f'{field_prefix}.{key} must be a list of 3 integers')
	if len(value) != 3:
		raise ValueError(f'{field_prefix}.{key} must have exactly 3 elements')

	out: list[int] = []
	for i, raw in enumerate(value):
		if isinstance(raw, bool) or not isinstance(raw, (int, np.integer)):
			raise ValueError(
				f'{field_prefix}.{key}[{i}] must be an integer >= {min_value}'
			)
		v = int(raw)
		if v < min_value:
			raise ValueError(
				f'{field_prefix}.{key}[{i}] must be an integer >= {min_value}'
			)
		out.append(v)
	return (out[0], out[1], out[2])


def validate_event_subsample_config(
	event_subsample: object,
	*,
	field: str = 'event_subsample',
	allow_keep_n_xyz: bool = True,
) -> dict[str, list[int]] | None:
	if event_subsample is None:
		return None
	if not isinstance(event_subsample, dict):
		raise ValueError(f'{field} must be a mapping')

	allowed = {'stride_ijk', 'keep_n_xyz'} if allow_keep_n_xyz else {'stride_ijk'}
	extra = set(event_subsample.keys()) - allowed
	if not allow_keep_n_xyz and 'keep_n_xyz' in event_subsample:
		raise ValueError(f'{field}.keep_n_xyz is not supported')
	if extra:
		raise ValueError(
			f'{field} contains unknown keys: {sorted(extra)!r}; '
			f'allowed: {sorted(allowed)!r}'
		)

	has_stride = 'stride_ijk' in event_subsample
	has_keep = 'keep_n_xyz' in event_subsample
	if has_stride and has_keep:
		raise ValueError(
			'event_subsample.stride_ijk and event_subsample.keep_n_xyz '
			'cannot be specified at the same time'
		)

	if has_stride:
		return {
			'stride_ijk': list(
				parse_event_subsample_3ints(
					event_subsample['stride_ijk'],
					key='stride_ijk',
					min_value=1,
					field_prefix=field,
				)
			)
		}
	if has_keep:
		return {
			'keep_n_xyz': list(
				parse_event_subsample_3ints(
					event_subsample['keep_n_xyz'],
					key='keep_n_xyz',
					min_value=1,
					field_prefix=field,
				)
			)
		}
	return None


def event_subsample_mask_from_xyz(
	df: pd.DataFrame,
	*,
	stride_ijk: tuple[int, int, int] | None,
	keep_n_xyz: tuple[int, int, int] | None,
) -> np.ndarray:
	if stride_ijk is not None and keep_n_xyz is not None:
		raise ValueError(
			'event_subsample.stride_ijk and event_subsample.keep_n_xyz '
			'cannot be specified at the same time'
		)

	xi = np.rint(df['x_m'].astype(float).to_numpy()).astype(int)
	yi = np.rint(df['y_m'].astype(float).to_numpy()).astype(int)
	zi = np.rint(df['z_m'].astype(float).to_numpy()).astype(int)

	ux = np.unique(xi)
	uy = np.unique(yi)
	uz = np.unique(zi)

	x_ijk = np.searchsorted(ux, xi).astype(int)
	y_ijk = np.searchsorted(uy, yi).astype(int)
	z_ijk = np.searchsorted(uz, zi).astype(int)

	if keep_n_xyz is not None:
		nx, ny, nz = keep_n_xyz
		if nx > int(ux.size) or ny > int(uy.size) or nz > int(uz.size):
			raise ValueError(
				'event_subsample.keep_n_xyz exceeds unique grid counts: '
				f'keep={keep_n_xyz} unique=({int(ux.size)}, {int(uy.size)}, {int(uz.size)})'
			)
		x_keep = np.unique(np.rint(np.linspace(0, ux.size - 1, nx)).astype(int))
		y_keep = np.unique(np.rint(np.linspace(0, uy.size - 1, ny)).astype(int))
		z_keep = np.unique(np.rint(np.linspace(0, uz.size - 1, nz)).astype(int))
		return (
			np.isin(x_ijk, x_keep)
			& np.isin(y_ijk, y_keep)
			& np.isin(z_ijk, z_keep)
		)

	if stride_ijk is None:
		stride_ijk = (1, 1, 1)
	sx, sy, sz = stride_ijk
	return ((x_ijk % sx) == 0) & ((y_ijk % sy) == 0) & ((z_ijk % sz) == 0)
