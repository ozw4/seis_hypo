from __future__ import annotations

import numpy as np
import zarr


def load_slices(root: zarr.hierarchy.Group) -> dict[str, tuple[int, int]]:
	d = root.attrs.get('slices', None)
	if not isinstance(d, dict):
		d = root.attrs.get('channel_slices_0based', None)
	if not isinstance(d, dict):
		raise ValueError(
			"Zarr attrs must contain dict 'slices' (or 'channel_slices_0based')."
		)

	out: dict[str, tuple[int, int]] = {}
	for k, v in d.items():
		out[str(k)] = (int(v[0]), int(v[1]))  # [start, stop) in raw channel id space
	return out


def ab_keep_idx_and_ids(
	root: zarr.hierarchy.Group,
	*,
	well_a_keep_0based_incl: tuple[int, int],
	well_b_keep_0based_incl: tuple[int, int],
	well_a_key: str = '78A',
	well_b_key: str = '78B',
) -> tuple[np.ndarray, np.ndarray]:
	slices = load_slices(root)
	a0, a1 = slices[well_a_key]
	b0, b1 = slices[well_b_key]

	ka0, ka1 = int(well_a_keep_0based_incl[0]), int(well_a_keep_0based_incl[1])
	kb0, kb1 = int(well_b_keep_0based_incl[0]), int(well_b_keep_0based_incl[1])

	if not (a0 <= ka0 <= ka1 < a1):
		raise ValueError(
			f'A keep out of range: {well_a_keep_0based_incl} not in [{a0},{a1})'
		)
	if not (b0 <= kb0 <= kb1 < b1):
		raise ValueError(
			f'B keep out of range: {well_b_keep_0based_incl} not in [{b0},{b1})'
		)

	n_a = a1 - a0
	off_a = 0
	off_b = n_a

	ia0 = off_a + (ka0 - a0)
	ia1 = off_a + (ka1 - a0)
	ib0 = off_b + (kb0 - b0)
	ib1 = off_b + (kb1 - b0)

	keep_idx = np.concatenate(
		[
			np.arange(ia0, ia1 + 1, dtype=np.int32),
			np.arange(ib0, ib1 + 1, dtype=np.int32),
		]
	)
	ch_ids = np.concatenate(
		[
			np.arange(ka0, ka1 + 1, dtype=np.int32),
			np.arange(kb0, kb1 + 1, dtype=np.int32),
		]
	)
	return keep_idx, ch_ids


def all_channel_ids_from_slices(
	root: zarr.hierarchy.Group,
	*,
	well_a_key: str = '78A',
	well_b_key: str = '78B',
) -> np.ndarray:
	slices = load_slices(root)
	a0, a1 = slices[well_a_key]
	b0, b1 = slices[well_b_key]
	return np.concatenate(
		[
			np.arange(a0, a1, dtype=np.int32),
			np.arange(b0, b1, dtype=np.int32),
		]
	)


def match_hits(names: np.ndarray, target: str, mode: str) -> np.ndarray:
	if mode == 'exact':
		return np.flatnonzero(names == target)
	if mode == 'endswith':
		return np.flatnonzero(
			np.fromiter(
				(str(x).endswith(target) for x in names), dtype=bool, count=names.size
			)
		)
	raise ValueError(f'MATCH_MODE must be exact or endswith, got {mode}')


def concat_blocks(blocks: np.ndarray, valid: np.ndarray) -> np.ndarray:
	# blocks: (B,C,Tb), valid: (B,)
	B, C, Tb = int(blocks.shape[0]), int(blocks.shape[1]), int(blocks.shape[2])
	v = valid.astype(np.int32, copy=False)
	total = int(v.sum())
	out = np.empty((C, total), dtype=np.float32)
	pos = 0
	for i in range(B):
		n = int(v[i])
		if n > 0:
			out[:, pos : pos + n] = blocks[i, :, :n]
			pos += n
	if pos != total:
		raise ValueError(
			f'concat size mismatch: pos={pos} total={total} (Tb={Tb} B={B})'
		)
	return out
