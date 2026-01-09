from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from common.time_util import utc_ms_to_iso
from io_util.forge_debug_plot_util import match_hits


def _temp_zarr_label(out_dir: Path) -> str:
	name = out_dir.name
	if '60s' in name:
		return '60s'
	return 'window'


def make_temp_zarr_window_from_tdms_start(
	*,
	root: zarr.hierarchy.Group,
	dataset_name: str,
	tdms_file_name: str,
	match_mode: str,
	tdms_occurrence: int,
	out_dir: Path,
	n_zarr: int,
) -> tuple[Path, int, int, float]:
	block = root[dataset_name]  # (B,C,Tb)
	tb = int(block.shape[2])
	fs_zarr = float(root.attrs['fs_out_hz'])
	seconds = float(n_zarr) / float(fs_zarr)

	names = np.asarray(root['file_name'][:], dtype=str)
	hits = match_hits(names, tdms_file_name, match_mode)
	if hits.size == 0:
		raise ValueError(f'TDMS not found: {tdms_file_name} (mode={match_mode})')
	if not (0 <= int(tdms_occurrence) < int(hits.size)):
		raise ValueError(
			f'TDMS_OCCURRENCE out of range: {tdms_occurrence} / hits={hits.size}'
		)

	i_target = int(hits[int(tdms_occurrence)])

	seg = np.asarray(root['segment_id'][:], dtype=np.int32)
	seg_id = int(seg[i_target])
	seg_idx = np.flatnonzero(seg == seg_id)
	b0 = int(seg_idx[0])
	b1 = int(seg_idx[-1])

	valid_seg = np.asarray(root['valid_out_samples'][b0 : b1 + 1], dtype=np.int32)
	cum = np.concatenate(
		[np.array([0], dtype=np.int64), np.cumsum(valid_seg.astype(np.int64))]
	)

	j_target = int(i_target - b0)
	start_off = int(cum[j_target])
	end_off = int(start_off + n_zarr)

	if end_off > int(cum[-1]):
		avail = float(int(cum[-1]) - start_off) / float(fs_zarr)
		label = _temp_zarr_label(out_dir)
		suffix = '60s' if label == '60s' else 'window'
		raise ValueError(
			f'not enough samples in the segment to take {suffix} from TDMS start: '
			f'need={seconds:.3f}s, available={avail:.3f}s'
		)

	j_end = int(np.searchsorted(cum, end_off, side='left'))
	if not (j_target < j_end):
		raise ValueError(
			f'bad block range: j_target={j_target} j_end={j_end} '
			f'(start_off={start_off} end_off={end_off})'
		)

	bs = int(b0 + j_target)
	be = int(b0 + j_end)  # exclusive

	blk = np.asarray(block[bs:be, :, :], dtype=np.float32)
	valid = valid_seg[j_target:j_end].astype(np.int32, copy=True)

	last_start = int(cum[j_end - 1])
	valid[-1] = int(end_off - last_start)

	start_ms = np.asarray(root['starttime_utc_ms'][bs:be], dtype=np.int64)
	file_name = np.asarray(root['file_name'][bs:be], dtype=str)

	out_dir.mkdir(parents=True, exist_ok=False)
	g = zarr.open_group(str(out_dir), mode='w')

	g.attrs['fs_out_hz'] = float(fs_zarr)
	slices = root.attrs.get('slices', None)
	if isinstance(slices, dict):
		g.attrs['slices'] = dict(slices)
	channel_slices = root.attrs.get('channel_slices_0based', None)
	if isinstance(channel_slices, dict):
		g.attrs['channel_slices_0based'] = dict(channel_slices)

	g.create_dataset(dataset_name, data=blk, dtype=np.float32, chunks=blk.shape)
	g.create_dataset(
		'valid_out_samples', data=valid, dtype=np.int32, chunks=valid.shape
	)
	g.create_dataset(
		'starttime_utc_ms', data=start_ms, dtype=np.int64, chunks=start_ms.shape
	)
	g.create_dataset(
		'segment_id', data=np.zeros((blk.shape[0],), dtype=np.int32), dtype=np.int32
	)
	g.create_dataset(
		'file_name', data=file_name.astype('U'), dtype=file_name.astype('U').dtype
	)

	first_window_start_ms = int(start_ms[0])

	label = _temp_zarr_label(out_dir)
	print(f'=== Temp Zarr ({label}) ===')
	print(f'tdms      : {tdms_file_name} (mode={match_mode}, occ={tdms_occurrence})')
	print(f'segment_id: {seg_id} (orig), temp_segment_id=0')
	print(f'blocks    : orig[{bs}:{be}] (B={be - bs}), Tb={tb}, fs_zarr={fs_zarr}')
	print(
		f'window    : start_off={start_off} samples, n_zarr={n_zarr} samples -> {seconds:.3f}s'
	)
	print(
		f'start_ms  : {first_window_start_ms} ({utc_ms_to_iso(first_window_start_ms)})'
	)
	print(f'temp_zarr : {out_dir}')

	return out_dir, first_window_start_ms, n_zarr, float(fs_zarr)
