from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr

# Reuse your repo functions (same style as pipeline_loki_waveform_stacking_eqt)


@dataclass(frozen=True)
class WindowMeta:
	segment_id: int
	block_start: int
	window_start_utc_ms: int
	fs_hz: float
	channels: tuple[int, int]  # [start, end)


class ZarrBlockWindowIterator:
	"""Read block-aligned windows from a Zarr store:
	  block: (B, C, Tb)
	Constraints:
	  - windows never cross segment_id boundaries
	  - trims trailing short blocks in a segment (valid_out_samples < Tb)

	Yields:
	  wave: (Csel, EQT_IN_SAMPLES)
	  meta: WindowMeta

	"""

	def __init__(
		self,
		zarr_dir: Path,
		dataset_name: str,
		in_samples: int,
		overlap: int,
		channel_range: tuple[int, int] | None = None,
	):
		self.zarr_dir = Path(zarr_dir)
		self.root = zarr.open_group(str(self.zarr_dir), mode='r')

		if dataset_name not in self.root:
			raise ValueError(
				f"Dataset '{dataset_name}' not found in Zarr: {self.zarr_dir}"
			)
		self.arr = self.root[dataset_name]
		if self.arr.ndim != 3:
			raise ValueError(f'Expected 3D (B,C,Tb), got ndim={self.arr.ndim}')

		self.n_blocks, self.n_ch, self.tb = self.arr.shape

		for k in ('starttime_utc_ms', 'valid_out_samples', 'segment_id'):
			if k not in self.root:
				raise ValueError(
					f"Missing required dataset '{k}' in Zarr: {self.zarr_dir}"
				)

		self.start_ms = self.root['starttime_utc_ms'][:].astype(np.int64, copy=False)
		self.valid_out = self.root['valid_out_samples'][:].astype(np.int32, copy=False)
		self.seg = self.root['segment_id'][:].astype(np.int32, copy=False)

		fs_out = self.root.attrs.get('fs_out_hz', None)
		if fs_out is None:
			raise ValueError('Missing fs_out_hz in Zarr attrs')
		self.fs_hz = float(fs_out)

		self.in_samples = int(in_samples)
		self.overlap = int(overlap)
		if self.overlap < 0 or self.overlap >= self.in_samples:
			raise ValueError('overlap must satisfy 0 <= overlap < in_samples')

		self.hop = self.in_samples - self.overlap

		if (self.in_samples % self.tb) != 0 or (self.hop % self.tb) != 0:
			raise ValueError(
				f'in_samples ({self.in_samples}) and hop ({self.hop}) must be multiples of Tb ({self.tb})'
			)

		self.w_blocks = self.in_samples // self.tb
		self.h_blocks = self.hop // self.tb

		if channel_range is None:
			self.ch0, self.ch1 = 0, self.n_ch
		else:
			self.ch0, self.ch1 = int(channel_range[0]), int(channel_range[1])
			if not (0 <= self.ch0 < self.ch1 <= self.n_ch):
				raise ValueError(
					f'Invalid channel_range={channel_range} for n_ch={self.n_ch}'
				)

	def _segment_ranges(self) -> Iterator[tuple[int, int, int]]:
		if self.n_blocks == 0:
			return
		s0 = int(self.seg[0])
		start = 0
		for i in range(1, self.n_blocks):
			si = int(self.seg[i])
			if si != s0:
				yield (s0, start, i)
				s0 = si
				start = i
		yield (s0, start, self.n_blocks)

	def __iter__(self) -> Iterator[tuple[np.ndarray, WindowMeta]]:
		tb = self.tb
		w = self.w_blocks
		hb = self.h_blocks
		ch0, ch1 = self.ch0, self.ch1

		for seg_id, s_start, s_end in self._segment_ranges():
			# trim trailing short blocks (common when segment ends with a short file)
			s_end_full = int(s_end)
			while s_end_full > s_start and int(self.valid_out[s_end_full - 1]) < tb:
				s_end_full -= 1

			if (s_end_full - s_start) < w:
				continue

			b = int(s_start)
			b_last = int(s_end_full - w)

			while b <= b_last:
				# window blocks are guaranteed within one segment here
				x = self.arr[b : b + w, ch0:ch1, :].astype(
					np.float32, copy=False
				)  # (w, Csel, Tb)
				wave = x.transpose(1, 0, 2).reshape(
					(ch1 - ch0, w * tb)
				)  # (Csel, in_samples)

				meta = WindowMeta(
					segment_id=int(seg_id),
					block_start=int(b),
					window_start_utc_ms=int(self.start_ms[b]),
					fs_hz=float(self.fs_hz),
					channels=(int(ch0), int(ch1)),
				)
				yield wave, meta
				b += hb
