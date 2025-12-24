from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr


@dataclass(frozen=True)
class WindowMeta:
	segment_id: int
	block_start: int
	starttime_utc_ms: int
	fs_hz: float
	channels: tuple[int, int]  # [start, end) in channel axis
	window_samples: int
	hop_samples: int


class ZarrBlockWindowIterator:
	"""Iterate windows from a Zarr store with layout (B, C, Tb).
	Designed for block-aligned windows where window_samples and hop_samples are multiples of Tb.

	Safety:
	- Windows never cross segment_id boundaries.
	- Windows containing any short block (valid_out_samples < Tb) are skipped by default.

	Output:
	- wave: np.ndarray shape (Csel, window_samples) float32
	- meta: WindowMeta
	"""

	def __init__(
		self,
		zarr_dir: Path,
		dataset_name: str = 'block',
		window_samples: int = 6000,
		hop_samples: int = 3000,
		channel_range: tuple[int, int] | None = None,  # [start, end)
		skip_short_blocks: bool = True,
		require_full_blocks: bool = True,
	):
		self.zarr_dir = Path(zarr_dir)
		self.root = zarr.open_group(str(self.zarr_dir), mode='r')

		if dataset_name not in self.root:
			raise ValueError(
				f"Dataset '{dataset_name}' not found in Zarr: {self.zarr_dir}"
			)

		self.arr = self.root[dataset_name]

		if self.arr.ndim != 3:
			raise ValueError(f'Expected 3D array (B,C,Tb), got ndim={self.arr.ndim}')

		self.n_blocks, self.n_ch, self.tb = self.arr.shape

		for key in ('starttime_utc_ms', 'valid_out_samples', 'segment_id'):
			if key not in self.root:
				raise ValueError(
					f"Missing required dataset '{key}' in Zarr: {self.zarr_dir}"
				)

		self.start_ms = self.root['starttime_utc_ms'][:].astype(np.int64, copy=False)
		self.valid_out = self.root['valid_out_samples'][:].astype(np.int32, copy=False)
		self.seg = self.root['segment_id'][:].astype(np.int32, copy=False)

		if self.start_ms.shape != (self.n_blocks,):
			raise ValueError('starttime_utc_ms shape mismatch')
		if self.valid_out.shape != (self.n_blocks,):
			raise ValueError('valid_out_samples shape mismatch')
		if self.seg.shape != (self.n_blocks,):
			raise ValueError('segment_id shape mismatch')

		self.window_samples = int(window_samples)
		self.hop_samples = int(hop_samples)

		if self.window_samples <= 0 or self.hop_samples <= 0:
			raise ValueError('window_samples and hop_samples must be positive')

		if (self.window_samples % self.tb) != 0 or (self.hop_samples % self.tb) != 0:
			raise ValueError(
				f'window_samples ({self.window_samples}) and hop_samples ({self.hop_samples}) '
				f'must be multiples of Tb ({self.tb}) for block-aligned iteration'
			)

		self.w_blocks = self.window_samples // self.tb
		self.h_blocks = self.hop_samples // self.tb

		if self.w_blocks < 1 or self.h_blocks < 1:
			raise ValueError('Invalid block counts derived from window/hop')

		if channel_range is None:
			self.ch0, self.ch1 = 0, self.n_ch
		else:
			self.ch0, self.ch1 = int(channel_range[0]), int(channel_range[1])
			if not (0 <= self.ch0 < self.ch1 <= self.n_ch):
				raise ValueError(
					f'Invalid channel_range={channel_range} for n_ch={self.n_ch}'
				)

		self.skip_short_blocks = bool(skip_short_blocks)
		self.require_full_blocks = bool(require_full_blocks)

		fs_out = self.root.attrs.get('fs_out_hz', None)
		if fs_out is None:
			fs_in = self.root.attrs.get('fs_in_hz', None)
			if fs_in is None:
				raise ValueError('Missing fs_out_hz (or fs_in_hz) in Zarr attrs')
			fs_out = float(fs_in)
		self.fs_hz = float(fs_out)

	def _segment_ranges(self) -> Iterator[tuple[int, int, int]]:
		"""Yield (seg_id, start_block, end_block_exclusive) for each contiguous segment run.
		Assumes seg array is aligned with time and typically non-decreasing.
		"""
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
		h = self.h_blocks
		ch0, ch1 = self.ch0, self.ch1

		for seg_id, s_start, s_end in self._segment_ranges():
			n_in_seg = s_end - s_start
			if n_in_seg < w:
				continue

			b = s_start
			b_last_start = s_end - w

			while b <= b_last_start:
				blocks = np.arange(b, b + w, dtype=np.int64)

				if self.skip_short_blocks:
					if np.any(self.valid_out[blocks] < tb):
						b += h
						continue
				elif self.require_full_blocks:
					if np.any(self.valid_out[blocks] != tb):
						b += h
						continue

				x = self.arr[b : b + w, ch0:ch1, :].astype(np.float32, copy=False)
				# x: (w, Csel, Tb) -> (Csel, w, Tb) -> (Csel, w*Tb)
				wave = x.transpose(1, 0, 2).reshape((ch1 - ch0, w * tb))

				meta = WindowMeta(
					segment_id=int(seg_id),
					block_start=int(b),
					starttime_utc_ms=int(self.start_ms[b]),
					fs_hz=self.fs_hz,
					channels=(int(ch0), int(ch1)),
					window_samples=int(self.window_samples),
					hop_samples=int(self.hop_samples),
				)
				yield wave, meta

				b += h
