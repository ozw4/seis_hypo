# file: src/pipelines/das_pick_csv_accumulator.py
from __future__ import annotations

import csv
from dataclasses import dataclass, field

import numpy as np

from common.time_util import utc_ms_to_iso
from pick.picks_from_probs import _detect_local_peaks_2d


@dataclass(slots=True)
class PickAccumulator:
	channel_range: tuple[int, int] | None
	channel_ids: np.ndarray | None
	min_pick_sep_samples: int
	p_threshold: float
	s_threshold: float
	det_gate_enable: bool
	det_threshold: float
	last_p_time_ms: np.ndarray | None = field(default=None, init=False)
	last_p_prob: np.ndarray | None = field(default=None, init=False)
	last_s_time_ms: np.ndarray | None = field(default=None, init=False)
	last_s_prob: np.ndarray | None = field(default=None, init=False)

	def reset(self) -> None:
		self.last_p_time_ms = None
		self.last_p_prob = None
		self.last_s_time_ms = None
		self.last_s_prob = None

	def _ensure_last(self, channels: int) -> None:
		if self.last_p_time_ms is not None:
			return
		self.last_p_time_ms = np.full(int(channels), -1, dtype=np.int64)
		self.last_p_prob = np.zeros(int(channels), dtype=np.float32)
		self.last_s_time_ms = np.full(int(channels), -1, dtype=np.int64)
		self.last_s_prob = np.zeros(int(channels), dtype=np.float32)

	def _chan_id(self, channel_index: int) -> int:
		if self.channel_ids is None:
			base = 0 if self.channel_range is None else int(self.channel_range[0])
			return int(base + channel_index)
		return int(self.channel_ids[channel_index])

	def flush(self, wcsv: csv.writer, seg_id: int, block_start: int) -> int:
		if self.last_p_time_ms is None:
			return 0
		count = 0
		channels = int(self.last_p_time_ms.shape[0])
		for c in range(channels):
			tp = int(self.last_p_time_ms[c])
			if tp >= 0:
				wcsv.writerow(
					[
						int(seg_id),
						int(block_start),
						self._chan_id(c),
						'P',
						int(tp),
						utc_ms_to_iso(int(tp)),
						float(self.last_p_prob[c]),
					]
				)
				count += 1

			ts = int(self.last_s_time_ms[c])
			if ts >= 0:
				wcsv.writerow(
					[
						int(seg_id),
						int(block_start),
						self._chan_id(c),
						'S',
						int(ts),
						utc_ms_to_iso(int(ts)),
						float(self.last_s_prob[c]),
					]
				)
				count += 1

		self.last_p_time_ms.fill(-1)
		self.last_p_prob.fill(0.0)
		self.last_s_time_ms.fill(-1)
		self.last_s_prob.fill(0.0)
		return int(count)

	def accumulate_chunk(
		self,
		wcsv: csv.writer,
		*,
		seg_id: int,
		block_start: int,
		chunk_p: np.ndarray,
		chunk_s: np.ndarray,
		chunk_start_ms: int,
		chunk_fs_hz: float,
		chunk_d: np.ndarray | None = None,
	) -> int:
		channels, _samples = chunk_p.shape
		self._ensure_last(int(channels))
		dt_ms = 1000.0 / float(chunk_fs_hz)
		tol_ms = int(round(float(self.min_pick_sep_samples) * float(dt_ms)))
		gate = chunk_d if bool(self.det_gate_enable) else None

		p_peaks = _detect_local_peaks_2d(
			prob=chunk_p,
			thr=float(self.p_threshold),
			min_sep=int(self.min_pick_sep_samples),
			gate=gate,
			gate_threshold=float(self.det_threshold),
		)
		s_peaks = _detect_local_peaks_2d(
			prob=chunk_s,
			thr=float(self.s_threshold),
			min_sep=int(self.min_pick_sep_samples),
			gate=gate,
			gate_threshold=float(self.det_threshold),
		)

		def _tidx_to_ms(t_idx: int) -> int:
			return int(round(int(chunk_start_ms) + (float(t_idx) * float(dt_ms))))

		count = 0
		for c_idx, t_idx, val in p_peaks:
			t_ms = _tidx_to_ms(int(t_idx))
			prev_t = int(self.last_p_time_ms[c_idx])
			prev_v = float(self.last_p_prob[c_idx])
			if prev_t < 0:
				self.last_p_time_ms[c_idx] = int(t_ms)
				self.last_p_prob[c_idx] = float(val)
				continue
			if int(t_ms) - int(prev_t) <= int(tol_ms):
				if float(val) > float(prev_v):
					self.last_p_time_ms[c_idx] = int(t_ms)
					self.last_p_prob[c_idx] = float(val)
				continue
			wcsv.writerow(
				[
					int(seg_id),
					int(block_start),
					self._chan_id(int(c_idx)),
					'P',
					int(prev_t),
					utc_ms_to_iso(int(prev_t)),
					float(prev_v),
				]
			)
			count += 1
			self.last_p_time_ms[c_idx] = int(t_ms)
			self.last_p_prob[c_idx] = float(val)

		for c_idx, t_idx, val in s_peaks:
			t_ms = _tidx_to_ms(int(t_idx))
			prev_t = int(self.last_s_time_ms[c_idx])
			prev_v = float(self.last_s_prob[c_idx])
			if prev_t < 0:
				self.last_s_time_ms[c_idx] = int(t_ms)
				self.last_s_prob[c_idx] = float(val)
				continue
			if int(t_ms) - int(prev_t) <= int(tol_ms):
				if float(val) > float(prev_v):
					self.last_s_time_ms[c_idx] = int(t_ms)
					self.last_s_prob[c_idx] = float(val)
				continue
			wcsv.writerow(
				[
					int(seg_id),
					int(block_start),
					self._chan_id(int(c_idx)),
					'S',
					int(prev_t),
					utc_ms_to_iso(int(prev_t)),
					float(prev_v),
				]
			)
			count += 1
			self.last_s_time_ms[c_idx] = int(t_ms)
			self.last_s_prob[c_idx] = float(val)

		return int(count)
