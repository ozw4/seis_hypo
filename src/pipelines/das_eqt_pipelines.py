# file: src/pipelines/das_eqt_pipelines.py
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from io_util.zarr_block import ZarrBlockWindowIterator
from pick.ept_runner import EqTWindowRunner
from pick.picks_from_probs import _detect_local_peaks_2d


@dataclass(frozen=True, slots=True)
class DasEqtPickStats:
	windows_processed: int
	picks_written: int


def _ensure_parent(p: Path) -> None:
	p.parent.mkdir(parents=True, exist_ok=True)


def _to_iso_utc(ms: int) -> str:
	return datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc).isoformat()


def pipeline_das_eqt_pick_to_csv(
	*,
	zarr_dir: Path,
	dataset_name: str,
	out_csv: Path,
	eqt_weights: str = 'original',
	eqt_in_samples: int = 6000,
	eqt_overlap: int = 3000,
	eqt_batch_traces: int = 64,
	channel_range: tuple[int, int] | None = None,
	det_gate_enable: bool = True,
	det_threshold: float = 0.30,
	p_threshold: float = 0.10,
	s_threshold: float = 0.10,
	min_pick_sep_samples: int = 50,
	overlap_merge: str = 'max',
	max_windows: int | None = None,
	print_every_windows: int = 50,
) -> DasEqtPickStats:
	"""Run EqTransformer on continuous DAS Zarr blocks and write phase picks to a CSV.

	Assumptions:
	- Zarr dataset is block-aligned: (B, C, Tb)
	- Zarr group contains:
		- starttime_utc_ms: (B,) int64, start time for each block
		- valid_out_samples: (B,) int32, valid samples per block (short blocks mark segment end)
		- segment_id: (B,) int32, segment boundary id
		- attrs['fs_out_hz']: output sampling rate in Hz
	- Window parameters (eqt_in_samples / eqt_overlap) are multiples of Tb.

	Windowing / stitching:
	- Iterate windows of length eqt_in_samples with hop = eqt_in_samples - eqt_overlap.
	- For adjacent windows, overlap region is merged by `overlap_merge` ('max' or 'mean').
	- Picks are produced from hop-sized chunks (except the final flush per segment).

	CSV columns:
		segment_id, block_start, channel, phase, pick_time_utc_ms, pick_time_utc_iso, prob
	"""
	if overlap_merge not in ('max', 'mean'):
		raise ValueError(f"overlap_merge must be 'max' or 'mean', got {overlap_merge}")

	L = int(eqt_in_samples)
	O = int(eqt_overlap)
	H = L - O
	if H <= 0:
		raise ValueError('eqt_overlap must be smaller than eqt_in_samples')

	it = ZarrBlockWindowIterator(
		zarr_dir=Path(zarr_dir),
		dataset_name=str(dataset_name),
		in_samples=int(eqt_in_samples),
		overlap=int(eqt_overlap),
		channel_range=channel_range,
	)
	runner = EqTWindowRunner(
		weights=str(eqt_weights),
		in_samples=int(eqt_in_samples),
		batch_traces=int(eqt_batch_traces),
	)

	ch_base = 0 if channel_range is None else int(channel_range[0])

	_ensure_parent(Path(out_csv))

	windows_processed = 0
	picks_written = 0

	fs_hz: float | None = None

	pending_d: np.ndarray | None = None
	pending_p: np.ndarray | None = None
	pending_s: np.ndarray | None = None
	pending_start_ms: int | None = None
	pending_seg_id: int | None = None
	pending_block_start: int | None = None

	last_p_time_ms: np.ndarray | None = None
	last_p_prob: np.ndarray | None = None
	last_s_time_ms: np.ndarray | None = None
	last_s_prob: np.ndarray | None = None

	def _reset_last(C: int) -> None:
		nonlocal last_p_time_ms, last_p_prob, last_s_time_ms, last_s_prob
		last_p_time_ms = np.full(int(C), -1, dtype=np.int64)
		last_p_prob = np.zeros(int(C), dtype=np.float32)
		last_s_time_ms = np.full(int(C), -1, dtype=np.int64)
		last_s_prob = np.zeros(int(C), dtype=np.float32)

	def _flush_last(wcsv: csv.writer, seg_id: int, block_start: int) -> None:
		nonlocal picks_written
		if last_p_time_ms is None:
			return
		C = int(last_p_time_ms.shape[0])
		for c in range(C):
			tp = int(last_p_time_ms[c])
			if tp >= 0:
				wcsv.writerow(
					[
						int(seg_id),
						int(block_start),
						int(ch_base + c),
						'P',
						tp,
						_to_iso_utc(tp),
						float(last_p_prob[c]),
					]
				)
				picks_written += 1

			ts = int(last_s_time_ms[c])
			if ts >= 0:
				wcsv.writerow(
					[
						int(seg_id),
						int(block_start),
						int(ch_base + c),
						'S',
						ts,
						_to_iso_utc(ts),
						float(last_s_prob[c]),
					]
				)
				picks_written += 1

		last_p_time_ms.fill(-1)
		last_p_prob.fill(0.0)
		last_s_time_ms.fill(-1)
		last_s_prob.fill(0.0)

	def _accumulate_chunk_picks(
		wcsv: csv.writer,
		*,
		seg_id: int,
		block_start: int,
		chunk_d: np.ndarray,
		chunk_p: np.ndarray,
		chunk_s: np.ndarray,
		chunk_start_ms: int,
		chunk_fs_hz: float,
	) -> None:
		"""Accumulate picks from one chunk (C, N) with cross-chunk per-channel de-dup."""
		nonlocal last_p_time_ms, last_p_prob, last_s_time_ms, last_s_prob, picks_written

		C, N = chunk_p.shape
		if last_p_time_ms is None:
			_reset_last(int(C))

		dt_ms = 1000.0 / float(chunk_fs_hz)
		tol_ms = int(round(float(min_pick_sep_samples) * float(dt_ms)))

		gate = chunk_d if bool(det_gate_enable) else None

		p_peaks = _detect_local_peaks_2d(
			prob=chunk_p,
			thr=float(p_threshold),
			min_sep=int(min_pick_sep_samples),
			gate=gate,
			gate_threshold=float(det_threshold),
		)
		s_peaks = _detect_local_peaks_2d(
			prob=chunk_s,
			thr=float(s_threshold),
			min_sep=int(min_pick_sep_samples),
			gate=gate,
			gate_threshold=float(det_threshold),
		)

		def _tidx_to_ms(t_idx: int) -> int:
			return int(round(int(chunk_start_ms) + (float(t_idx) * float(dt_ms))))

		# P
		for c_idx, t_idx, val in p_peaks:
			t_ms = _tidx_to_ms(int(t_idx))
			prev_t = int(last_p_time_ms[c_idx])
			prev_v = float(last_p_prob[c_idx])
			if prev_t < 0:
				last_p_time_ms[c_idx] = int(t_ms)
				last_p_prob[c_idx] = float(val)
				continue
			if int(t_ms) - int(prev_t) <= int(tol_ms):
				if float(val) > float(prev_v):
					last_p_time_ms[c_idx] = int(t_ms)
					last_p_prob[c_idx] = float(val)
				continue

			wcsv.writerow(
				[
					int(seg_id),
					int(block_start),
					int(ch_base + c_idx),
					'P',
					int(prev_t),
					_to_iso_utc(int(prev_t)),
					float(prev_v),
				]
			)
			picks_written += 1
			last_p_time_ms[c_idx] = int(t_ms)
			last_p_prob[c_idx] = float(val)

		# S
		for c_idx, t_idx, val in s_peaks:
			t_ms = _tidx_to_ms(int(t_idx))
			prev_t = int(last_s_time_ms[c_idx])
			prev_v = float(last_s_prob[c_idx])
			if prev_t < 0:
				last_s_time_ms[c_idx] = int(t_ms)
				last_s_prob[c_idx] = float(val)
				continue
			if int(t_ms) - int(prev_t) <= int(tol_ms):
				if float(val) > float(prev_v):
					last_s_time_ms[c_idx] = int(t_ms)
					last_s_prob[c_idx] = float(val)
				continue

			wcsv.writerow(
				[
					int(seg_id),
					int(block_start),
					int(ch_base + c_idx),
					'S',
					int(prev_t),
					_to_iso_utc(int(prev_t)),
					float(prev_v),
				]
			)
			picks_written += 1
			last_s_time_ms[c_idx] = int(t_ms)
			last_s_prob[c_idx] = float(val)

	with Path(out_csv).open('w', newline='', encoding='utf-8') as f:
		wcsv = csv.writer(f)
		wcsv.writerow(
			[
				'segment_id',
				'block_start',
				'channel',
				'phase',
				'pick_time_utc_ms',
				'pick_time_utc_iso',
				'prob',
			]
		)

		prev_seg: int | None = None

		for wave, meta in it:
			if fs_hz is None:
				fs_hz = float(meta.fs_hz)
			elif float(meta.fs_hz) != float(fs_hz):
				raise ValueError(
					f'fs_hz changed within iterator: {meta.fs_hz} vs {fs_hz}'
				)

			if prev_seg is None:
				prev_seg = int(meta.segment_id)

			# segment boundary: flush remaining stitched buffer + last picks
			if int(meta.segment_id) != int(prev_seg):
				if pending_p is not None:
					_accumulate_chunk_picks(
						wcsv,
						seg_id=int(pending_seg_id),
						block_start=int(pending_block_start),
						chunk_d=pending_d
						if pending_d is not None
						else np.zeros_like(pending_p),
						chunk_p=pending_p,
						chunk_s=pending_s
						if pending_s is not None
						else np.zeros_like(pending_p),
						chunk_start_ms=int(pending_start_ms),
						chunk_fs_hz=float(fs_hz),
					)
					_flush_last(wcsv, int(pending_seg_id), int(pending_block_start))

				pending_d = None
				pending_p = None
				pending_s = None
				pending_start_ms = None
				pending_seg_id = None
				pending_block_start = None

				last_p_time_ms = None
				last_p_prob = None
				last_s_time_ms = None
				last_s_prob = None

				prev_seg = int(meta.segment_id)

			# EqT inference for this window across all selected channels
			det_w, p_w, s_w = runner.predict_window(wave)
			if det_w.shape != p_w.shape or det_w.shape != s_w.shape:
				raise ValueError('EqT outputs shape mismatch')
			if det_w.shape[1] != int(L):
				raise ValueError('EqT output length mismatch')

			if pending_p is None:
				pending_d = det_w
				pending_p = p_w
				pending_s = s_w
				pending_start_ms = int(meta.window_start_utc_ms)
				pending_seg_id = int(meta.segment_id)
				pending_block_start = int(meta.block_start)
			else:
				# merge overlap: pending[H:] with current[:O]
				if overlap_merge == 'max':
					pending_d[:, H:L] = np.maximum(pending_d[:, H:L], det_w[:, 0:O])
					pending_p[:, H:L] = np.maximum(pending_p[:, H:L], p_w[:, 0:O])
					pending_s[:, H:L] = np.maximum(pending_s[:, H:L], s_w[:, 0:O])
				else:
					pending_d[:, H:L] = (pending_d[:, H:L] + det_w[:, 0:O]) * 0.5
					pending_p[:, H:L] = (pending_p[:, H:L] + p_w[:, 0:O]) * 0.5
					pending_s[:, H:L] = (pending_s[:, H:L] + s_w[:, 0:O]) * 0.5

				# emit first hop part of pending
				_accumulate_chunk_picks(
					wcsv,
					seg_id=int(pending_seg_id),
					block_start=int(pending_block_start),
					chunk_d=pending_d[:, 0:H],
					chunk_p=pending_p[:, 0:H],
					chunk_s=pending_s[:, 0:H],
					chunk_start_ms=int(pending_start_ms),
					chunk_fs_hz=float(fs_hz),
				)

				# shift: [pending overlap part] + [current tail part]
				pending_d = np.concatenate([pending_d[:, H:L], det_w[:, O:L]], axis=1)
				pending_p = np.concatenate([pending_p[:, H:L], p_w[:, O:L]], axis=1)
				pending_s = np.concatenate([pending_s[:, H:L], s_w[:, O:L]], axis=1)
				pending_start_ms = int(meta.window_start_utc_ms)
				pending_seg_id = int(meta.segment_id)
				pending_block_start = int(meta.block_start)

			windows_processed += 1
			if (
				int(print_every_windows) > 0
				and (windows_processed % int(print_every_windows)) == 0
			):
				print(f'[INFO] windows={windows_processed} picks={picks_written}')

			if max_windows is not None and windows_processed >= int(max_windows):
				break

		# final flush
		if pending_p is not None:
			_accumulate_chunk_picks(
				wcsv,
				seg_id=int(pending_seg_id),
				block_start=int(pending_block_start),
				chunk_d=pending_d
				if pending_d is not None
				else np.zeros_like(pending_p),
				chunk_p=pending_p,
				chunk_s=pending_s
				if pending_s is not None
				else np.zeros_like(pending_p),
				chunk_start_ms=int(pending_start_ms),
				chunk_fs_hz=float(fs_hz),
			)
			_flush_last(wcsv, int(pending_seg_id), int(pending_block_start))

	print(
		f'[DONE] wrote CSV: {out_csv} windows={windows_processed} picks={picks_written}'
	)
	return DasEqtPickStats(
		windows_processed=int(windows_processed),
		picks_written=int(picks_written),
	)
