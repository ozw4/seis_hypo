# file: src/pipelines/das_phasenet_pipelines.py
from __future__ import annotations

import csv
from fractions import Fraction
from pathlib import Path

import numpy as np
import zarr

from common.core import as_int_rate
from common.time_util import utc_ms_to_iso
from io_util.zarr_block import ZarrBlockWindowIterator
from pick.phasenet_runner import PhaseNetWindowRunner
from pick.picks_from_probs import _detect_local_peaks_2d
from pipelines.das_eqt_pipelines import (
	DasEqtPickStats,
	_ab_keep_indices_and_channel_ids,
	_ensure_parent,
)
from waveform.preprocess import (
	bandpass_window,
	resample_window_poly,
	strainrate_to_pseudovel,
)


def pipeline_das_phasenet_pick_to_csv(
	*,
	zarr_dir: Path,
	dataset_name: str,
	out_csv: Path,
	phasenet_weights: str = 'instance',
	phasenet_in_samples: int = 3001,
	phasenet_overlap: int = 1500,
	phasenet_batch_traces: int = 256,
	channel_range: tuple[int, int] | None = None,
	p_threshold: float = 0.10,
	s_threshold: float = 0.10,
	min_pick_sep_samples: int = 50,
	overlap_merge: str = 'max',
	max_windows: int | None = None,
	print_every_windows: int = 50,
	# ---- discontiguous A/B keep ----
	apply_well_ab_keep: bool = False,
	well_a_keep_0based_incl: tuple[int, int] = (92, 1062),
	well_b_keep_0based_incl: tuple[int, int] = (1216, 2385),
	# ---- conversion / preprocess ----
	convert_strainrate_to_pseudovel: bool = False,
	target_fs_hz: float = 100.0,
	post_bp_low_hz: float = 25.0,
	post_bp_high_hz: float = 45.0,
	post_bp_order: int = 4,
	pseudovel_scale: float = 1.0,
) -> DasEqtPickStats:
	"""PhaseNet版（DAS連続→CSV）。
	注意: det_gate_enable は非対応（P/Sのみでピーク検出）。
	"""
	if overlap_merge not in ('max', 'mean'):
		raise ValueError(f"overlap_merge must be 'max' or 'mean', got {overlap_merge}")

	L = int(phasenet_in_samples)
	O = int(phasenet_overlap)
	H = L - O
	if H <= 0:
		raise ValueError('phasenet_overlap must be smaller than phasenet_in_samples')

	# Zarr info (needed to align iterator windows to Tb)
	root = zarr.open_group(str(Path(zarr_dir)), mode='r')
	block = root[str(dataset_name)]
	tb = int(block.shape[2])
	fs_zarr = float(root.attrs['fs_out_hz'])
	fi = as_int_rate(fs_zarr, 'zarr fs_out_hz')
	fo = as_int_rate(target_fs_hz, 'target_fs_hz')
	apply_resample = int(fi) != int(fo)
	fs_used = float(fo) if bool(apply_resample) else float(fi)

	in_samples_zarr = int(L)
	overlap_zarr = int(O)
	if bool(apply_resample):
		r = Fraction(fi, fo)  # e.g., 250/100 = 5/2
		if (int(L) * int(r.numerator)) % int(r.denominator) != 0:
			raise ValueError(
				'phasenet_in_samples cannot be converted to an integer-length Zarr window'
			)
		if (int(O) * int(r.numerator)) % int(r.denominator) != 0:
			raise ValueError(
				'phasenet_overlap cannot be converted to an integer-length Zarr overlap'
			)
		in_samples_zarr = (int(L) * int(r.numerator)) // int(r.denominator)
		overlap_zarr = (int(O) * int(r.numerator)) // int(r.denominator)

	hop_zarr = int(in_samples_zarr) - int(overlap_zarr)
	if hop_zarr <= 0:
		raise ValueError('overlap_zarr must be smaller than in_samples_zarr')
	if (int(in_samples_zarr) % int(tb)) != 0 or (int(hop_zarr) % int(tb)) != 0:
		raise ValueError(
			f'in_samples_zarr ({in_samples_zarr}) and hop_zarr ({hop_zarr}) '
			f'must be multiples of Tb ({tb}).'
		)

	keep_idx: np.ndarray | None = None
	channel_ids: np.ndarray | None = None
	if bool(apply_well_ab_keep):
		keep_idx, channel_ids = _ab_keep_indices_and_channel_ids(
			root=root,
			well_a_keep_0based_incl=tuple(well_a_keep_0based_incl),
			well_b_keep_0based_incl=tuple(well_b_keep_0based_incl),
		)

	it = ZarrBlockWindowIterator(
		zarr_dir=Path(zarr_dir),
		dataset_name=str(dataset_name),
		in_samples=int(in_samples_zarr),
		overlap=int(overlap_zarr),
		channel_range=channel_range,
	)

	runner = PhaseNetWindowRunner(
		weights=str(phasenet_weights),
		in_samples=int(phasenet_in_samples),
		batch_traces=int(phasenet_batch_traces),
	)

	_ensure_parent(Path(out_csv))

	windows_processed = 0
	picks_written = 0

	fs_hz: float | None = None

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

	def _chan_id(c: int) -> int:
		if channel_ids is None:
			base = 0 if channel_range is None else int(channel_range[0])
			return int(base + c)
		return int(channel_ids[c])

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
						_chan_id(c),
						'P',
						tp,
						utc_ms_to_iso(tp),
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
						_chan_id(c),
						'S',
						ts,
						utc_ms_to_iso(ts),
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
		chunk_p: np.ndarray,
		chunk_s: np.ndarray,
		chunk_start_ms: int,
		chunk_fs_hz: float,
	) -> None:
		nonlocal last_p_time_ms, last_p_prob, last_s_time_ms, last_s_prob, picks_written

		C, _N = chunk_p.shape
		if last_p_time_ms is None:
			_reset_last(int(C))

		dt_ms = 1000.0 / float(chunk_fs_hz)
		tol_ms = int(round(float(min_pick_sep_samples) * float(dt_ms)))

		p_peaks = _detect_local_peaks_2d(
			prob=chunk_p,
			thr=float(p_threshold),
			min_sep=int(min_pick_sep_samples),
			gate=None,
		)
		s_peaks = _detect_local_peaks_2d(
			prob=chunk_s,
			thr=float(s_threshold),
			min_sep=int(min_pick_sep_samples),
			gate=None,
		)

		def _tidx_to_ms(t_idx: int) -> int:
			return int(round(int(chunk_start_ms) + (float(t_idx) * float(dt_ms))))

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
					_chan_id(int(c_idx)),
					'P',
					int(prev_t),
					utc_ms_to_iso(int(prev_t)),
					float(prev_v),
				]
			)
			picks_written += 1
			last_p_time_ms[c_idx] = int(t_ms)
			last_p_prob[c_idx] = float(val)

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
					_chan_id(int(c_idx)),
					'S',
					int(prev_t),
					utc_ms_to_iso(int(prev_t)),
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

			if int(meta.segment_id) != int(prev_seg):
				if pending_p is not None:
					_accumulate_chunk_picks(
						wcsv,
						seg_id=int(pending_seg_id),
						block_start=int(pending_block_start),
						chunk_p=pending_p,
						chunk_s=pending_s
						if pending_s is not None
						else np.zeros_like(pending_p),
						chunk_start_ms=int(pending_start_ms),
						chunk_fs_hz=float(fs_used),
					)
					_flush_last(wcsv, int(pending_seg_id), int(pending_block_start))

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

			wave_in = np.asarray(wave, dtype=np.float32)

			if keep_idx is not None:
				wave_in = wave_in[np.asarray(keep_idx, dtype=np.int32), :]

			if bool(convert_strainrate_to_pseudovel):
				wave_in = strainrate_to_pseudovel(
					wave_in,
					fs_in=float(fi),
					pseudovel_scale=float(pseudovel_scale),
				)

			if bool(apply_resample):
				wave_in = resample_window_poly(
					wave_in,
					fs_in=float(fi),
					fs_out=float(fo),
					out_len=int(L),
				)

			wave_in = bandpass_window(
				wave_in,
				fs=float(fs_used),
				post_bp_low_hz=float(post_bp_low_hz),
				post_bp_high_hz=float(post_bp_high_hz),
				post_bp_order=int(post_bp_order),
			)
			if int(wave_in.shape[1]) != int(L):
				raise ValueError(
					f'wave length mismatch: got {wave_in.shape[1]}, expected {L}'
				)

			_det_w, p_w, s_w = runner.predict_window(wave_in)

			if pending_p is None:
				pending_p = p_w
				pending_s = s_w
				pending_start_ms = int(meta.window_start_utc_ms)
				pending_seg_id = int(meta.segment_id)
				pending_block_start = int(meta.block_start)
			else:
				if overlap_merge == 'max':
					pending_p[:, H:L] = np.maximum(pending_p[:, H:L], p_w[:, 0:O])
					pending_s[:, H:L] = np.maximum(pending_s[:, H:L], s_w[:, 0:O])
				else:
					pending_p[:, H:L] = (pending_p[:, H:L] + p_w[:, 0:O]) * 0.5
					pending_s[:, H:L] = (pending_s[:, H:L] + s_w[:, 0:O]) * 0.5

				_accumulate_chunk_picks(
					wcsv,
					seg_id=int(pending_seg_id),
					block_start=int(pending_block_start),
					chunk_p=pending_p[:, 0:H],
					chunk_s=pending_s[:, 0:H],
					chunk_start_ms=int(pending_start_ms),
					chunk_fs_hz=float(fs_used),
				)

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

		if pending_p is not None:
			_accumulate_chunk_picks(
				wcsv,
				seg_id=int(pending_seg_id),
				block_start=int(pending_block_start),
				chunk_p=pending_p,
				chunk_s=pending_s
				if pending_s is not None
				else np.zeros_like(pending_p),
				chunk_start_ms=int(pending_start_ms),
				chunk_fs_hz=float(fs_used),
			)
			_flush_last(wcsv, int(pending_seg_id), int(pending_block_start))

	print(
		f'[DONE] wrote CSV: {out_csv} windows={windows_processed} picks={picks_written}'
	)
	return DasEqtPickStats(
		windows_processed=int(windows_processed), picks_written=int(picks_written)
	)
