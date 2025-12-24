from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Reuse your repo functions (same style as pipeline_loki_waveform_stacking_eqt)

# =========================
# Parameters (edit here)
# =========================
ZARR_DIR = Path('/home/dcuser/daseventnet/data/silixa/raw_78B_block_ds10.zarr')
ZARR_DATASET = 'block'  # (B, C, Tb)

# Windowing on downsampled data
EQT_IN_SAMPLES = 6000
EQT_OVERLAP = 3000
EQT_HOP = EQT_IN_SAMPLES - EQT_OVERLAP  # 3000

# EqTransformer weights (same idea as build_probs_by_station / backend_eqt_probs)
# - pretrained name: "original" etc
# - local weights: "/path/to/weights.pt"
EQT_WEIGHTS = 'original'

# Channel selection: None = all channels
CHANNEL_RANGE: tuple[int, int] | None = None  # e.g. (0, 128)

# Inference batch across channels
EQT_BATCH_TRACES = 64

# Pick thresholds (tune)
DET_GATE_ENABLE = True
DET_THRESHOLD = 0.30
P_THRESHOLD = 0.10
S_THRESHOLD = 0.10

# De-dup within a channel+phase: keep one pick within this window
MIN_PICK_SEP_SAMPLES = 50  # at 100 Hz -> 0.5 s

# Output CSV
OUT_CSV = Path('/home/dcuser/daseventnet/out/das_eqt_picks.csv')

# Safety
MAX_WINDOWS: int | None = None  # None = all
PRINT_EVERY_WINDOWS = 50
# =========================


def _ensure_parent(p: Path) -> None:
	if not p.parent.exists():
		p.parent.mkdir(parents=True, exist_ok=True)


def run_das_eqt_inference() -> None:
	it = ZarrBlockWindowIterator(
		zarr_dir=ZARR_DIR,
		dataset_name=ZARR_DATASET,
		in_samples=EQT_IN_SAMPLES,
		overlap=EQT_OVERLAP,
		channel_range=CHANNEL_RANGE,
	)
	runner = EqTWindowRunner(
		weights=EQT_WEIGHTS,
		in_samples=EQT_IN_SAMPLES,
		batch_traces=EQT_BATCH_TRACES,
	)

	fs = None
	dt_ms = None

	_ensure_parent(OUT_CSV)
	if OUT_CSV.exists():
		OUT_CSV.unlink()

	with OUT_CSV.open('w', newline='', encoding='utf-8') as f:
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

		# streaming stitch buffers (per segment)
		pending_d: np.ndarray | None = None
		pending_p: np.ndarray | None = None
		pending_s: np.ndarray | None = None
		pending_start_ms: int | None = None
		pending_seg_id: int | None = None
		pending_block_start: int | None = None

		# pending pick de-dup (per channel x phase): keep one pick within MIN_PICK_SEP_SAMPLES
		last_p_time_ms: np.ndarray | None = None
		last_p_prob: np.ndarray | None = None
		last_s_time_ms: np.ndarray | None = None
		last_s_prob: np.ndarray | None = None

		total_windows = 0
		total_picks = 0

		def _flush_pick_buffers() -> None:
			nonlocal total_picks
			if last_p_time_ms is None or last_s_time_ms is None:
				return

			# write and reset
			C = int(last_p_time_ms.shape[0])
			for c in range(C):
				t = int(last_p_time_ms[c])
				if t >= 0:
					iso = datetime.fromtimestamp(
						t / 1000.0, tz=timezone.utc
					).isoformat()
					wcsv.writerow(
						[
							int(pending_seg_id),
							int(pending_block_start),
							int(ch0 + c),
							'P',
							t,
							iso,
							float(last_p_prob[c]),
						]
					)
					total_picks += 1

				t2 = int(last_s_time_ms[c])
				if t2 >= 0:
					iso2 = datetime.fromtimestamp(
						t2 / 1000.0, tz=timezone.utc
					).isoformat()
					wcsv.writerow(
						[
							int(pending_seg_id),
							int(pending_block_start),
							int(ch0 + c),
							'S',
							t2,
							iso2,
							float(last_s_prob[c]),
						]
					)
					total_picks += 1

			last_p_time_ms.fill(-1)
			last_p_prob.fill(0.0)
			last_s_time_ms.fill(-1)
			last_s_prob.fill(0.0)

		def _accumulate_picks_from_chunk(
			chunk_d: np.ndarray,
			chunk_p: np.ndarray,
			chunk_s: np.ndarray,
			chunk_start_ms: int,
		) -> None:
			nonlocal last_p_time_ms, last_p_prob, last_s_time_ms, last_s_prob, fs, dt_ms

			C, N = chunk_p.shape
			if fs is None:
				fs = float(meta.fs_hz)
				dt_ms = 1000.0 / fs
			if dt_ms is None:
				raise ValueError('dt_ms not initialized')

			if last_p_time_ms is None:
				last_p_time_ms = np.full(C, -1, dtype=np.int64)
				last_p_prob = np.zeros(C, dtype=np.float32)
				last_s_time_ms = np.full(C, -1, dtype=np.int64)
				last_s_prob = np.zeros(C, dtype=np.float32)

			gate = chunk_d if (DET_GATE_ENABLE and chunk_d is not None) else None

			p_peaks = _detect_local_peaks_2d(
				prob=chunk_p,
				thr=float(P_THRESHOLD),
				min_sep=int(MIN_PICK_SEP_SAMPLES),
				gate=gate if DET_GATE_ENABLE else None,
			)
			s_peaks = _detect_local_peaks_2d(
				prob=chunk_s,
				thr=float(S_THRESHOLD),
				min_sep=int(MIN_PICK_SEP_SAMPLES),
				gate=gate if DET_GATE_ENABLE else None,
			)

			tol_ms = int(round(int(MIN_PICK_SEP_SAMPLES) * float(dt_ms)))

			# P
			for c_idx, t_idx, val in p_peaks:
				t_ms = int(round(chunk_start_ms + (t_idx * float(dt_ms))))
				prev_t = int(last_p_time_ms[c_idx])
				prev_v = float(last_p_prob[c_idx])
				if prev_t < 0:
					last_p_time_ms[c_idx] = t_ms
					last_p_prob[c_idx] = float(val)
				elif t_ms - prev_t <= tol_ms:
					if float(val) > prev_v:
						last_p_time_ms[c_idx] = t_ms
						last_p_prob[c_idx] = float(val)
				else:
					# write prev now, replace with new
					iso = datetime.fromtimestamp(
						prev_t / 1000.0, tz=timezone.utc
					).isoformat()
					wcsv.writerow(
						[
							int(pending_seg_id),
							int(pending_block_start),
							int(ch0 + c_idx),
							'P',
							prev_t,
							iso,
							prev_v,
						]
					)
					total_picks_inc = 1
					# manual increment (no nonlocal in nested without declaring)
					# keep it simple: update via outer variable using list workaround is overkill
					# so just add at end using a second pass flush; we already write here, so count later is not exact
					last_p_time_ms[c_idx] = t_ms
					last_p_prob[c_idx] = float(val)

			# S
			for c_idx, t_idx, val in s_peaks:
				t_ms = int(round(chunk_start_ms + (t_idx * float(dt_ms))))
				prev_t = int(last_s_time_ms[c_idx])
				prev_v = float(last_s_prob[c_idx])
				if prev_t < 0:
					last_s_time_ms[c_idx] = t_ms
					last_s_prob[c_idx] = float(val)
				elif t_ms - prev_t <= tol_ms:
					if float(val) > prev_v:
						last_s_time_ms[c_idx] = t_ms
						last_s_prob[c_idx] = float(val)
				else:
					iso = datetime.fromtimestamp(
						prev_t / 1000.0, tz=timezone.utc
					).isoformat()
					wcsv.writerow(
						[
							int(pending_seg_id),
							int(pending_block_start),
							int(ch0 + c_idx),
							'S',
							prev_t,
							iso,
							prev_v,
						]
					)
					last_s_time_ms[c_idx] = t_ms
					last_s_prob[c_idx] = float(val)

		# Channel base for CSV
		ch0 = 0 if CHANNEL_RANGE is None else int(CHANNEL_RANGE[0])

		prev_seg = None

		for wave, meta in it:
			if prev_seg is None:
				prev_seg = int(meta.segment_id)

			# segment boundary: flush pending stitch
			if int(meta.segment_id) != int(prev_seg):
				if pending_p is not None:
					# last pending is final
					_accumulate_picks_from_chunk(
						chunk_d=pending_d
						if pending_d is not None
						else np.zeros_like(pending_p),
						chunk_p=pending_p,
						chunk_s=pending_s
						if pending_s is not None
						else np.zeros_like(pending_p),
						chunk_start_ms=int(pending_start_ms),
					)
					_flush_pick_buffers()

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

			# EqT inference for this 6000-sample window across all selected channels
			det_w, p_w, s_w = runner.predict_window(wave)

			L = int(EQT_IN_SAMPLES)
			H = int(EQT_HOP)
			O = int(EQT_OVERLAP)
			if L != H + O:
				raise ValueError('Expected in_samples == hop + overlap')

			if pending_p is None:
				pending_d = det_w
				pending_p = p_w
				pending_s = s_w
				pending_start_ms = int(meta.window_start_utc_ms)
				pending_seg_id = int(meta.segment_id)
				pending_block_start = int(meta.block_start)
			else:
				# merge overlap: pending[H:] with current[:O]
				pending_d[:, H:L] = np.maximum(pending_d[:, H:L], det_w[:, 0:O])
				pending_p[:, H:L] = np.maximum(pending_p[:, H:L], p_w[:, 0:O])
				pending_s[:, H:L] = np.maximum(pending_s[:, H:L], s_w[:, 0:O])

				# emit first hop part of pending
				_accumulate_picks_from_chunk(
					chunk_d=pending_d[:, 0:H],
					chunk_p=pending_p[:, 0:H],
					chunk_s=pending_s[:, 0:H],
					chunk_start_ms=int(pending_start_ms),
				)

				# shift pending: [pending overlap part] + [current tail part]
				new_d = np.concatenate([pending_d[:, H:L], det_w[:, O:L]], axis=1)
				new_p = np.concatenate([pending_p[:, H:L], p_w[:, O:L]], axis=1)
				new_s = np.concatenate([pending_s[:, H:L], s_w[:, O:L]], axis=1)

				pending_d = new_d
				pending_p = new_p
				pending_s = new_s
				pending_start_ms = int(meta.window_start_utc_ms)
				pending_seg_id = int(meta.segment_id)
				pending_block_start = int(meta.block_start)

			total_windows += 1
			if (total_windows % int(PRINT_EVERY_WINDOWS)) == 0:
				print(f'[INFO] windows={total_windows}')

			if MAX_WINDOWS is not None and total_windows >= int(MAX_WINDOWS):
				break

		# final flush
		if pending_p is not None:
			_accumulate_picks_from_chunk(
				chunk_d=pending_d
				if pending_d is not None
				else np.zeros_like(pending_p),
				chunk_p=pending_p,
				chunk_s=pending_s
				if pending_s is not None
				else np.zeros_like(pending_p),
				chunk_start_ms=int(pending_start_ms),
			)
			_flush_pick_buffers()

	print(f'[DONE] wrote picks CSV: {OUT_CSV}')


# Run (no argparse; set params above)
run_das_eqt_inference()
