# %%
# file: proc/pick_continuous/forge/debug_plot_phasenet_60s_from_tdms_start.py
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr

from common.core import as_int_rate
from common.time_util import utc_ms_to_iso
from io_util.forge_debug_plot_util import (
	ab_keep_idx_and_ids,
	all_channel_ids_from_slices,
	concat_blocks,
)
from io_util.forge_temp_zarr_from_tdms import make_temp_zarr_window_from_tdms_start
from pick.phasenet_runner import PhaseNetWindowRunner
from waveform.preprocess import (
	bandpass_window,
	resample_window_poly,
	strainrate_to_pseudovel,
)

# =========================
# Parameters (edit here)
# =========================
BIG_ZARR_DIR = Path(
	'/home/dcuser/daseventnet/data/silixa/forge_dfit_block_78AB_250Hz.zarr'
)
ZARR_DATASET = 'block'

TDMS_FILE_NAME = 'FORGE_DFIT_UTC_20220417_105941.202.tdms'
MATCH_MODE = 'exact'  # 'exact' or 'endswith'
TDMS_OCCURRENCE = 0  # 同名が複数ヒットしたとき何個目を起点にするか（0-based）

# channel selection
CHANNEL_RANGE: tuple[int, int] | None = None  # contiguous-only in stored channel axis
APPLY_WELL_AB_KEEP = True
WELL_A_KEEP_0BASED_INCL = (92, 1062)
WELL_B_KEEP_0BASED_INCL = (1216, 2385)

# preprocessing
CONVERT_STRAINRATE_TO_PSEUDOVEL = False
PSEUDOVEL_SCALE = 1.0
TARGET_FS_HZ = 100.0
POST_BP_LOW_HZ = 25.0
POST_BP_HIGH_HZ = 45.0
POST_BP_ORDER = 4

# window length to visualize (60s at 100Hz)
WINDOW_SECONDS = 60.0
WINDOW_IN_SAMPLES = int(round(WINDOW_SECONDS * TARGET_FS_HZ))

# PhaseNet windowing (model expects fixed in_samples)
PHASENET_WEIGHTS = 'instance'  # or local .pth/.pt
PHASENET_IN_SAMPLES = 3001
PHASENET_OVERLAP = 1500
PHASENET_BATCH_TRACES = 256
OVERLAP_MERGE = 'max'  # 'max' or 'mean'

# overlay
P_CONTOUR_LEVELS = (0.3, 0.5, 0.7)
S_CONTOUR_LEVELS = (0.3, 0.5, 0.7)

TMP_PARENT_DIR = Path('/tmp')
# =========================


def _phasenet_predict_full_window(
	*,
	runner: PhaseNetWindowRunner,
	wave_100: np.ndarray,  # (C, N)
	in_samples: int,
	overlap: int,
	overlap_merge: str,
) -> tuple[np.ndarray, np.ndarray]:
	if overlap_merge not in ('max', 'mean'):
		raise ValueError(f"OVERLAP_MERGE must be 'max' or 'mean', got {overlap_merge}")

	C, N = int(wave_100.shape[0]), int(wave_100.shape[1])
	L = int(in_samples)
	O = int(overlap)
	H = int(L - O)
	if H <= 0:
		raise ValueError('PHASENET_OVERLAP must be smaller than PHASENET_IN_SAMPLES')
	if N <= 0:
		raise ValueError('empty wave_100')

	if N < L:
		pad = np.zeros((C, L), dtype=np.float32)
		pad[:, :N] = wave_100.astype(np.float32, copy=False)
		_, p_w, s_w = runner.predict_window(pad)
		return p_w[:, :N], s_w[:, :N]

	starts = list(range(0, N - L + 1, H))
	last = int(N - L)
	if starts[-1] != last:
		starts.append(last)

	p_all = np.zeros((C, N), dtype=np.float32)
	s_all = np.zeros((C, N), dtype=np.float32)
	filled = np.zeros((N,), dtype=bool)

	for st in starts:
		w = wave_100[:, st : st + L].astype(np.float32, copy=False)
		_, p_w, s_w = runner.predict_window(w)

		sl = slice(int(st), int(st + L))
		m = filled[sl]

		if not bool(m.any()):
			p_all[:, sl] = p_w
			s_all[:, sl] = s_w
		else:
			idx_new = np.flatnonzero(~m)
			if idx_new.size > 0:
				g = int(st) + idx_new
				p_all[:, g] = p_w[:, idx_new]
				s_all[:, g] = s_w[:, idx_new]

			idx_ov = np.flatnonzero(m)
			if idx_ov.size > 0:
				g = int(st) + idx_ov
				if overlap_merge == 'max':
					p_all[:, g] = np.maximum(p_all[:, g], p_w[:, idx_ov])
					s_all[:, g] = np.maximum(s_all[:, g], s_w[:, idx_ov])
				else:
					p_all[:, g] = (p_all[:, g] + p_w[:, idx_ov]) * 0.5
					s_all[:, g] = (s_all[:, g] + s_w[:, idx_ov]) * 0.5

		filled[sl] = True

	return p_all, s_all


big = zarr.open_group(str(BIG_ZARR_DIR), mode='r')
fs_zarr = float(big.attrs['fs_out_hz'])

fi = as_int_rate(fs_zarr, 'fs_zarr')
fo = as_int_rate(TARGET_FS_HZ, 'target_fs_hz')
if (int(WINDOW_IN_SAMPLES) * int(fi)) % int(fo) != 0:
	raise ValueError(
		'WINDOW_IN_SAMPLES*fs_zarr must be divisible by target_fs: '
		f'L={WINDOW_IN_SAMPLES} fi={fi} fo={fo}'
	)
n_zarr = (int(WINDOW_IN_SAMPLES) * int(fi)) // int(fo)

ts = datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%SZ')
tmp_dir = TMP_PARENT_DIR / f'tmp_debug_phasenet_window_{ts}.zarr'

tmp_zarr_dir, first_window_start_ms, n_zarr, fs_zarr = (
	make_temp_zarr_window_from_tdms_start(
		root=big,
		dataset_name=ZARR_DATASET,
		tdms_file_name=TDMS_FILE_NAME,
		match_mode=MATCH_MODE,
		tdms_occurrence=TDMS_OCCURRENCE,
		out_dir=tmp_dir,
		n_zarr=int(n_zarr),
	)
)

root = zarr.open_group(str(tmp_zarr_dir), mode='r')
block = root[ZARR_DATASET]  # (B,C,Tb)
valid = np.asarray(root['valid_out_samples'][:], dtype=np.int32)

blk = np.asarray(block[:, :, :], dtype=np.float32)

if APPLY_WELL_AB_KEEP:
	keep_idx, channel_ids = ab_keep_idx_and_ids(
		root,
		well_a_keep_0based_incl=WELL_A_KEEP_0BASED_INCL,
		well_b_keep_0based_incl=WELL_B_KEEP_0BASED_INCL,
	)
	blk = blk[:, keep_idx, :]
elif CHANNEL_RANGE is not None:
	c0, c1 = int(CHANNEL_RANGE[0]), int(CHANNEL_RANGE[1])
	blk = blk[:, c0:c1, :]
	channel_ids = np.arange(c0, c1, dtype=np.int32)
else:
	channel_ids = all_channel_ids_from_slices(root)

wave_zarr = concat_blocks(blk, valid)  # (C, n_zarr)
if wave_zarr.shape[1] != int(n_zarr):
	raise ValueError(
		f'wave_zarr length mismatch: got={wave_zarr.shape[1]} expected={n_zarr}'
	)

if CONVERT_STRAINRATE_TO_PSEUDOVEL:
	wave_zarr = strainrate_to_pseudovel(
		wave_zarr, fs_in=float(fs_zarr), pseudovel_scale=float(PSEUDOVEL_SCALE)
	)

wave_100 = resample_window_poly(
	wave_zarr,
	fs_in=float(fs_zarr),
	fs_out=float(TARGET_FS_HZ),
	out_len=int(WINDOW_IN_SAMPLES),
)
wave_100 = bandpass_window(
	wave_100,
	fs=float(TARGET_FS_HZ),
	post_bp_low_hz=float(POST_BP_LOW_HZ),
	post_bp_high_hz=float(POST_BP_HIGH_HZ),
	post_bp_order=int(POST_BP_ORDER),
)

C, N = int(wave_100.shape[0]), int(wave_100.shape[1])
if int(WINDOW_IN_SAMPLES) != N:
	raise ValueError(f'wave_100 length mismatch: got={N} expected={WINDOW_IN_SAMPLES}')

runner = PhaseNetWindowRunner(
	weights=str(PHASENET_WEIGHTS),
	in_samples=int(PHASENET_IN_SAMPLES),
	batch_traces=int(PHASENET_BATCH_TRACES),
)

p, s = _phasenet_predict_full_window(
	runner=runner,
	wave_100=wave_100,
	in_samples=int(PHASENET_IN_SAMPLES),
	overlap=int(PHASENET_OVERLAP),
	overlap_merge=str(OVERLAP_MERGE),
)

t = np.arange(N, dtype=np.float32) / float(TARGET_FS_HZ)
extent = (float(t[0]), float(t[-1]), float(channel_ids[0]), float(channel_ids[-1]))
T, Y = np.meshgrid(t, channel_ids)

absmax = 1

title = (
	f'Amplitude + PhaseNet P(solid)/S(dashed) | start={utc_ms_to_iso(first_window_start_ms)} | '
	f'{TDMS_FILE_NAME} (+{WINDOW_SECONDS:.0f}s window)'
)

fig = plt.figure(figsize=(14, 12))
ax0 = fig.add_subplot(3, 1, 1)
ax1 = fig.add_subplot(3, 1, 2, sharex=ax0)
ax2 = fig.add_subplot(3, 1, 3, sharex=ax0)

from waveform.filters import zscore_tracewise

wave_plot = zscore_tracewise(wave_100)

ax0.imshow(
	wave_plot,
	aspect='auto',
	cmap='seismic',
	origin='lower',
	extent=extent,
	vmin=-absmax,
	vmax=absmax,
	interpolation='none',
)
ax0.contour(T, Y, p, levels=list(P_CONTOUR_LEVELS), linestyles='solid')
ax0.contour(T, Y, s, levels=list(S_CONTOUR_LEVELS), linestyles='dashed')
ax0.set_ylabel('channel id')
ax0.set_title(title)

ax1.imshow(
	p,
	aspect='auto',
	origin='lower',
	extent=extent,
	vmin=0.0,
	vmax=0.1,
	interpolation='none',
)
ax1.set_ylabel('channel id')
ax1.set_title('PhaseNet P probability (stitched)')

ax2.imshow(
	s,
	aspect='auto',
	origin='lower',
	extent=extent,
	vmin=0.0,
	vmax=0.1,
	interpolation='none',
)
ax2.set_ylabel('channel id')
ax2.set_xlabel('time (s) from window start')
ax2.set_title('PhaseNet S probability (stitched)')

fig.tight_layout()
plt.show()
