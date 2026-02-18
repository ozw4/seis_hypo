# %%
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
from pick.ept_runner import EqTWindowRunner
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

TDMS_FILE_NAME = 'FORGE_DFIT_UTC_20220417_105956.202.tdms'
MATCH_MODE = 'exact'  # 'exact' or 'endswith'
TDMS_OCCURRENCE = 0  # 同名が複数ヒットしたとき何個目を起点にするか（0-based）

# channel selection
CHANNEL_RANGE: tuple[int, int] | None = None  # contiguous-only in stored channel axis
APPLY_WELL_AB_KEEP = True
WELL_A_KEEP_0BASED_INCL = (92, 1062)
WELL_B_KEEP_0BASED_INCL = (1216, 2385)

# preprocessing (zscore is ONLY inside EqTWindowRunner)
CONVERT_STRAINRATE_TO_PSEUDOVEL = False
PSEUDOVEL_SCALE = 1.0
TARGET_FS_HZ = 250.0
POST_BP_LOW_HZ = 25.0
POST_BP_HIGH_HZ = 120.0
POST_BP_ORDER = 1

# EqT windowing (single 60s window)
EQT_WEIGHTS = '/workspace/model_weight/010_Train_EqT_FT-STEAD_rot30_Hinet selftrain.pth'
EQT_IN_SAMPLES = 6000
EQT_BATCH_TRACES = 64

# overlay
P_CONTOUR_LEVELS = (0.3, 0.5, 0.7)
S_CONTOUR_LEVELS = (0.3, 0.5, 0.7)

TMP_PARENT_DIR = Path('/tmp')
# =========================

big = zarr.open_group(str(BIG_ZARR_DIR), mode='r')
fs_zarr = float(big.attrs['fs_out_hz'])
fi = as_int_rate(fs_zarr, 'fs_zarr')
fo = as_int_rate(TARGET_FS_HZ, 'target_fs_hz')
if (int(EQT_IN_SAMPLES) * int(fi)) % int(fo) != 0:
	raise ValueError(
		'EQT_IN_SAMPLES*fs_zarr must be divisible by target_fs: '
		f'L={EQT_IN_SAMPLES} fi={fi} fo={fo}'
	)
n_zarr = (int(EQT_IN_SAMPLES) * int(fi)) // int(fo)

ts = datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%SZ')
tmp_dir = TMP_PARENT_DIR / f'tmp_debug_eqt_60s_{ts}.zarr'

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
	out_len=int(EQT_IN_SAMPLES),
)
wave_100 = bandpass_window(
	wave_100,
	fs=float(TARGET_FS_HZ),
	post_bp_low_hz=float(POST_BP_LOW_HZ),
	post_bp_high_hz=float(POST_BP_HIGH_HZ),
	post_bp_order=int(POST_BP_ORDER),
)

C, N = int(wave_100.shape[0]), int(wave_100.shape[1])
if int(EQT_IN_SAMPLES) != N:
	raise ValueError(f'wave_100 length mismatch: got={N} expected={EQT_IN_SAMPLES}')

runner = EqTWindowRunner(
	weights=str(EQT_WEIGHTS),
	in_samples=int(EQT_IN_SAMPLES),
	batch_traces=int(EQT_BATCH_TRACES),
)
_, p, s = runner.predict_window(wave_100)  # zscoreはrunner内

t = np.arange(N, dtype=np.float32) / float(TARGET_FS_HZ)
extent = (float(t[0]), float(t[-1]), float(channel_ids[0]), float(channel_ids[-1]))
T, Y = np.meshgrid(t, channel_ids)

absmax = 3

title = (
	f'Amplitude + P(solid)/S(dashed) | start={utc_ms_to_iso(first_window_start_ms)} | '
	f'{TDMS_FILE_NAME} (+60s window)'
)

fig = plt.figure(figsize=(14, 12))
ax0 = fig.add_subplot(3, 1, 1)
ax1 = fig.add_subplot(3, 1, 2, sharex=ax0)
ax2 = fig.add_subplot(3, 1, 3, sharex=ax0)
from waveform.filters import zscore_tracewise

wave_100 = zscore_tracewise(wave_100)
ax0.imshow(
	wave_100,
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
ax1.set_title('P probability')

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
ax2.set_title('S probability')

fig.tight_layout()
plt.show()
