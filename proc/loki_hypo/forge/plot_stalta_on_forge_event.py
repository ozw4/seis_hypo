# %%
# file: proc/forge/plot_stalta_on_forge_event.py
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.config import LokiWaveformStackingInputs
from common.load_config import load_config
from io_util.stream import build_stream_from_forge_event_npy
from pick.stalta_probs import StaltaProbSpec, build_probs_by_station_stalta
from waveform.preprocess import preprocess_stream_detrend_bandpass, spec_from_inputs

# ========= USER EDIT HERE =========
EVENT_DIR = Path(
	'/home/dcuser/daseventnet/data/silixa/cut_events_for_loki/event_000002'
)

# DAS設定
DAS_CHANNEL_CODE = 'DASZ'  # build_stream_from_forge_event_npy が要求（末尾Z）
COMPONENT = 'Z'  # STALTA対象成分（channel末尾と一致）

# 前処理（任意）
USE_PREPROCESS = True
INPUTS_YAML = Path('/workspace/data/config/loki_inputs.yaml')
INPUTS_PRESET = 'forge_das'

# STALTA設定（まずはP用の raw transform）
STALTA_SPEC = StaltaProbSpec(
	transform='raw',
	sta_sec=0.1,
	lta_sec=0.5,
	smooth_sec=None,
	clip_p=99.5,
	log1p=False,
)

# 可視化
N_CHANNELS_PLOT = 1000  # 1000chでもここで間引いて表示
MAX_TIME_SAMPLES_PLOT = 8000  # 表示用に時間方向も間引く（計算自体はフル長）
OUT_PNG = EVENT_DIR / 'stalta_debug.png'
OUT_NPY = EVENT_DIR / 'stalta_score_P.npy'  # 解析用に保存したいとき
SAVE_SCORE_NPY = True
# ================================


def _select_indices_evenly(n_total: int, n_pick: int) -> np.ndarray:
	if n_total <= 0:
		raise ValueError('n_total must be > 0')
	if n_pick <= 0:
		raise ValueError('n_pick must be > 0')
	if n_pick >= n_total:
		return np.arange(n_total, dtype=int)
	idx = np.linspace(0, n_total - 1, num=int(n_pick), dtype=int)
	# 重複除去しつつ順序維持
	seen = set()
	out = []
	for i in idx.tolist():
		if i not in seen:
			seen.add(i)
			out.append(i)
	return np.asarray(out, dtype=int)


def _decimate_for_plot(x: np.ndarray, *, max_len: int) -> tuple[np.ndarray, int]:
	if x.ndim != 2:
		raise ValueError(f'x must be 2D, got shape={x.shape}')
	t = int(x.shape[1])
	if t <= 0:
		raise ValueError('time length must be > 0')
	if max_len <= 0:
		raise ValueError('max_len must be > 0')
	if t <= max_len:
		return x, 1
	step = int(np.ceil(t / float(max_len)))
	return x[:, ::step], step


def _mad_scale_rows(x: np.ndarray, *, eps: float = 1.0e-12) -> np.ndarray:
	# 表示用：各chをMADで割って見やすくする（値の比較は目的にしない）
	if x.ndim != 2:
		raise ValueError(f'x must be 2D, got shape={x.shape}')
	med = np.median(x, axis=1, keepdims=True)
	mad = np.median(np.abs(x - med), axis=1, keepdims=True)
	return x / (mad + float(eps))


# def main() -> None:
event_dir = Path(EVENT_DIR)
if not event_dir.is_dir():
	raise FileNotFoundError(f'EVENT_DIR not found: {event_dir}')

# 1) npy -> ObsPy Stream
st = build_stream_from_forge_event_npy(event_dir, channel_code=str(DAS_CHANNEL_CODE))
if len(st) == 0:
	raise ValueError(f'empty stream: {event_dir}')

fs = float(st[0].stats.sampling_rate)
npts = int(st[0].stats.npts)
starttime = st[0].stats.starttime  # UTCDateTime
if fs <= 0:
	raise ValueError(f'invalid fs: {fs}')
if npts <= 0:
	raise ValueError(f'invalid npts: {npts}')

# 2) 前処理（任意）
if USE_PREPROCESS:
	inputs = load_config(LokiWaveformStackingInputs, INPUTS_YAML, INPUTS_PRESET)
	pre_spec = spec_from_inputs(inputs)
	fs_expected = float(inputs.base_sampling_rate_hz)
	if abs(fs - fs_expected) > 1e-6:
		raise ValueError(f'fs mismatch: stream fs={fs} vs inputs fs={fs_expected}')
	preprocess_stream_detrend_bandpass(st, spec=pre_spec, fs_expected=fs_expected)

# 3) STALTA（score 0..1）
probs = build_probs_by_station_stalta(
	st,
	fs=fs,
	component=str(COMPONENT),
	phase='P',
	spec=STALTA_SPEC,
)

# 4) Stream順（stations.csvのindex昇順）で行列化
stations = [str(tr.stats.station) for tr in st]
wave = np.vstack([np.asarray(tr.data, dtype=np.float32) for tr in st])  # (C,T)

score_list = []
for sta in stations:
	d = probs.get(sta)
	if d is None or 'P' not in d:
		raise ValueError(f'missing STALTA score for station={sta}')
	s = np.asarray(d['P'], dtype=np.float32)
	if s.ndim != 1 or int(s.size) != npts:
		raise ValueError(
			f'invalid score shape: station={sta} shape={s.shape} expected=({npts},)'
		)
	score_list.append(s)
score = np.vstack(score_list)  # (C,T)

if SAVE_SCORE_NPY:
	np.save(Path(OUT_NPY), score, allow_pickle=False)

# 5) 表示用にチャンネルを間引く
c = int(wave.shape[0])
idx = _select_indices_evenly(c, int(N_CHANNELS_PLOT))
w_sel = wave[idx]
s_sel = score[idx]

# 6) 表示用に時間方向も間引く
w_plot, t_step = _decimate_for_plot(w_sel, max_len=int(MAX_TIME_SAMPLES_PLOT))
s_plot, _ = _decimate_for_plot(s_sel, max_len=int(MAX_TIME_SAMPLES_PLOT))

# 時間軸（秒）
t = (np.arange(w_plot.shape[1], dtype=float) * float(t_step)) / fs

# 7) 可視化
w_plot_disp = _mad_scale_rows(w_plot)

# stack（平均）
stack_p = np.mean(s_sel, axis=0)
t_full = np.arange(npts, dtype=float) / fs
i_peak = int(np.argmax(stack_p))
t_peak = float(t_full[i_peak])

fig = plt.figure(figsize=(14, 10))

ax1 = fig.add_subplot(3, 1, 1)
im1 = ax1.imshow(
	w_plot_disp,
	vmin=-10,
	vmax=10,
	aspect='auto',
	origin='lower',
	cmap='seismic',
	interpolation='None',
)
ax1.set_title(
	f'Waveform (MAD-scaled)  event={event_dir.name}  start={starttime}  fs={fs:.3f}Hz'
)
ax1.set_ylabel('channel (subsampled)')
ax1.set_xticks([])

ax2 = fig.add_subplot(3, 1, 2)
im2 = ax2.imshow(s_plot, aspect='auto', origin='lower')
ax2.set_title('STALTA score (P, 0..1)')
ax2.set_ylabel('channel (subsampled)')
ax2.set_xticks([])

ax3 = fig.add_subplot(3, 1, 3)
ax3.plot(t_full, stack_p)
ax3.axvline(t_peak)
ax3.set_title(f'Mean STALTA score across channels (peak @ {t_peak:.3f} s)')
ax3.set_xlabel('time (s from window_start)')
ax3.set_ylabel('mean score')

fig.tight_layout()
out_png = Path(OUT_PNG)
out_png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_png, dpi=200)
# plt.close(fig)

print(f'[OK] saved: {out_png}')
if SAVE_SCORE_NPY:
	print(f'[OK] saved: {OUT_NPY}')
print(f'[INFO] peak(mean score): t={t_peak:.3f}s (sample={i_peak}/{npts})')


# if __name__ == '__main__':
# main()
