# %%
# file: proc/loki_hypo/forge/plot_stalta_on_forge_event.py
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from obspy import UTCDateTime

from common.config import LokiWaveformStackingInputs
from common.load_config import load_config
from common.stride import normalize_channel_stride
from io_util.stream import build_stream_from_forge_event_npy
from pick.stalta_probs import StaltaProbSpec, build_probs_by_station_stalta
from waveform.preprocess import preprocess_stream_detrend_bandpass, spec_from_inputs

for event_id in range(1, 51):
	# ========= USER EDIT HERE =========
	EVENT_DIR = Path(
		f'/home/dcuser/daseventnet/data/silixa/cut_events_for_loki/event_{event_id:06d}'
	)

	# LOKI .phs settings (overlay picks)
	# If None, overlay is disabled.
	LOKI_OUTPUT_ROOT: Path | None = Path(
		'/workspace/proc/loki_hypo/forge/loki_output_forge_das_pass1'
	)

	LOKI_OUTPUT_SUBDIR = 'pass1_stalta_p_das_deci10'
	TRIAL = 0
	PHS_GLOB: str | None = None  # if None -> f"*trial{TRIAL}.phs"
	SHOW_S = False

	# Overlay appearance
	P_PICK_COLOR = 'Red'
	S_PICK_COLOR = 'cyan'
	P_PICK_MARKER = '_'
	S_PICK_MARKER = 'x'

	# DAS settings
	DAS_CHANNEL_CODE = 'DASZ'  # build_stream_from_forge_event_npy requires channel_code ending with 'Z'
	COMPONENT = 'Z'  # STALTA target component (must match channel suffix)
	CHANNEL_STRIDE: int | None = None  # >=2 enables subsampling; None/<=1 disables.

	# Preprocess (optional)
	USE_PREPROCESS = True
	INPUTS_YAML = Path('/workspace/data/config/loki_inputs.yaml')
	INPUTS_PRESET = 'forge_das'

	# STALTA spec (P, raw transform)
	STALTA_SPEC = StaltaProbSpec(
		transform='raw',
		sta_sec=0.02,
		lta_sec=0.1,
		smooth_sec=None,
		clip_p=99.5,
		log1p=False,
	)

	# Visualization
	N_CHANNELS_PLOT = 1000  # select this many channels for display
	MAX_TIME_SAMPLES_PLOT = 8000  # decimate time axis for display only
	OUT_PNG = EVENT_DIR / 'stalta_debug_time_vertical.png'
	OUT_NPY = EVENT_DIR / 'stalta_score_P.npy'
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

		seen: set[int] = set()
		out: list[int] = []
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
		if x.ndim != 2:
			raise ValueError(f'x must be 2D, got shape={x.shape}')
		med = np.median(x, axis=1, keepdims=True)
		mad = np.median(np.abs(x - med), axis=1, keepdims=True)
		return x / (mad + float(eps))

	def _parse_loki_phs(
		path: Path,
	) -> tuple[dict[str, UTCDateTime], dict[str, UTCDateTime]]:
		p_abs_by_station: dict[str, UTCDateTime] = {}
		s_abs_by_station: dict[str, UTCDateTime] = {}

		lines = path.read_text(encoding='utf-8').splitlines()
		for line in lines:
			line = line.strip()
			if not line or line.startswith('#'):
				continue
			parts = line.split()
			if len(parts) < 2:
				raise ValueError(f'invalid .phs line (need >=2 cols): {line}')
			station = parts[0]
			p_abs_by_station[station] = UTCDateTime(parts[1])
			if len(parts) >= 3:
				s_value = parts[2]
				if s_value.lower() not in {'', 'none', 'nan', 'null'}:
					s_abs_by_station[station] = UTCDateTime(s_value)

		return p_abs_by_station, s_abs_by_station

	def _resolve_phs_path(event_dir: Path) -> Path | None:
		if LOKI_OUTPUT_ROOT is None:
			return None

		root = Path(LOKI_OUTPUT_ROOT)
		phs_dir = root / str(LOKI_OUTPUT_SUBDIR) / event_dir.name
		pattern = str(PHS_GLOB) if PHS_GLOB is not None else f'*trial{int(TRIAL)}.phs'

		matches = sorted(phs_dir.glob(pattern))
		if not matches:
			raise FileNotFoundError(f'no .phs match under {phs_dir} (glob={pattern})')
		if len(matches) > 1:
			raise ValueError(
				f'multiple .phs matches under {phs_dir} (glob={pattern}): {matches}'
			)
		return matches[0]

	def _imshow_extent_time_vertical(
		n_ch: int, n_t: int, *, dt: float
	) -> tuple[float, float, float, float]:
		# time=0 を上側にしたいので、y方向の extent を反転させる (bottom > top)
		# x centers at 0..n_ch-1 -> extent x = (-0.5, n_ch-0.5)
		# y centers at 0, dt, 2dt, ... -> extent y = ((n_t-0.5)dt, -0.5dt)
		if n_ch <= 0 or n_t <= 0:
			raise ValueError(f'invalid shape for extent: n_ch={n_ch} n_t={n_t}')
		return (
			-0.5,
			float(n_ch) - 0.5,
			(float(n_t) - 0.5) * float(dt),
			-0.5 * float(dt),
		)

	# ===== Main (script-style; no argparse) =====
	event_dir = Path(EVENT_DIR)
	if not event_dir.is_dir():
		raise FileNotFoundError(f'EVENT_DIR not found: {event_dir}')

	# 1) npy -> ObsPy Stream
	st = build_stream_from_forge_event_npy(
		event_dir, channel_code=str(DAS_CHANNEL_CODE)
	)
	if len(st) == 0:
		raise ValueError(f'empty stream: {event_dir}')

	stride = normalize_channel_stride(CHANNEL_STRIDE)
	if stride is not None:
		orig_n_channels = len(st)
		kept_indices = list(range(0, orig_n_channels, stride))
		st = st.__class__([st[i] for i in kept_indices])
		msg_tail = ' (truncated)' if len(kept_indices) > 10 else ''
		print(
			f'[STALTA-PLOT-DAS] channel stride enabled: stride={stride} '
			f'kept={len(st)} original={orig_n_channels} '
			f'indices_head={kept_indices[:10]}{msg_tail}'
		)

	fs = float(st[0].stats.sampling_rate)
	npts = int(st[0].stats.npts)
	starttime = st[0].stats.starttime  # UTCDateTime
	if fs <= 0:
		raise ValueError(f'invalid fs: {fs}')
	if npts <= 0:
		raise ValueError(f'invalid npts: {npts}')

	# 2) Preprocess (optional)
	if USE_PREPROCESS:
		inputs = load_config(LokiWaveformStackingInputs, INPUTS_YAML, INPUTS_PRESET)
		pre_spec = spec_from_inputs(inputs)
		fs_expected = float(inputs.base_sampling_rate_hz)
		if abs(fs - fs_expected) > 1e-6:
			raise ValueError(f'fs mismatch: stream fs={fs} vs inputs fs={fs_expected}')
		preprocess_stream_detrend_bandpass(st, spec=pre_spec, fs_expected=fs_expected)

	# 3) STALTA (score 0..1)
	probs = build_probs_by_station_stalta(
		st,
		fs=fs,
		component=str(COMPONENT),
		phase='P',
		spec=STALTA_SPEC,
	)

	# 3.5) LOKI .phs (optional overlay)
	phs_path = _resolve_phs_path(event_dir)
	p_abs_by_station: dict[str, UTCDateTime] = {}
	s_abs_by_station: dict[str, UTCDateTime] = {}
	if phs_path is None:
		print('[INFO] LOKI overlay disabled (LOKI_OUTPUT_ROOT is None).')
	else:
		p_abs_by_station, s_abs_by_station = _parse_loki_phs(phs_path)
		print(f'[INFO] LOKI .phs loaded: {phs_path}')

	# 4) Matrixify in Stream order
	stations = [str(tr.stats.station) for tr in st]
	wave = np.vstack([np.asarray(tr.data, dtype=np.float32) for tr in st])  # (C,T)

	score_list: list[np.ndarray] = []
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

	# Validate station name matching if overlay enabled
	if phs_path is not None:
		n_match = sum(1 for sta in stations if sta in p_abs_by_station)
		if n_match == 0:
			raise ValueError(
				'LOKI overlay enabled but no station names matched between Stream and .phs. '
				'Check how build_stream_from_forge_event_npy sets Trace.stats.station '
				'and how station names appear in .phs.'
			)
		print(f'[INFO] LOKI station matches: {n_match}/{len(stations)}')

	# Build per-trace pick arrays (relative seconds)
	window_len_sec = float(npts) / fs
	p_sec_by_trace = np.full(len(st), np.nan, dtype=float)
	s_sec_by_trace = np.full(len(st), np.nan, dtype=float)

	for i, sta in enumerate(stations):
		p_abs = p_abs_by_station.get(sta)
		if p_abs is not None:
			rel_sec = float(p_abs - starttime)
			if 0.0 <= rel_sec < window_len_sec:
				p_sec_by_trace[i] = rel_sec

		s_abs = s_abs_by_station.get(sta)
		if s_abs is not None:
			rel_sec = float(s_abs - starttime)
			if 0.0 <= rel_sec < window_len_sec:
				s_sec_by_trace[i] = rel_sec

	if SAVE_SCORE_NPY:
		np.save(Path(OUT_NPY), score, allow_pickle=False)

	# 5) Subselect channels for display
	c = int(wave.shape[0])
	idx = _select_indices_evenly(c, int(N_CHANNELS_PLOT))
	w_sel = wave[idx]
	s_sel = score[idx]
	p_sec_sel = p_sec_by_trace[idx]
	s_sec_sel = s_sec_by_trace[idx]

	# 6) Decimate time axis for display
	w_plot, t_step = _decimate_for_plot(w_sel, max_len=int(MAX_TIME_SAMPLES_PLOT))
	s_plot = s_sel[:, ::t_step]
	dt_plot = float(t_step) / fs

	# 7) Visualization (time axis vertical)
	w_plot_disp = _mad_scale_rows(w_plot)

	# Prepare images (transpose so time is vertical)
	# wave: (C, Tplot) -> (Tplot, C)
	w_img = w_plot_disp.T
	s_img = s_plot.T

	extent = _imshow_extent_time_vertical(
		n_ch=int(w_img.shape[1]), n_t=int(w_img.shape[0]), dt=dt_plot
	)
	t_max_plot = float(w_img.shape[0] - 1) * dt_plot  # 表示している最終時刻

	fig = plt.figure(figsize=(8, 12))

	ax1 = fig.add_subplot(1, 2, 1)
	ax1.imshow(
		w_img,
		vmin=-10,
		vmax=10,
		aspect='auto',
		origin='upper',
		cmap='seismic',
		interpolation='None',
		extent=extent,
	)

	ax1.set_title('waveform (normalized by MAD)')
	ax1.set_xlabel('channel (selected)')
	ax1.set_ylabel('time (s from window_start)')
	ax1.set_xticks([])
	ax1.set_ylim(t_max_plot, 0.0)  # ★ time=0 を上側へ

	ax2 = fig.add_subplot(1, 2, 2)
	ax2.imshow(
		s_img,
		aspect='auto',
		origin='upper',
		interpolation='None',
		extent=extent,
	)
	ax2.set_title('STALTA score (P, 0..1)')
	ax2.set_xlabel('channel (selected)')
	ax2.set_ylabel('time (s from window_start)')
	ax2.set_xticks([])
	ax2.set_ylim(t_max_plot, 0.0)  # ★ time=0 を上側へ

	# Overlay picks (time is y-axis in seconds, channel is x-axis)
	ch_x = np.arange(len(p_sec_sel), dtype=float)

	if np.any(np.isfinite(p_sec_sel)):
		p_mask = np.isfinite(p_sec_sel)
		ax1.scatter(
			ch_x[p_mask],
			p_sec_sel[p_mask],
			s=15,
			marker=P_PICK_MARKER,
			color=P_PICK_COLOR,
			alpha=0.8,
			linewidths=0.7,
			rasterized=True,
		)
		ax2.scatter(
			ch_x[p_mask],
			p_sec_sel[p_mask],
			s=15,
			marker=P_PICK_MARKER,
			color=P_PICK_COLOR,
			alpha=0.8,
			linewidths=0.7,
			rasterized=True,
		)

	if SHOW_S and np.any(np.isfinite(s_sec_sel)):
		s_mask = np.isfinite(s_sec_sel)
		ax1.scatter(
			ch_x[s_mask],
			s_sec_sel[s_mask],
			s=15,
			marker=S_PICK_MARKER,
			color=S_PICK_COLOR,
			alpha=0.8,
			linewidths=0.7,
			rasterized=True,
		)
		ax2.scatter(
			ch_x[s_mask],
			s_sec_sel[s_mask],
			s=15,
			marker=S_PICK_MARKER,
			color=S_PICK_COLOR,
			alpha=0.8,
			linewidths=0.7,
			rasterized=True,
		)
	plt.suptitle(f'event={event_dir.name}  start={starttime}')
	fig.tight_layout()
	out_png = Path(OUT_PNG)
	out_png.parent.mkdir(parents=True, exist_ok=True)
	plt.show()
	fig.savefig(out_png, dpi=200)
	plt.close(fig)
	print(f'[OK] saved: {out_png}')
	if SAVE_SCORE_NPY:
		print(f'[OK] saved: {OUT_NPY}')
	print(f'[INFO] peak(mean score): t={t_peak:.3f}s (sample={i_peak}/{npts})')
	print(
		f'[INFO] display: n_channels={len(idx)} n_time_rows={w_img.shape[0]} '
		f't_step={t_step} dt={dt_plot:.6f}s'
	)
	if phs_path is not None:
		print(f'[INFO] phs: {phs_path} show_s={SHOW_S}')
