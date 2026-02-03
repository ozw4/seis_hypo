# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import detrend as sp_detrend

from jma.prepare.event_paths import resolve_single_evt
from jma.station_reader import read_hinet_channel_table
from jma.win32_reader import get_evt_info, read_win32_resampled
from waveform.filters import bandpass_iir_filtfilt

SNR_CSV = Path('/workspace/data/waveform/jma/snr_pick_table.csv')
OUT_DIR = Path('/workspace/data/waveform/jma/snr_qc_out')

CONT_SUBDIR = 'continuous'  # cntが入ってる想定のサブdir（違うなら変更）

# ============================
# ★ここだけいじればOK：どのSNR列でQCするか
# "snr_db" | "energy" | "rms" | "stalta"
SNR_FB = 'rms'
# ============================

# ============================
# ★site correction settings（毎回このスクリプト内で計算）
APPLY_SITE_CORRECTION = True
# shrinkage strength: larger => stronger pull toward 0 for small-n stations
SITE_SHRINK_K = 30.0
# ============================

# 代表トレースの選び方（dBスケール想定）
SNR_BINS = [-1e9, 0.0, 10.0, 20.0, 1e9]
SNR_BIN_LABELS = ['snr<0', '0-10', '10-20', '>=20']
QUANTILES_PER_BIN = np.linspace(0, 1, 5)  # 各binから3本

# 描画窓（pick=0を中心）
PLOT_PRE_SEC = 8.0
PLOT_POST_SEC = 12.0

# フィルタ設計（SNRテーブルの fpass を使う。fstop は簡易に 0.8/1.2 倍）
USE_BANDPASS = True


def _snr_col_from_fb(snr_fb: str) -> str:
	s = str(snr_fb).strip().lower()
	if s in {'snr_db', 'default', 'primary'}:
		return 'snr_db'
	if s in {'energy', 'snr_db_energy'}:
		return 'snr_db_energy'
	if s in {'rms', 'snr_db_rms'}:
		return 'snr_db_rms'
	if s in {'stalta', 'snr_db_stalta'}:
		return 'snr_db_stalta'
	raise ValueError(f'unknown SNR_FB: {snr_fb} (use snr_db/energy/rms/stalta)')


def _require_file(p: Path, name: str) -> None:
	if not p.is_file():
		raise FileNotFoundError(f'{name} not found: {p}')


def _parse_win_s(win_s: str) -> tuple[float, float]:
	a, b = str(win_s).split(',')
	return float(a), float(b)


def _resolve_win32_paths(event_dir: Path, source_id: str) -> tuple[Path, Path]:
	sid = str(source_id)

	if sid == 'evt':
		evt_path = resolve_single_evt(event_dir, allow_none=False)
		stem = evt_path.stem

		ch_active = event_dir / f'{stem}_active.ch'
		ch_plain = event_dir / f'{stem}.ch'
		if ch_active.is_file():
			ch_path = ch_active
		elif ch_plain.is_file():
			print(f'[warn] active .ch not found, using plain .ch: {ch_plain}')
			ch_path = ch_plain
		else:
			raise FileNotFoundError(f'no .ch found for evt: {ch_active} / {ch_plain}')

		return evt_path, ch_path

	if sid.startswith('cnt:'):
		name = sid.split(':', 1)[1]
		win_path = event_dir / CONT_SUBDIR / name
		if not win_path.is_file():
			alt = event_dir / name
			if alt.is_file():
				print(f'[warn] cnt not found under {CONT_SUBDIR}, using: {alt}')
				win_path = alt
			else:
				raise FileNotFoundError(f'cnt not found: {win_path} (or {alt})')
		ch_path = win_path.with_suffix('.ch')
		if not ch_path.is_file():
			raise FileNotFoundError(f'cnt .ch not found: {ch_path}')
		return win_path, ch_path

	if sid == 'cnt':
		cands = sorted((event_dir / CONT_SUBDIR).glob('*.cnt'))
		if len(cands) != 1:
			raise ValueError(
				f'cannot resolve single cnt in {event_dir}/{CONT_SUBDIR}: {len(cands)}'
			)
		win_path = cands[0]
		ch_path = win_path.with_suffix('.ch')
		if not ch_path.is_file():
			raise FileNotFoundError(f'cnt .ch not found: {ch_path}')
		return win_path, ch_path

	raise ValueError(f'unknown source_id: {sid}')


def _load_single_trace_processed(row: pd.Series) -> tuple[np.ndarray, float, dict]:
	event_dir = Path(row['event_dir'])
	win_path, ch_path = _resolve_win32_paths(event_dir, str(row['source_id']))

	_require_file(win_path, 'WIN32')
	_require_file(ch_path, 'channel table')

	station_df = read_hinet_channel_table(ch_path)

	ch_int = int(row['ch_int'])
	hit = station_df[station_df['ch_int'].astype(int) == ch_int]
	if hit.empty:
		raise ValueError(f'ch_int={ch_int} not found in {ch_path.name}')
	if len(hit) != 1:
		raise ValueError(f'ch_int={ch_int} duplicated in {ch_path.name} (n={len(hit)})')

	ch_hex = str(hit['ch_hex'].iloc[0])

	info = get_evt_info(win_path, scan_rate_blocks=50)
	fs = float(row['fs_target_hz'])
	if int(fs) <= 0:
		raise ValueError(f'invalid fs_target_hz: {row["fs_target_hz"]}')

	x = read_win32_resampled(
		win_path,
		station_df,
		target_sampling_rate_HZ=int(fs),
		duration_SECOND=int(info.span_seconds),
		channels_hex=[ch_hex],
	)
	if x.shape[0] != 1:
		raise ValueError(f'expected 1ch read, got shape={x.shape}')

	x1 = np.asarray(x[0], dtype=float)
	x1 = sp_detrend(x1, type='linear')

	if USE_BANDPASS:
		f_lo = float(row['bandpass_fpass_lo_hz'])
		f_hi = float(row['bandpass_fpass_hi_hz'])
		if not (0.0 < f_lo < f_hi):
			raise ValueError(f'invalid bandpass fpass: lo={f_lo}, hi={f_hi}')
		x1 = bandpass_iir_filtfilt(
			x1,
			fs=float(fs),
			fstop_lo=float(f_lo) * 0.8,
			fpass_lo=float(f_lo),
			fpass_hi=float(f_hi),
			fstop_hi=float(f_hi) * 1.2,
			gpass=1.0,
			gstop=40.0,
		)

	meta = {
		'win_path': str(win_path),
		'ch_path': str(ch_path),
		'evt_start_time': info.start_time,
		'span_seconds': int(info.span_seconds),
		'fs': float(fs),
		'ch_hex': ch_hex,
	}
	return x1, float(fs), meta


def _pick_index(row: pd.Series, t_start, fs: float, n_t: int) -> float:
	from jma.picks import pick_time_to_index

	pick_time = pd.to_datetime(
		str(row['pick_time']), format='ISO8601', errors='raise'
	).to_pydatetime()
	idx = pick_time_to_index(pick_time, fs_hz=float(fs), t_start=t_start, n_t=int(n_t))
	if not np.isfinite(idx):
		raise ValueError(f'pick index out of range: pick_time={row["pick_time"]}')
	return float(idx)


def _site_correction_median_shrinkage(
	df: pd.DataFrame, *, snr_col: str, shrink_k: float
) -> tuple[pd.Series, pd.Series, pd.Series]:
	"""site_term = median(snr|station) - median(snr|all)
	shrink = n/(n+k)
	site_term_shrunk = site_term*shrink
	snr_used = snr_raw - site_term_shrunk
	"""
	snr = df[snr_col].astype(float)
	sta = df['station'].astype(str)

	global_med = float(np.nanmedian(snr.to_numpy(dtype=float)))

	g = df.assign(_snr=snr, _sta=sta).groupby('_sta', dropna=False)['_snr']
	sta_med = g.median()
	sta_n = g.size().astype(float)

	site_term = sta_med - global_med
	shrink = sta_n / (sta_n + float(shrink_k))
	site_term_shrunk = site_term * shrink

	term_map = site_term_shrunk.to_dict()
	shrink_map = shrink.to_dict()

	term_s = sta.map(term_map).astype(float)
	shrink_s = sta.map(shrink_map).astype(float)

	snr_used = snr - term_s
	return snr_used, term_s, shrink_s


def _prepare_snr_used(df: pd.DataFrame, *, snr_col: str) -> pd.DataFrame:
	df2 = df.copy()
	df2['snr_raw_db'] = df2[snr_col].astype(float)

	if not APPLY_SITE_CORRECTION:
		df2['snr_used_db'] = df2['snr_raw_db']
		df2['site_term_db'] = np.nan
		df2['site_shrink'] = np.nan
		return df2

	snr_used, term, shrink = _site_correction_median_shrinkage(
		df2, snr_col=snr_col, shrink_k=float(SITE_SHRINK_K)
	)
	df2['snr_used_db'] = snr_used.astype(float)
	df2['site_term_db'] = term.astype(float)
	df2['site_shrink'] = shrink.astype(float)
	return df2


def _select_representatives(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()

	df['snr_bin'] = pd.cut(
		df['snr_used_db'],
		bins=SNR_BINS,
		labels=SNR_BIN_LABELS,
		right=False,
		include_lowest=True,
	)

	picks: list[pd.Series] = []
	used = set()

	for lab in SNR_BIN_LABELS:
		sub = df[df['snr_bin'] == lab].copy()
		if sub.empty:
			continue

		sub = sub.sort_values('snr_used_db').reset_index(drop=False)

		for q in QUANTILES_PER_BIN:
			target = float(sub['snr_used_db'].quantile(q))
			sub['abs_diff'] = np.abs(sub['snr_used_db'] - target)
			sub2 = sub.sort_values(['abs_diff', 'snr_used_db']).copy()

			chosen = None
			for _, r in sub2.iterrows():
				k = int(r['index'])
				if k not in used:
					chosen = r
					used.add(k)
					break
			if chosen is not None:
				picks.append(chosen)

	return pd.DataFrame(picks).reset_index(drop=True)


def main() -> None:
	_require_file(SNR_CSV, 'SNR table')
	OUT_DIR.mkdir(parents=True, exist_ok=True)

	df = pd.read_csv(SNR_CSV, low_memory=False)

	snr_col = _snr_col_from_fb(SNR_FB)
	if snr_col not in df.columns:
		raise ValueError(
			f'SNR column not found: {snr_col}\n'
			f'Available snr columns: {sorted([c for c in df.columns if c.startswith("snr")])}'
		)

	req = {
		'event_dir',
		'event_id',
		'station',
		'pick_time',
		'source_id',
		'ch_int',
		'fs_target_hz',
		'bandpass_fpass_lo_hz',
		'bandpass_fpass_hi_hz',
		'noise_win_s',
		'signal_win_s',
		'distance_hypo_km',
		'mag1',
		'depth_km',
		snr_col,
	}
	miss = sorted(list(req - set(df.columns)))
	if miss:
		raise ValueError(f'snr csv missing columns: {miss}')

	df2 = _prepare_snr_used(df, snr_col=snr_col)

	reps = _select_representatives(df2)
	if reps.empty:
		raise ValueError('no representative traces selected (check bins/labels)')

	tag = 'sitecorr' if APPLY_SITE_CORRECTION else 'raw'
	sel_csv = OUT_DIR / f'snr_qc_selected_traces__{snr_col}__{tag}.csv'
	reps.to_csv(sel_csv, index=False)

	n = len(reps)
	ncols = min(3, n)
	nrows = int(np.ceil(n / ncols))

	fig = plt.figure(figsize=(6.2 * ncols, 3.2 * nrows))
	axes = [fig.add_subplot(nrows, ncols, i + 1) for i in range(n)]

	for i, (_, row) in enumerate(reps.iterrows()):
		ax = axes[i]

		x_full, fs, meta = _load_single_trace_processed(row)
		idx = _pick_index(row, meta['evt_start_time'], fs, n_t=len(x_full))

		pre = int(round(float(PLOT_PRE_SEC) * fs))
		post = int(round(float(PLOT_POST_SEC) * fs))

		i0 = int(round(idx)) - pre
		i1 = int(round(idx)) + post

		from common.core import slice_with_pad

		seg = slice_with_pad(x_full, i0, i1)

		t = (np.arange(len(seg), dtype=float) - float(pre)) / float(fs)

		n0, n1 = _parse_win_s(row['noise_win_s'])
		s0, s1 = _parse_win_s(row['signal_win_s'])
		ax.axvspan(n0, n1, alpha=0.15, label='noise window')
		ax.axvspan(s0, s1, alpha=0.15, label='signal window')

		ax.plot(t, seg, linewidth=0.9)
		ax.axvline(0.0, linewidth=1.2)

		snr_raw = float(row['snr_raw_db'])
		snr_used = float(row['snr_used_db'])
		lab = str(row['snr_bin'])

		sta = str(row['station'])
		eid = str(row['event_id'])
		dist = float(row['distance_hypo_km'])
		mag = float(row['mag1'])
		dep = float(row['depth_km'])
		f_lo = float(row['bandpass_fpass_lo_hz'])
		f_hi = float(row['bandpass_fpass_hi_hz'])

		term = (
			float(row['site_term_db']) if 'site_term_db' in row.index else float('nan')
		)
		shr = float(row['site_shrink']) if 'site_shrink' in row.index else float('nan')

		zr = float(np.mean(seg == 0.0))
		mx = float(np.max(np.abs(seg))) if len(seg) else float('nan')

		site_s = ''
		if APPLY_SITE_CORRECTION and np.isfinite(term) and np.isfinite(shr):
			site_s = f' | site={term:+.1f}dB shrink={shr:.2f}'

		ax.set_title(
			f'{lab} | used={snr_used:.1f} dB (raw={snr_raw:.1f}){site_s}\n'
			f'{sta} | eid={eid} | M={mag:.1f} dep={dep:.1f}km R={dist:.1f}km | bp={f_lo:g}-{f_hi:g}Hz | '
			f'max|x|={mx:.3g} zero%={zr * 100:.1f}'
		)
		ax.set_xlabel('time from pick (s)')
		ax.set_ylabel('amp (processed)')
		ax.grid(True)

		if i == 0:
			ax.legend(loc='upper right', fontsize=9)

	fig.tight_layout()
	out_png = OUT_DIR / f'snr_qc_gallery__{snr_col}__{tag}.png'
	fig.savefig(out_png, dpi=180)
	plt.close(fig)

	print(f'[ok] snr_col={snr_col} (SNR_FB={SNR_FB})')
	print(f'[ok] site_correction={APPLY_SITE_CORRECTION} shrink_k={SITE_SHRINK_K}')
	print(f'[ok] wrote: {sel_csv}')
	print(f'[ok] wrote: {out_png}')


if __name__ == '__main__':
	main()
