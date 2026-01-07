# %%
# /workspace/proc/prepare_data/forge/cut_events_fromzarr_for_loki.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

# =========================
# 設定（ここだけ触ればOK）
# =========================
DATA_DIR = Path('/home/dcuser/daseventnet/data/silixa')

ZARR_PATH = DATA_DIR / 'forge_dfit_block_78AB_250Hz.zarr'
EVENT_CSV = DATA_DIR / 'FORGE_DFIT_NAV_with_silixa_locations.csv'
STATION_META_CSV = Path(
	'/workspace/proc/prepare_data/forge/forge_das_station_metadata.csv'
)

OUT_BASE_DIR = DATA_DIR / 'cut_events_for_loki'
DB_DIR = OUT_BASE_DIR / 'db'

# Zarr keys
ZARR_KEY_BLOCK = 'block'  # (B,C,T)
ZARR_KEY_START_MS = 'starttime_utc_ms'
ZARR_KEY_DONE = 'done'

# Zarrが「旧keep(=2216ch)」のときの keep 範囲（0-based inclusive）
# 例: A=69–1078 (1010ch), B=1195–2400 (1206ch) -> 2216
ZARR_WELL_A_KEEP_0BASED_INCL = (69, 1078)
ZARR_WELL_B_KEEP_0BASED_INCL = (1195, 2400)

# station_meta columns
STATION_NAME_COL = 'station_id'
ORDER_COL = 'index'
FULL_CHANNEL_COL = 'channel'
WELL_COL = 'well'
WELL_A_NAME = '78A-32'
WELL_B_NAME = '78B-32'
FULL_CHANNEL_COUNT = 2432

# event csv columns
EVENTNUM_COL = 'eventNum'
NAV_TIME_COL = 'File Timestamp (UTC)'  # 例: 20220417 04:03:26.80500

# window (方針B): file_time_utc を中心に切り出す
PRE_SEC = 3.0
POST_SEC = 7.0

# outputs
WRITE_NPY = True
WRITE_EVENT_JSON = True
WRITE_STATIONS_FOR_EVENT = True
WRITE_MANIFEST = True

WRITE_HEADER = True
HEADER_FILE = DB_DIR / 'header.hdr'

CHANNEL_CODE = 'DASZ'  # comp=('Z',) 前提

MAX_EVENTS: int | None = None
EVENTNUM_WHITELIST: set[int] | None = None

SKIP_MISSING_EVENTS = True


# =========================
# util
# =========================
def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
	missing = [c for c in cols if c not in df.columns]
	if missing:
		raise ValueError(f'{name} missing columns: {missing}. cols={list(df.columns)}')


def _parse_nav_time_utc(series: pd.Series) -> pd.Series:
	return pd.to_datetime(
		series.astype(str),
		utc=True,
		errors='raise',
		format='%Y%m%d %H:%M:%S.%f',
	)


def _read_events(event_csv: Path) -> pd.DataFrame:
	if not event_csv.exists():
		raise FileNotFoundError(f'EVENT_CSV not found: {event_csv}')

	df = pd.read_csv(event_csv)
	_require_cols(df, [EVENTNUM_COL, NAV_TIME_COL], 'EVENT_CSV')

	out = df.copy()
	out[EVENTNUM_COL] = out[EVENTNUM_COL].astype(int)
	out['file_time_utc'] = _parse_nav_time_utc(out[NAV_TIME_COL])
	out['file_time_ms'] = (out['file_time_utc'].astype('int64') // 10**6).astype(
		np.int64
	)

	if EVENTNUM_WHITELIST is not None:
		out = out[out[EVENTNUM_COL].isin(set(EVENTNUM_WHITELIST))].reset_index(
			drop=True
		)

	out = out.sort_values('file_time_ms').reset_index(drop=True)

	if MAX_EVENTS is not None and int(MAX_EVENTS) > 0:
		out = out.iloc[: int(MAX_EVENTS)].reset_index(drop=True)

	return out


def _open_zarr(path: Path) -> tuple[zarr.hierarchy.Group, float, int, int, int]:
	if not path.exists():
		raise FileNotFoundError(f'ZARR_PATH not found: {path}')

	root = zarr.open_group(str(path), mode='r')
	for k in (ZARR_KEY_BLOCK, ZARR_KEY_START_MS, ZARR_KEY_DONE):
		if k not in root:
			raise KeyError(f'Zarr key not found: {k}. keys={list(root.array_keys())}')

	block = root[ZARR_KEY_BLOCK]
	if block.ndim != 3:
		raise ValueError(f'block must be 3D (B,C,T). got shape={block.shape}')

	b, c, t = int(block.shape[0]), int(block.shape[1]), int(block.shape[2])

	fs = None
	for key in ('fs_out_hz', 'fs_hz', 'fs'):
		if key in root.attrs:
			fs = float(root.attrs[key])
			break
	if fs is None:
		raise ValueError('Zarr attrs missing sampling rate (fs_out_hz/fs_hz/fs)')

	return root, float(fs), b, c, t


def _load_station_meta(path: Path) -> pd.DataFrame:
	if not path.exists():
		raise FileNotFoundError(f'STATION_META_CSV not found: {path}')

	sta = pd.read_csv(path)
	_require_cols(
		sta, [STATION_NAME_COL, ORDER_COL, FULL_CHANNEL_COL, WELL_COL], 'station_meta'
	)
	if WRITE_HEADER:
		_require_cols(sta, ['lat', 'lon', 'elev_m'], 'station_meta(for header)')

	sta[ORDER_COL] = sta[ORDER_COL].astype(int)
	sta[FULL_CHANNEL_COL] = sta[FULL_CHANNEL_COL].astype(int)
	sta[STATION_NAME_COL] = sta[STATION_NAME_COL].astype(str)
	sta[WELL_COL] = sta[WELL_COL].astype(str)

	sta = sta.sort_values(ORDER_COL).reset_index(drop=True)

	if sta[ORDER_COL].min() != 0:
		raise ValueError(f'{ORDER_COL} must start at 0. min={sta[ORDER_COL].min()}')
	if sta[ORDER_COL].nunique() != len(sta):
		raise ValueError(f'{ORDER_COL} must be unique per row')
	if sta[ORDER_COL].max() != len(sta) - 1:
		raise ValueError(
			f'{ORDER_COL} must end at N-1. max={sta[ORDER_COL].max()} N={len(sta)}'
		)

	return sta


def _build_select_idx_for_keep_superset_zarr(
	sta: pd.DataFrame, zarr_c: int
) -> np.ndarray:
	a0, a1 = int(ZARR_WELL_A_KEEP_0BASED_INCL[0]), int(ZARR_WELL_A_KEEP_0BASED_INCL[1])
	b0, b1 = int(ZARR_WELL_B_KEEP_0BASED_INCL[0]), int(ZARR_WELL_B_KEEP_0BASED_INCL[1])
	if a1 < a0 or b1 < b0:
		raise ValueError('Bad ZARR_WELL_*_KEEP_0BASED_INCL')

	a_len = a1 - a0 + 1
	b_len = b1 - b0 + 1
	expected_c = a_len + b_len
	if int(zarr_c) != int(expected_c):
		raise ValueError(
			f'Unexpected Zarr C={zarr_c}. keep-superset expected C={expected_c} '
			f'from A={ZARR_WELL_A_KEEP_0BASED_INCL} B={ZARR_WELL_B_KEEP_0BASED_INCL}'
		)

	well = sta[WELL_COL].to_numpy(dtype=str)
	ch = sta[FULL_CHANNEL_COL].to_numpy(dtype=int)

	is_a = well == WELL_A_NAME
	is_b = well == WELL_B_NAME
	if not np.all(is_a | is_b):
		bad = np.unique(well[~(is_a | is_b)]).tolist()
		raise ValueError(f'station_meta has unknown well values: {bad}')

	sel = np.empty(len(sta), dtype=int)
	sel[is_a] = ch[is_a] - int(a0)
	sel[is_b] = int(a_len) + (ch[is_b] - int(b0))

	if sel.min() < 0 or sel.max() >= int(zarr_c):
		raise ValueError(
			f'select_idx out of range: min={sel.min()} max={sel.max()} zarr_c={zarr_c}'
		)
	return sel


def _detect_mode_and_select_idx(
	sta: pd.DataFrame, zarr_c: int
) -> tuple[str, np.ndarray | None]:
	if int(zarr_c) == len(sta):
		return 'keep_only_zarr', None

	if int(zarr_c) == int(FULL_CHANNEL_COUNT):
		sel = sta[FULL_CHANNEL_COL].to_numpy(dtype=int)
		if sel.min() < 0 or sel.max() >= int(FULL_CHANNEL_COUNT):
			raise ValueError('station_meta channel out of range for full zarr')
		return 'full_2432_zarr', sel

	sel = _build_select_idx_for_keep_superset_zarr(sta, int(zarr_c))
	return 'keep_superset_zarr', sel


def _sort_blocks_by_time(
	start_ms_raw: np.ndarray, done_raw: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	if start_ms_raw.ndim != 1 or done_raw.ndim != 1:
		raise ValueError('start_ms_raw/done_raw must be 1D')
	if len(start_ms_raw) != len(done_raw):
		raise ValueError('start_ms_raw/done_raw length mismatch')

	sort_idx = np.argsort(start_ms_raw)
	return (
		start_ms_raw[sort_idx].astype(np.int64),
		done_raw[sort_idx].astype(bool),
		sort_idx.astype(np.int64),
	)


def _block_duration_ms(fs: float, t_samples: int) -> int:
	dur_ms = int(round((float(t_samples) / float(fs)) * 1000.0))
	if dur_ms <= 0:
		raise ValueError('bad block duration')
	return dur_ms


def _extract_window_from_blocks(
	block_arr,
	start_ms_sorted: np.ndarray,
	done_sorted: np.ndarray,
	sort_idx: np.ndarray,
	fs: float,
	t_samples: int,
	win_start_ms: int,
	win_end_ms: int,
	*,
	select_idx: np.ndarray | None,
) -> tuple[np.ndarray, int, int]:
	dur_ms = _block_duration_ms(fs, t_samples)
	if win_end_ms <= win_start_ms:
		raise ValueError('win_end_ms must be > win_start_ms')

	i0 = int(np.searchsorted(start_ms_sorted, win_start_ms, side='right') - 1)
	if i0 < 0:
		raise ValueError('window starts before Zarr begins')

	parts: list[np.ndarray] = []
	j = i0
	last = i0

	while j < len(start_ms_sorted):
		s0 = int(start_ms_sorted[j])
		e0 = int(s0 + dur_ms)

		if e0 <= win_start_ms:
			j += 1
			continue
		if s0 >= win_end_ms:
			break

		if not bool(done_sorted[j]):
			raise ValueError(f'block idx(sorted)={j} done=False')

		seg_start = max(win_start_ms, s0)
		seg_end = min(win_end_ms, e0)

		a0 = int(np.floor(((seg_start - s0) / 1000.0) * float(fs)))
		a1 = int(np.ceil(((seg_end - s0) / 1000.0) * float(fs)))
		a0 = max(0, a0)
		a1 = min(t_samples, a1)
		if a1 <= a0:
			j += 1
			continue

		orig_j = int(sort_idx[j])
		x = np.asarray(block_arr[orig_j], dtype=np.float32)  # (Czarr, T)
		if select_idx is not None:
			x = x[select_idx, :]
		parts.append(x[:, a0:a1])
		last = j

		if e0 >= win_end_ms:
			break
		j += 1

	if len(parts) == 0:
		raise ValueError('no waveform parts extracted')

	xcat = np.concatenate(parts, axis=1)

	expected = int(round(((win_end_ms - win_start_ms) / 1000.0) * float(fs)))
	if xcat.shape[1] < expected:
		raise ValueError(
			f'extracted samples too short: got={xcat.shape[1]} expected={expected}'
		)
	if xcat.shape[1] > expected:
		xcat = xcat[:, :expected]

	return xcat.astype(np.float32), int(i0), int(last)


def _write_header(sta: pd.DataFrame, out_path: Path) -> None:
	lines: list[str] = []
	for station, lat, lon, elev in sta[
		[STATION_NAME_COL, 'lat', 'lon', 'elev_m']
	].itertuples(index=False, name=None):
		lines.append(f'{station} {float(lat):.6f} {float(lon):.6f} {float(elev):.2f}')
	out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


# =========================
# main
# =========================
def main() -> None:
	OUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

	events = _read_events(EVENT_CSV)
	sta = _load_station_meta(STATION_META_CSV)

	root, fs, b, c, t = _open_zarr(ZARR_PATH)
	start_ms_raw = np.asarray(root[ZARR_KEY_START_MS][:], dtype=np.int64)
	done_raw = np.asarray(root[ZARR_KEY_DONE][:], dtype=bool)
	block_arr = root[ZARR_KEY_BLOCK]

	if len(start_ms_raw) != int(b) or len(done_raw) != int(b):
		raise ValueError('starttime_utc_ms/done length mismatch with B dimension')

	start_ms, done, sort_idx = _sort_blocks_by_time(start_ms_raw, done_raw)
	mode, select_idx = _detect_mode_and_select_idx(sta, int(c))
	station_names = sta[STATION_NAME_COL].astype(str).tolist()

	if WRITE_HEADER:
		DB_DIR.mkdir(parents=True, exist_ok=True)
		_write_header(sta, HEADER_FILE)
		sta.to_csv(DB_DIR / 'stations_loki.csv', index=False)

	dur_ms = _block_duration_ms(float(fs), int(t))
	pre_ms = int(round(float(PRE_SEC) * 1000.0))
	post_ms = int(round(float(POST_SEC) * 1000.0))

	zarr_t0 = int(start_ms[0])
	zarr_t1 = int(start_ms[-1] + dur_ms)

	print(f'Zarr: B={b} C={c} T={t} fs={fs} mode={mode}')
	print(
		f'Events: {len(events)} center=file_time_utc window=[-{PRE_SEC}s,+{POST_SEC}s]'
	)

	manifest_rows: list[dict] = []

	for row in events.itertuples(index=False):
		evnum = int(getattr(row, EVENTNUM_COL))
		file_time_ms = int(row.file_time_ms)

		win_start_ms = int(file_time_ms - pre_ms)
		win_end_ms = int(file_time_ms + post_ms)

		if win_start_ms < zarr_t0 or win_end_ms > zarr_t1:
			if SKIP_MISSING_EVENTS:
				continue
			raise ValueError(f'window out of zarr range: eventNum={evnum}')

		x_win, b_first, b_last = _extract_window_from_blocks(
			block_arr,
			start_ms,
			done,
			sort_idx,
			fs=float(fs),
			t_samples=int(t),
			win_start_ms=int(win_start_ms),
			win_end_ms=int(win_end_ms),
			select_idx=select_idx,
		)

		ev_dir = OUT_BASE_DIR / f'event_{evnum:06d}'
		ev_dir.mkdir(parents=True, exist_ok=True)

		npy_path = ev_dir / 'waveform.npy'
		if WRITE_NPY:
			np.save(npy_path, x_win.astype(np.float32), allow_pickle=False)

		meta: dict = {
			'event_id': f'event_{evnum:06d}',
			'eventNum': evnum,
			'mode': mode,
			'fs_hz': float(fs),
			'channel_code': str(CHANNEL_CODE),
			'pre_sec': float(PRE_SEC),
			'post_sec': float(POST_SEC),
			'file_time_utc': pd.Timestamp(
				int(file_time_ms), unit='ms', tz='UTC'
			).isoformat(),
			'window_start_utc': pd.Timestamp(
				int(win_start_ms), unit='ms', tz='UTC'
			).isoformat(),
			'window_end_utc': pd.Timestamp(
				int(win_end_ms), unit='ms', tz='UTC'
			).isoformat(),
			'zarr_block_first_sorted': int(b_first),
			'zarr_block_last_sorted': int(b_last),
			'shape': {'C': int(x_win.shape[0]), 'T': int(x_win.shape[1])},
			'stations': station_names,
		}

		for k in ('snr', 'easting_m', 'northing_m', 'tvd_m', 'dt_sec'):
			if hasattr(row, k):
				v = getattr(row, k)
				if isinstance(v, (np.floating, float)) and np.isnan(v):
					continue
				meta[k] = (
					float(v) if isinstance(v, (np.floating, float, int)) else str(v)
				)

		if WRITE_EVENT_JSON:
			(ev_dir / 'meta.json').write_text(
				json.dumps(meta, indent=2), encoding='utf-8'
			)

		if WRITE_STATIONS_FOR_EVENT:
			sta.to_csv(ev_dir / 'stations.csv', index=False)

		if WRITE_MANIFEST:
			manifest_rows.append(
				{
					'eventNum': evnum,
					'file_time_utc': meta['file_time_utc'],
					'window_start_utc': meta['window_start_utc'],
					'window_end_utc': meta['window_end_utc'],
					'zarr_block_first_sorted': int(b_first),
					'zarr_block_last_sorted': int(b_last),
					'C': int(x_win.shape[0]),
					'T': int(x_win.shape[1]),
					'npy_path': str(npy_path) if WRITE_NPY else '',
				}
			)

	if WRITE_MANIFEST:
		pd.DataFrame(manifest_rows).to_csv(OUT_BASE_DIR / 'manifest.csv', index=False)

	print(f'done. out={OUT_BASE_DIR}')


if __name__ == '__main__':
	main()
