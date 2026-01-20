# %%


from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jma.picks import (
	build_pick_table_for_event,
	find_event_id_by_origin,
	pick_time_to_index,
)
from jma.prepare.event_txt import read_origin_jst_iso
from jma.station_reader import read_hinet_channel_table
from jma.stationcode_common import normalize_code
from jma.stationcode_mappingdb import load_mapping_db
from jma.stationcode_presence import load_presence_db
from jma.win32_reader import get_evt_info, read_win32_resampled
from viz.gather import plot_gather

# =========================
# 設定（直書き）
# =========================

WIN_EVENT_DIR = Path('/workspace/data/waveform/jma/event').resolve()

# JMA検測CSV
MEAS_CSV = Path(
	'/workspace/data/arrivetime/JMA/arrivetime_measurements_2023.0.csv'
).resolve()
EPI_CSV = Path(
	'/workspace/data/arrivetime/JMA/arrivetime_epicenters_2023.0.csv'
).resolve()

# 現行ルールのマッピング（match_out_final）
MAPPING_REPORT_CSV = Path(
	'/workspace/proc/prepare_data/jma/stationcode_match/v1/match_out_final/mapping_report.csv'
).resolve()
NEAR0_CSV = Path(
	'/workspace/proc/prepare_data/jma/stationcode_match/near0_suggestions.csv'
).resolve()

# presence（月別）
PRES_CSV = Path(
	'/workspace/proc/prepare_data/jma/stationcode_match/v1/snapshots/monthly/monthly_presence.csv'
).resolve()

EVENT_DIR_GLOB = 'D2023*'
EVENT_FILE_GLOB = '*.evt'

TARGET_FS_HZ = 100
SCAN_RATE_BLOCKS = 1000

SAVE_DPI = 150
SAVE_SUFFIX = '_gather.png'

# 可視化なので、event_id 解決や pick 生成に失敗しても「波形だけ描く」ことを許容する
ALLOW_NO_PICKS = True


# =========================
# util
# =========================


def _event_month_from_origin_iso(origin_iso: str) -> str:
	t = pd.to_datetime(origin_iso, format='ISO8601', errors='raise')
	return f'{t.year:04d}-{t.month:02d}'


def _pick_idx_arrays(
	station_df: pd.DataFrame,
	pick_df: pd.DataFrame,
	*,
	fs_hz: float,
	t_start,
	n_t: int,
) -> tuple[np.ndarray, np.ndarray]:
	n = len(station_df)
	p_idx = np.full(n, np.nan, dtype=float)
	s_idx = np.full(n, np.nan, dtype=float)

	st_keys = station_df['station'].astype(str).map(normalize_code).to_numpy()

	for i, sta in enumerate(st_keys):
		if sta not in pick_df.index:
			continue
		row = pick_df.loc[sta]
		p_idx[i] = pick_time_to_index(
			row.get('p_time'),
			fs_hz=float(fs_hz),
			t_start=t_start,
			n_t=int(n_t),
		)
		s_idx[i] = pick_time_to_index(
			row.get('s_time'),
			fs_hz=float(fs_hz),
			t_start=t_start,
			n_t=int(n_t),
		)

	return p_idx, s_idx


# =========================
# main
# =========================


def main() -> None:
	if not WIN_EVENT_DIR.is_dir():
		raise FileNotFoundError(WIN_EVENT_DIR)
	if not MEAS_CSV.is_file():
		raise FileNotFoundError(MEAS_CSV)
	if not EPI_CSV.is_file():
		raise FileNotFoundError(EPI_CSV)
	if not MAPPING_REPORT_CSV.is_file():
		raise FileNotFoundError(MAPPING_REPORT_CSV)
	if not PRES_CSV.is_file():
		raise FileNotFoundError(PRES_CSV)

	meas_df = pd.read_csv(MEAS_CSV, low_memory=False)
	epi_df = pd.read_csv(EPI_CSV, low_memory=False)

	req_meas = {
		'event_id',
		'station_code',
		'phase_name_1',
		'phase_name_2',
		'phase1_time',
		'phase2_time',
	}
	if not req_meas.issubset(meas_df.columns):
		raise ValueError(
			f'measurements csv missing columns: {sorted(req_meas - set(meas_df.columns))}'
		)

	req_epi = {'event_id', 'origin_time'}
	if not req_epi.issubset(epi_df.columns):
		raise ValueError(
			f'epicenters csv missing columns: {sorted(req_epi - set(epi_df.columns))}'
		)

	mdb = load_mapping_db(MAPPING_REPORT_CSV, NEAR0_CSV)
	pdb = load_presence_db(PRES_CSV)

	count = 0

	for event_dir in sorted(WIN_EVENT_DIR.glob(EVENT_DIR_GLOB)):
		if not event_dir.is_dir():
			continue

		for evt_path in sorted(event_dir.glob(EVENT_FILE_GLOB)):
			if not evt_path.is_file():
				continue

			txt_path = evt_path.with_suffix('.txt')
			ch_path = evt_path.with_name(f'{evt_path.stem}_active.ch')

			if not txt_path.is_file():
				raise FileNotFoundError(txt_path)
			if not ch_path.is_file():
				raise FileNotFoundError(ch_path)

			origin_iso = read_origin_jst_iso(txt_path)
			event_time = pd.to_datetime(
				origin_iso, format='ISO8601', errors='raise'
			).to_pydatetime()
			event_month = _event_month_from_origin_iso(origin_iso)

			event_id: int | None = None
			try:
				event_id = find_event_id_by_origin(epi_df, origin_iso, tol_seconds=0.5)
			except ValueError as e:
				if not ALLOW_NO_PICKS:
					raise
				print(
					f'[warn] cannot resolve event_id for origin_time={origin_iso}: {e}'
				)

			info = get_evt_info(evt_path, scan_rate_blocks=int(SCAN_RATE_BLOCKS))

			station_df = read_hinet_channel_table(ch_path)
			x = read_win32_resampled(
				evt_path,
				station_df,
				target_sampling_rate_HZ=int(TARGET_FS_HZ),
				duration_SECOND=int(info.span_seconds),
			)

			n_ch, n_t = x.shape
			st_used = station_df.iloc[:n_ch].copy()

			p_idx: np.ndarray | None = None
			s_idx: np.ndarray | None = None

			if event_id is not None:
				try:
					pick_df, _log_rows = build_pick_table_for_event(
						meas_df,
						event_id=int(event_id),
						event_month=event_month,
						mdb=mdb,
						pdb=pdb,
					)
					pick_df2 = pick_df.copy()
					pick_df2.index = pick_df2.index.astype(str).map(normalize_code)

					p_idx, s_idx = _pick_idx_arrays(
						st_used,
						pick_df2,
						fs_hz=float(TARGET_FS_HZ),
						t_start=info.start_time,
						n_t=int(n_t),
					)
				except Exception as e:
					if not ALLOW_NO_PICKS:
						raise
					print(f'[warn] cannot build picks for event_id={event_id}: {e!r}')

			n_station = st_used['station'].astype(str).map(normalize_code).nunique()
			title = f'{evt_path.name}  origin={origin_iso}  stations={n_station}  fs={TARGET_FS_HZ}Hz'
			if event_id is not None:
				title = f'{title}  event_id={event_id}'
			print(title)

			plot_gather(
				x,
				station_df=st_used,
				title=title,
				p_idx=p_idx,
				s_idx=s_idx,
				y_time='relative',
				fs=float(TARGET_FS_HZ),
				t_start=info.start_time,
				event_time=event_time,
				amp=1.0,
				detrend='linear',
				taper_frac=0.05,
			)

			out_png = evt_path.with_name(f'{evt_path.stem}{SAVE_SUFFIX}')
			plt.savefig(out_png, dpi=int(SAVE_DPI), bbox_inches='tight')
			plt.close()
			count += 1

	print(f'[done] saved gather images: {count}')


if __name__ == '__main__':
	main()
