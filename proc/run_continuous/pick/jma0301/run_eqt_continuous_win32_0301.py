from __future__ import annotations

import datetime as dt
from pathlib import Path

from pipelines.win32_eqt_continuous_pipelines import (
	parse_win32_cnt_filename,
	pipeline_win32_eqt_pick_to_csv,
)

# =========================
# Parameters (edit here)
# =========================
CNT_DIR = Path('/workspace/data/izu2009/continuous/0301')
CH_PATH = Path(
	'/workspace/proc/prepare_data/izu2009/download_continuous/continuous_ch47/0301/win_0301_200912170000_10m_4dd999af.ch'
)

# naive datetime is treated as JST
START_JST = dt.datetime(2009, 12, 17, 0, 0, 0)
END_JST = dt.datetime(2009, 12, 21, 0, 0, 0)  # exclusive

EQT_WEIGHTS = '/workspace/model_weight/010_Train_EqT_FT-STEAD_rot30_Hinet selftrain.pth'
EQT_IN_SAMPLES = 6000
EQT_OVERLAP = 3000
EQT_BATCH_STATIONS = 64

USE_RESAMPLED = False
TARGET_FS_HZ = 100.0

DET_GATE_ENABLE = True
DET_THRESHOLD = 0.30
P_THRESHOLD = 0.10
S_THRESHOLD = 0.10
MIN_PICK_SEP_SAMPLES = 50
OVERLAP_MERGE = 'max'

OUT_CSV = Path('proc/run_continuous/pick/jma0301/out/eqt_picks_win32_0301.csv')
PRINT_EVERY_WINDOWS = 50
# =========================


def _list_cnt_in_period() -> list[Path]:
	cands: list[tuple[dt.datetime, Path]] = []
	for p in CNT_DIR.glob('*.cnt'):
		info = parse_win32_cnt_filename(p)
		file_start = info.start_jst
		file_end = file_start + dt.timedelta(minutes=int(info.span_min))
		if file_end <= START_JST or file_start >= END_JST:
			continue
		cands.append((file_start, p))

	if not cands:
		raise ValueError(f'no .cnt files selected in period: {START_JST} - {END_JST}')

	cands.sort(key=lambda x: x[0])
	return [x[1] for x in cands]


def main() -> None:
	cnt_paths = _list_cnt_in_period()
	pipeline_win32_eqt_pick_to_csv(
		cnt_paths=cnt_paths,
		ch_table=CH_PATH,
		out_csv=OUT_CSV,
		eqt_weights=EQT_WEIGHTS,
		eqt_in_samples=EQT_IN_SAMPLES,
		eqt_overlap=EQT_OVERLAP,
		eqt_batch_stations=EQT_BATCH_STATIONS,
		use_resampled=USE_RESAMPLED,
		target_fs_hz=TARGET_FS_HZ,
		det_gate_enable=DET_GATE_ENABLE,
		det_threshold=DET_THRESHOLD,
		p_threshold=P_THRESHOLD,
		s_threshold=S_THRESHOLD,
		min_pick_sep_samples=MIN_PICK_SEP_SAMPLES,
		overlap_merge=OVERLAP_MERGE,
		include_network_code=True,
		print_every_windows=PRINT_EVERY_WINDOWS,
	)


if __name__ == '__main__':
	main()
