# %%
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

# One or more directories that contain WIN32 *.cnt files
CNT_DIRS = [
	Path('/workspace/data/izu2009/continuous/0101'),
	Path('/workspace/data/izu2009/continuous/0203'),
	Path('/workspace/data/izu2009/continuous/0207'),
	Path('/workspace/data/izu2009/continuous/0301'),
]

# naive datetime is treated as JST
START_JST = dt.datetime(2009, 12, 17, 0, 0, 0)
END_JST = dt.datetime(2009, 12, 21, 0, 0, 0)  # exclusive

# If set, process only these network codes. None = all found in CNT_DIRS.
NETWORK_FILTER: set[str] | None = None

# Network-code -> channel table (.ch)
# MUST contain mapping for every network you process (no fallback).
CH_TABLE_BY_NETWORK: dict[str, Path] = {
	'0101': Path(
		'/workspace/proc/prepare_data/izu2009/download_continuous/continuous_ch47/0101/win_0101_200912170000_10m_aa3c27a4.ch'
	),
	'0203': Path(
		'/workspace/proc/prepare_data/izu2009/download_continuous/continuous_ch47/0203/win_0203_200912170000_10m_9a3c463f.ch'
	),
	'0207': Path(
		'/workspace/proc/prepare_data/izu2009/download_continuous/continuous_ch47/0207/win_0207_200912170000_10m_1c7df708.ch'
	),
	'0301': Path(
		'/workspace/proc/prepare_data/izu2009/download_continuous/continuous_ch47/0301/win_0301_200912170000_10m_4dd999af.ch'
	),
}

EQT_WEIGHTS = '/workspace/model_weight/010_Train_EqT_FT-STEAD_rot30_Hinet selftrain.pth'
EQT_IN_SAMPLES = 6000
EQT_OVERLAP = 3000
EQT_BATCH_STATIONS = 64

USE_RESAMPLED = True
RESAMPLED_MISSING_CHANNEL_POLICY = 'zero'  # 'raise' | 'drop' | 'zero'
TARGET_FS_HZ = 100.0

DET_GATE_ENABLE = True
DET_THRESHOLD = 0.30
P_THRESHOLD = 0.10
S_THRESHOLD = 0.10
MIN_PICK_SEP_SAMPLES = 50
OVERLAP_MERGE = 'max'

OUT_DIR = Path('proc/run_continuous/pick/win32/out')
PRINT_EVERY_WINDOWS = 50
# =========================


def _resolve_ch_table(network_code: str) -> Path:
	p = CH_TABLE_BY_NETWORK.get(str(network_code))
	if p is None:
		raise ValueError(
			f'CH_TABLE_BY_NETWORK has no entry for network: {network_code}'
		)
	return p


def _list_cnt_by_network_in_period() -> dict[str, list[Path]]:
	groups: dict[str, list[tuple[dt.datetime, Path]]] = {}

	for d in CNT_DIRS:
		for p in d.glob('*.cnt'):
			info = parse_win32_cnt_filename(p)
			file_start = info.start_jst
			file_end = file_start + dt.timedelta(minutes=int(info.span_min))
			if file_end <= START_JST or file_start >= END_JST:
				continue

			net = str(info.network_code)
			if NETWORK_FILTER is not None and net not in NETWORK_FILTER:
				continue

			groups.setdefault(net, []).append((file_start, p))

	if not groups:
		raise ValueError(f'no .cnt files selected in period: {START_JST} - {END_JST}')

	out: dict[str, list[Path]] = {}
	for net, items in groups.items():
		items.sort(key=lambda x: x[0])
		out[net] = [p for _, p in items]
	return out


def _ensure_ch_tables_exist(cnt_by_net: dict[str, list[Path]]) -> None:
	missing = [net for net in cnt_by_net if net not in CH_TABLE_BY_NETWORK]
	if missing:
		raise ValueError(f'missing .ch mapping for networks: {missing}')


def _split_contiguous(cnt_paths: list[Path]) -> list[list[Path]]:
	if not cnt_paths:
		return []

	segs: list[list[Path]] = []
	cur: list[Path] = []
	prev_end: dt.datetime | None = None

	for p in cnt_paths:
		info = parse_win32_cnt_filename(p)
		start = info.start_jst
		end = start + dt.timedelta(minutes=int(info.span_min))

		if prev_end is None or start == prev_end:
			cur.append(p)
		else:
			segs.append(cur)
			cur = [p]

		prev_end = end

	if cur:
		segs.append(cur)
	return segs


def _out_csv_path(network_code: str, seg_idx: int, n_seg: int) -> Path:
	OUT_DIR.mkdir(parents=True, exist_ok=True)
	if int(n_seg) <= 1:
		return OUT_DIR / f'eqt_picks_win32_{network_code}.csv'
	return OUT_DIR / f'eqt_picks_win32_{network_code}_seg{seg_idx:02d}.csv'


def main() -> None:
	cnt_by_net = _list_cnt_by_network_in_period()
	_ensure_ch_tables_exist(cnt_by_net)

	for network_code, cnt_paths in sorted(cnt_by_net.items(), key=lambda x: x[0]):
		print(network_code, len(cnt_paths))
		ch_path = _resolve_ch_table(network_code)
		segs = _split_contiguous(cnt_paths)

		print(
			f'[INFO] network={network_code} cnt_files={len(cnt_paths)} segments={len(segs)}'
		)

		for i, seg_paths in enumerate(segs):
			out_csv = _out_csv_path(network_code, seg_idx=i, n_seg=len(segs))

			pipeline_win32_eqt_pick_to_csv(
				cnt_paths=seg_paths,
				ch_table=ch_path,
				out_csv=out_csv,
				eqt_weights=EQT_WEIGHTS,
				eqt_in_samples=EQT_IN_SAMPLES,
				eqt_overlap=EQT_OVERLAP,
				eqt_batch_stations=EQT_BATCH_STATIONS,
				use_resampled=USE_RESAMPLED,
				resampled_missing_channel_policy=RESAMPLED_MISSING_CHANNEL_POLICY,
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
