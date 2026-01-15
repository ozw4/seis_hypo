# %%
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from jma.station_reader import read_hinet_channel_table
from jma.win32_reader import (
	get_evt_info,
	read_win32_resampled,
	scan_channel_sampling_rate_map_win32,
)

_CH_HEX_PAT = re.compile(rb'^\s*([0-9A-Fa-f]{4,6})\b')

# =========================
# 設定（直書き）
# =========================
EVT_DIR = Path('/workspace/data/waveform/jma/event').resolve()
TARGET_FS_HZ = 100
SCAN_RATE_BLOCKS = 1000


def write_active_ch_file(
	ch_path: Path,
	*,
	keep_ch_hex: set[str],
	out_path: Path,
) -> None:
	src = ch_path.read_bytes().splitlines(keepends=True)

	pending_comments: list[bytes] = []
	out_lines: list[bytes] = []

	for raw in src:
		s = raw.strip()
		if not s:
			continue
		if raw.lstrip().startswith(b'#'):
			pending_comments.append(raw)
			continue

		m = _CH_HEX_PAT.match(raw)
		if m is None:
			pending_comments.append(raw)
			continue

		ch_hex = m.group(1).decode('ascii').upper()
		if ch_hex in keep_ch_hex:
			if pending_comments:
				out_lines.extend(pending_comments[-1:])
				pending_comments.clear()
			out_lines.append(raw)

	out_path.write_bytes(b''.join(out_lines))


def _filter_station_df_to_win32_present(station_df: pd.DataFrame, evt_path: Path):
	# 選択した .ch のうち、WIN32本体に実在する ch_int だけを残す
	fs_by_ch = scan_channel_sampling_rate_map_win32(evt_path)
	present = set(int(k) for k in fs_by_ch.keys())

	ch_int = station_df['ch_int'].astype(int)
	keep = ch_int.isin(present)
	out = station_df.loc[keep].copy()
	return out, int(keep.sum()), int((~keep).sum())


def make_active_ch_for_evt(
	evt_path: Path,
	ch_path: Path,
	*,
	out_ch_path: Path | None = None,
) -> Path:
	info = get_evt_info(evt_path, scan_rate_blocks=SCAN_RATE_BLOCKS)

	# .ch を DataFrame 化（行順を維持）
	station_df_all = read_hinet_channel_table(ch_path)

	station_df, n_keep, n_drop = _filter_station_df_to_win32_present(
		station_df_all, evt_path
	)
	if station_df.empty:
		raise ValueError(
			f'no WIN32-present channels in {ch_path.name} for {evt_path.name}'
		)

	# resampled 読み込み（混在fsを内側で分割→TARGET_FSへ統一）
	arr = read_win32_resampled(
		evt_path,
		station_df,
		target_sampling_rate_HZ=int(TARGET_FS_HZ),
		duration_SECOND=int(info.span_seconds),
	)

	arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
	active_mask = np.any(arr != 0.0, axis=1)
	active_hex = {
		str(x).upper() for x in station_df.loc[active_mask, 'ch_hex'].tolist()
	}

	if out_ch_path is None:
		out_ch_path = ch_path.with_name(f'{ch_path.stem}_active.ch')

	write_active_ch_file(ch_path, keep_ch_hex=active_hex, out_path=out_ch_path)

	print(
		f'[active_ch] {evt_path.name}: '
		f'kept={n_keep} dropped_not_in_win32={n_drop} '
		f'active={int(active_mask.sum())} -> {out_ch_path.name}'
	)

	return out_ch_path


def main() -> None:
	# event_dir/*.evt と同名の *.ch を探して、*_active.ch を作る
	for event_dir in sorted(EVT_DIR.glob('*')):
		if not event_dir.is_dir():
			continue

		for evt_path in sorted(event_dir.glob('*.evt')):
			ch_path = evt_path.with_suffix('.ch')
			if not ch_path.is_file():
				continue

			try:
				make_active_ch_for_evt(evt_path, ch_path)
			except ValueError as e:
				# タイムスタンプ無し等は「棄却」でOKという運用に合わせ、警告してスキップ
				msg = str(e)
				if 'no valid timestamps found' in msg:
					print(f'[warn] skip {evt_path}: {msg}')
					continue
				raise


if __name__ == '__main__':
	main()
