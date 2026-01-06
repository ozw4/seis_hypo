# %%
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from jma.station_reader import read_hinet_channel_table
from jma.win32_reader import get_evt_info, read_win32

_CH_HEX_PAT = re.compile(rb'^\s*([0-9A-Fa-f]{4,6})\b')


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
			continue  # 空行いらないなら捨てる
		if raw.lstrip().startswith(b'#'):
			pending_comments.append(raw)  # 次に残るデータ行が来たら出す
			continue

		m = _CH_HEX_PAT.match(raw)
		if m is None:
			# 形式不明行はコメント扱いで、残る観測点にだけ付ける
			pending_comments.append(raw)
			continue

		ch_hex = m.group(1).decode('ascii').upper()
		if ch_hex in keep_ch_hex:
			if pending_comments:
				out_lines.extend(pending_comments[-1:])
				pending_comments.clear()
			out_lines.append(raw)
			# print(raw)
	out_path.write_bytes(b''.join(out_lines))


def make_active_ch_for_evt(
	evt_path: Path,
	ch_path: Path,
	*,
	out_ch_path: Path | None = None,
	scan_rate_blocks: int = 1000,
) -> Path:
	info = get_evt_info(evt_path, scan_rate_blocks=scan_rate_blocks)

	# .ch を DataFrame 化（ch_hex 行順が read_win32 出力行順と対応する前提）
	station_df = read_hinet_channel_table(ch_path)

	arr = read_win32(
		evt_path,
		station_df,
		base_sampling_rate_HZ=info.base_sampling_rate_hz,
		duration_SECOND=info.span_seconds,
	)

	arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
	active_mask = np.any(arr != 0.0, axis=1)
	active_hex = {
		str(x).upper() for x in station_df.loc[active_mask, 'ch_hex'].tolist()
	}

	if out_ch_path is None:
		out_ch_path = ch_path.with_name(f'{ch_path.stem}_active.ch')

	# print(
	# f'[active] {evt_path.name}: {int(active_mask.sum())}/{active_mask.size} channels, writing to {out_ch_path}'
	# )
	write_active_ch_file(ch_path, keep_ch_hex=active_hex, out_path=out_ch_path)

	return out_ch_path


# =========================
# 使い方（ここを直書き）
# =========================
EVT_DIR = Path('/workspace/data/waveform/jma/event').resolve()


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
			except Exception as e:
				print(f'Error processing {evt_path}: {e}')


if __name__ == '__main__':
	main()
