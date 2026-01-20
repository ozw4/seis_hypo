# %%
# proc/prepare_data/jma/run_build_event_inventory.py
from __future__ import annotations
from pathlib import Path

from jma.prepare.inventory import build_inventory, write_inventory_outputs

# =========================
# 設定（ここを直書きでOK）
# =========================

# イベントディレクトリ（添付zipを展開したフォルダを指定）
EVENT_DIR = Path('/workspace/data/waveform/jma/event/D20230118000041_20').resolve()

# continuous サブディレクトリ名
CONT_SUBDIR = 'continuous'

# 出力先
OUTDIR = EVENT_DIR / 'inventory'

# バージョン
SCHEMA_VERSION = 'event_inventory_v1'

# get_evt_info の sampling rate 走査ブロック数（多めにしておく）
EVT_INFO_SCAN_RATE_BLOCKS = 60

# scan_channel_sampling_rate_map_win32 の secondblock 走査上限（None で全走査）
SCAN_MAX_SECOND_BLOCKS = None

# component 揺らぎ吸収の優先度（小さいほど強い）
# axis U/N/E に対して、どの表記を優先するか
COMP_PRIORITY = {
	'U': ['U', 'Z'],
	'N': ['N', 'Y'],
	'E': ['E', 'X'],
}

# 末尾1文字で軸推定を許可する文字（wU, xxN, ...）
AXIS_TAIL_CHARS = set(['U', 'N', 'E', 'Z', 'X', 'Y'])


# =========================
# 実装
# =========================


def _build_inventory_for_event(event_dir: Path) -> dict:
	result = build_inventory(
		event_dir,
		cont_subdir=CONT_SUBDIR,
		schema_version=SCHEMA_VERSION,
		evt_info_scan_rate_blocks=EVT_INFO_SCAN_RATE_BLOCKS,
		scan_max_second_blocks=SCAN_MAX_SECOND_BLOCKS,
		comp_priority=COMP_PRIORITY,
		axis_tail_chars=AXIS_TAIL_CHARS,
	)
	write_inventory_outputs(event_dir, OUTDIR, SCHEMA_VERSION, result)
	return result.inventory


def main() -> None:
	_build_inventory_for_event(EVENT_DIR)


if __name__ == '__main__':
	main()

# %%
