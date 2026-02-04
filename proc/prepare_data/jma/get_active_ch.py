# %%
from __future__ import annotations

from pathlib import Path

from jma.prepare.active_channel import make_active_ch_for_evt

# =========================
# 設定（直書き）
# =========================
EVT_DIR = Path('/workspace/data/waveform/jma/event').resolve()
TARGET_FS_HZ = 100
SCAN_RATE_BLOCKS = 1000
SKIP_IF_EXISTS = True


def main() -> None:
	# event_dir/*.evt と同名の *.ch を探して、*_active.ch を作る
	for event_dir in sorted(EVT_DIR.glob('*')):
		if not event_dir.is_dir():
			continue
		if SKIP_IF_EXISTS:
			active_ch_files = list(event_dir.glob('*_active.ch'))
			if active_ch_files:
				continue
		for evt_path in sorted(event_dir.glob('*.evt')):
			ch_path = evt_path.with_suffix('.ch')
			if not ch_path.is_file():
				continue

			try:
				make_active_ch_for_evt(
					evt_path,
					ch_path,
					target_sampling_rate_HZ=TARGET_FS_HZ,
					scan_rate_blocks=SCAN_RATE_BLOCKS,
				)
			except ValueError as e:
				# タイムスタンプ無し等は「棄却」でOKという運用に合わせ、警告してスキップ
				msg = str(e)

				print(f'[warn] skip {evt_path}: {msg}')
				continue
				# raise


if __name__ == '__main__':
	main()

# %%
