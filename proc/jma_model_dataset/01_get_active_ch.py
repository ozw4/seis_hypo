from __future__ import annotations

import argparse
from pathlib import Path

from jma_model_dataset.step1_active_channel import build_active_ch_for_event

DEFAULT_TARGET_FS_HZ = 100
DEFAULT_SCAN_RATE_BLOCKS = 1000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			'Create flow-scoped *_active.ch files for the '
			'JMA model dataset Step1 workflow.'
		)
	)
	parser.add_argument(
		'event_dirs',
		nargs='+',
		type=Path,
		help=(
			'Event directory path(s) whose raw inputs are under raw/ and whose '
			'flow-scoped *_active.ch will be written under flows/jma_model_dataset/active/'
		),
	)
	parser.add_argument(
		'--target-fs-hz',
		type=int,
		default=DEFAULT_TARGET_FS_HZ,
		help=f'Target sampling rate in Hz (default: {DEFAULT_TARGET_FS_HZ})',
	)
	parser.add_argument(
		'--scan-rate-blocks',
		type=int,
		default=DEFAULT_SCAN_RATE_BLOCKS,
		help=(
			'Number of WIN32 blocks to scan when detecting per-channel sampling rates '
			f'(default: {DEFAULT_SCAN_RATE_BLOCKS})'
		),
	)
	parser.add_argument(
		'--skip-if-exists',
		action='store_true',
		help='Skip an event directory when its flow-scoped *_active.ch already exists',
	)
	return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
	args = parse_args(argv)

	for event_dir in args.event_dirs:
		build_active_ch_for_event(
			event_dir,
			target_sampling_rate_HZ=args.target_fs_hz,
			scan_rate_blocks=args.scan_rate_blocks,
			skip_if_exists=args.skip_if_exists,
		)


if __name__ == '__main__':
	main()
