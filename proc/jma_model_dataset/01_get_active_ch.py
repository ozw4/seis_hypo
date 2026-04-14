# ruff: noqa: INP001
"""Build flow-scoped active channel files for JMA model dataset events."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from jma.prepare.event_paths import resolve_single_evt
from jma_model_dataset.paths import raw_root
from jma_model_dataset.step1_active_channel import build_active_ch_for_event

DEFAULT_TARGET_FS_HZ = 100
DEFAULT_SCAN_RATE_BLOCKS = 1000

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	"""Parse command-line arguments for the active-channel batch runner."""
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
			'flow-scoped *_active.ch will be written under '
			'flows/jma_model_dataset/active/'
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


def _configure_logging() -> None:
	if logging.getLogger().handlers:
		return
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s %(levelname)s %(name)s: %(message)s',
	)


def _resolve_evt_path_for_log(event_dir: Path) -> Path | None:
	raw_dir = raw_root(event_dir)
	if not raw_dir.is_dir():
		return None
	return resolve_single_evt(raw_dir, allow_none=True)


def main(argv: list[str] | None = None) -> None:
	"""Process each event directory and continue past per-event failures."""
	_configure_logging()
	args = parse_args(argv)
	processed = 0
	succeeded = 0
	skipped = 0
	failed = 0

	for event_dir_arg in args.event_dirs:
		processed += 1
		event_dir = Path(event_dir_arg).resolve()
		evt_path = _resolve_evt_path_for_log(event_dir)
		try:
			build_active_ch_for_event(
				event_dir,
				target_sampling_rate_HZ=args.target_fs_hz,
				scan_rate_blocks=args.scan_rate_blocks,
				skip_if_exists=args.skip_if_exists,
			)
		except Exception as exc:
			skipped += 1
			exception_message = str(exc)
			logger.exception(
				(
					'skipping event after exception: event_dir=%s evt_path=%s '
					'exception_class=%s exception_message=%s'
				),
				event_dir,
				evt_path if evt_path is not None else 'N/A',
				exc.__class__.__name__,
				exception_message,
			)
			continue
		succeeded += 1

	logger.info(
		'summary: processed=%d succeeded=%d skipped=%d failed=%d',
		processed,
		succeeded,
		skipped,
		failed,
	)


if __name__ == '__main__':
	main()
