from __future__ import annotations

import argparse
from pathlib import Path

from jma.download import create_hinet_client
from jma_model_dataset.step2_missing_continuous import (
	download_missing_continuous_for_event,
)

DEFAULT_RUN_TAG = 'v1'
DEFAULT_THREADS = 8


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			'Download flow-scoped continuous waveform supplements for the '
			'JMA model dataset Step2 workflow.'
		)
	)
	parser.add_argument(
		'event_dirs',
		nargs='+',
		type=Path,
		help=(
			'Event directory path(s) whose raw inputs are under raw/ and whose '
			'flow-scoped *_missing_continuous.txt exists under '
			'flows/jma_model_dataset/missing/'
		),
	)
	parser.add_argument(
		'--run-tag',
		default=DEFAULT_RUN_TAG,
		help=f'Run tag recorded in flow-scoped done markers (default: {DEFAULT_RUN_TAG})',
	)
	parser.add_argument(
		'--threads',
		type=int,
		default=DEFAULT_THREADS,
		help=f'Number of HinetPy download threads (default: {DEFAULT_THREADS})',
	)
	parser.add_argument(
		'--cleanup',
		action=argparse.BooleanOptionalAction,
		default=True,
		help='Whether to remove temporary HinetPy download artifacts (default: true)',
	)
	parser.add_argument(
		'--skip-if-exists',
		action=argparse.BooleanOptionalAction,
		default=True,
		help='Skip a network when its flow-scoped .cnt and .ch already exist (default: true)',
	)
	parser.add_argument(
		'--skip-if-done',
		action=argparse.BooleanOptionalAction,
		default=True,
		help='Skip a network when its flow-scoped done marker already matches run_tag (default: true)',
	)
	return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
	args = parse_args(argv)
	client = create_hinet_client()

	for event_dir in args.event_dirs:
		result = download_missing_continuous_for_event(
			event_dir,
			client,
			run_tag=args.run_tag,
			threads=args.threads,
			cleanup=args.cleanup,
			skip_if_exists=args.skip_if_exists,
			skip_if_done=args.skip_if_done,
		)
		print(
			f'[step2] {result.evt_path.name}: '
			f'networks={result.n_networks_total} '
			f'downloaded={result.n_downloaded} '
			f'exists={result.n_exists} '
			f'skipped_done={result.n_skipped_done} '
			f'-> {result.outdir}'
		)
		print(f'[log] {result.evt_path.name} -> {result.log_path}')


if __name__ == '__main__':
	main()
