from __future__ import annotations

import argparse
from pathlib import Path

from jma.download import create_hinet_client
from jma_model_dataset.event_skip_log import (
	MISSING_INPUT_EXCEPTIONS,
	append_event_skip_log,
	is_expected_missing_input_error,
)
from jma_model_dataset.paths import missing_dir
from jma_model_dataset.step2_missing_continuous import (
	download_missing_continuous_for_event,
)

DEFAULT_RUN_TAG = 'v1'
DEFAULT_THREADS = 8
STEP_NAME = '04_get_missing_continuous_waveform'


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
	parser.add_argument(
		'--skip-missing-inputs',
		action=argparse.BooleanOptionalAction,
		default=False,
		help=(
			'Skip the current event and append a flow-scoped skip log when '
			'required per-event input files are missing (default: false)'
		),
	)
	return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
	args = parse_args(argv)
	client = create_hinet_client()
	failed_events: list[tuple[Path, str]] = []
	skipped_events: list[tuple[Path, str, Path]] = []
	ok_count = 0

	for event_dir in args.event_dirs:
		event_dir2 = Path(event_dir).resolve()
		failure_marker = (
			missing_dir(event_dir2) / '04_get_missing_continuous.failed.txt'
		)
		try:
			result = download_missing_continuous_for_event(
				event_dir2,
				client,
				run_tag=args.run_tag,
				threads=args.threads,
				cleanup=args.cleanup,
				skip_if_exists=args.skip_if_exists,
				skip_if_done=args.skip_if_done,
			)
		except MISSING_INPUT_EXCEPTIONS as error:
			if args.skip_missing_inputs and is_expected_missing_input_error(
				error, event_dir=event_dir2
			):
				log_path = append_event_skip_log(
					event_dir2,
					step_name=STEP_NAME,
					reason='missing_input',
					error=error,
				)
				message = str(error)
				print(
					f'[skip] step=04 event_dir={event_dir2} '
					f'error={error.__class__.__name__}: {message} log={log_path}'
				)
				skipped_events.append((event_dir2, message, log_path))
				continue
			if not isinstance(error, ValueError):
				raise
			failure_marker.parent.mkdir(parents=True, exist_ok=True)
			message = str(error)
			failure_marker.write_text(
				f'event_dir={event_dir2}\nerror={message}\n',
				encoding='utf-8',
			)
			print(f'[error] event_dir={event_dir2} error={message}')
			failed_events.append((event_dir2, message))
			continue

		if failure_marker.exists():
			failure_marker.unlink()

		ok_count += 1
		print(
			f'[step2] {result.evt_path.name}: '
			f'networks={result.n_networks_total} '
			f'downloaded={result.n_downloaded} '
			f'exists={result.n_exists} '
			f'skipped_done={result.n_skipped_done} '
			f'-> {result.outdir}'
		)
		print(f'[log] {result.evt_path.name} -> {result.log_path}')

	print(
		f'[summary] ok={ok_count} '
		f'skipped_missing_input={len(skipped_events)} '
		f'failed={len(failed_events)}'
	)
	if skipped_events:
		for event_dir, message, log_path in skipped_events:
			print(f'[skipped] event_dir={event_dir} error={message} log={log_path}')
	if failed_events:
		for event_dir, message in failed_events:
			print(f'[failed] event_dir={event_dir} error={message}')


if __name__ == '__main__':
	main()
