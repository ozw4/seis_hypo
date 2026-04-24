from __future__ import annotations

import argparse
from pathlib import Path

from jma.download import create_hinet_client
from jma.stationcode_presence import load_presence_db
from jma_model_dataset.event_skip_log import (
	MISSING_INPUT_EXCEPTIONS,
	append_event_skip_log,
	is_expected_missing_input_error,
)
from jma_model_dataset.step3_fill_to_48 import (
	download_fill_to_48_for_event,
	load_fill_to_48_station_geo,
)

DEFAULT_RUN_TAG = 'v1'
DEFAULT_THREADS = 8
STEP_NAME = '05_fill_to_48_stations'


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			'Fill JMA model dataset events up to 48 stations using flow-scoped '
			'raw/ and flows/jma_model_dataset/{active,missing,continuous} inputs.'
		)
	)
	parser.add_argument(
		'event_dirs',
		nargs='+',
		type=Path,
		help=(
			'Event directory path(s) whose raw inputs are under raw/ and whose '
			'flow-scoped fill_to_48 outputs will be written under '
			'flows/jma_model_dataset/{continuous,logs,done}/'
		),
	)
	parser.add_argument(
		'--pres-csv',
		required=True,
		type=Path,
		help='Path to monthly_presence.csv used for event-month 0101 station filtering',
	)
	parser.add_argument(
		'--hinet-channel-table',
		required=True,
		type=Path,
		help='Path to the Hi-net channel table used for station coordinates',
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
		help='Skip downloading when the flow-scoped fill_to_48 .cnt and .ch already exist (default: true)',
	)
	parser.add_argument(
		'--skip-if-done',
		action=argparse.BooleanOptionalAction,
		default=True,
		help='Skip an event when its flow-scoped fill_to_48 done marker already matches run_tag (default: true)',
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
	pdb = load_presence_db(args.pres_csv)
	station_geo_0101 = load_fill_to_48_station_geo(args.hinet_channel_table)
	client = create_hinet_client()
	skipped_events: list[tuple[Path, str, Path]] = []
	ok_count = 0

	for event_dir in args.event_dirs:
		event_dir2 = Path(event_dir).resolve()
		try:
			result = download_fill_to_48_for_event(
				event_dir2,
				client,
				pdb=pdb,
				station_geo_0101=station_geo_0101,
				run_tag=args.run_tag,
				threads=args.threads,
				cleanup=args.cleanup,
				skip_if_exists=args.skip_if_exists,
				skip_if_done=args.skip_if_done,
			)
		except MISSING_INPUT_EXCEPTIONS as error:
			if not (
				args.skip_missing_inputs
				and is_expected_missing_input_error(error, event_dir=event_dir2)
			):
				raise
			log_path = append_event_skip_log(
				event_dir2,
				step_name=STEP_NAME,
				reason='missing_input',
				error=error,
			)
			message = str(error)
			print(
				f'[skip] step=05 event_dir={event_dir2} '
				f'error={error.__class__.__name__}: {message} log={log_path}'
			)
			skipped_events.append((event_dir2, message, log_path))
			continue

		ok_count += 1
		print(
			f'[step3] {result.evt_path.name}: '
			f'status={result.status} '
			f'before={result.n_before} '
			f'selected={result.n_selected} '
			f'after={result.n_after} '
			f'-> {result.outdir}'
		)
		print(f'[log] {result.evt_path.name} -> {result.log_path}')
		print(f'[done] {result.evt_path.name} -> {result.done_path}')

	print(f'[summary] ok={ok_count} skipped_missing_input={len(skipped_events)}')
	if skipped_events:
		for event_dir, message, log_path in skipped_events:
			print(f'[skipped] event_dir={event_dir} error={message} log={log_path}')


if __name__ == '__main__':
	main()
