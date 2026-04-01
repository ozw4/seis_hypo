from __future__ import annotations

import argparse
from pathlib import Path

from jma.download import create_hinet_client
from jma.stationcode_presence import load_presence_db
from jma_model_dataset.step3_fill_to_48 import (
	download_fill_to_48_for_event,
	load_fill_to_48_station_geo,
)

DEFAULT_RUN_TAG = 'v1'
DEFAULT_THREADS = 8


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
	return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
	args = parse_args(argv)
	pdb = load_presence_db(args.pres_csv)
	station_geo_0101 = load_fill_to_48_station_geo(args.hinet_channel_table)
	client = create_hinet_client()

	for event_dir in args.event_dirs:
		result = download_fill_to_48_for_event(
			event_dir,
			client,
			pdb=pdb,
			station_geo_0101=station_geo_0101,
			run_tag=args.run_tag,
			threads=args.threads,
			cleanup=args.cleanup,
			skip_if_exists=args.skip_if_exists,
			skip_if_done=args.skip_if_done,
		)
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


if __name__ == '__main__':
	main()
