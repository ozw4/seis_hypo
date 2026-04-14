"""Download JMA arrival-time catalogs from Hi-net."""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

from jma.download import create_hinet_client
from jma.get_arrivetime import download_arrivaltime_range

DEFAULT_START_DATE = date(2002, 6, 3)
DEFAULT_END_DATE = date(2025, 11, 15)
DEFAULT_SPAN_DAYS = 1
DEFAULT_OUTPUT_DIR = Path('/workspace/data/arrivetime/JMA')
DEFAULT_NETRC_MACHINE = 'hinet'
DEFAULT_LINE_ENDING = 'UNIX'
DEFAULT_LOG_LEVEL = 'INFO'


def parse_date(value: str) -> date:
	"""Parse an ISO calendar date."""
	return date.fromisoformat(value)


def build_parser() -> argparse.ArgumentParser:
	"""Build the CLI parser."""
	parser = argparse.ArgumentParser(
		description=(
			'Download JMA arrival time catalogs from Hi-net using '
			'credentials from .netrc.'
		)
	)
	parser.add_argument(
		'--start-date',
		type=parse_date,
		default=DEFAULT_START_DATE,
		help='inclusive start date in YYYY-MM-DD format',
	)
	parser.add_argument(
		'--end-date',
		type=parse_date,
		default=DEFAULT_END_DATE,
		help='exclusive end date in YYYY-MM-DD format',
	)
	parser.add_argument(
		'--span-days',
		type=int,
		default=DEFAULT_SPAN_DAYS,
		help='request length in days for each download',
	)
	parser.add_argument(
		'--output-dir',
		type=Path,
		default=DEFAULT_OUTPUT_DIR,
		help='base output directory',
	)
	parser.add_argument(
		'--netrc-machine',
		default=DEFAULT_NETRC_MACHINE,
		help='machine name in ~/.netrc',
	)
	parser.add_argument(
		'--os',
		choices=('UNIX', 'DOS'),
		default=DEFAULT_LINE_ENDING,
		help='line ending format passed to HinetPy',
	)
	parser.add_argument(
		'--overwrite',
		action='store_true',
		help='overwrite existing files',
	)
	parser.add_argument(
		'--log-level',
		choices=('DEBUG', 'INFO', 'WARNING', 'ERROR'),
		default=DEFAULT_LOG_LEVEL,
		help='logging level',
	)
	return parser


def main() -> None:
	"""Run the arrival-time download CLI."""
	parser = build_parser()
	args = parser.parse_args()

	logging.basicConfig(
		level=getattr(logging, args.log_level),
		format='%(asctime)s %(levelname)s %(message)s',
	)

	client = create_hinet_client(machine=args.netrc_machine)
	download_arrivaltime_range(
		client,
		start=args.start_date,
		end=args.end_date,
		span_days=args.span_days,
		output_dir=args.output_dir,
		line_ending=args.os,
		overwrite=args.overwrite,
	)


if __name__ == '__main__':
	main()
