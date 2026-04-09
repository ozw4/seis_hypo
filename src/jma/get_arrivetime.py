#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from collections.abc import Iterator
from datetime import date, datetime, timedelta
from netrc import netrc
from pathlib import Path

from HinetPy import Client

DEFAULT_START_DATE = date(2002, 6, 3)
DEFAULT_END_DATE = date(2025, 11, 15)
DEFAULT_SPAN_DAYS = 1
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_NETRC_MACHINE = 'www.hinet.bosai.go.jp'

LOGGER = logging.getLogger(__name__)


def parse_date(value: str) -> date:
	return datetime.strptime(value, '%Y-%m-%d').date()


def iter_request_dates(start: date, end: date, span_days: int) -> Iterator[date]:
	if span_days <= 0:
		raise ValueError('span_days must be positive')
	if start >= end:
		raise ValueError('start date must be earlier than end date')

	step = timedelta(days=span_days)
	current = start
	while current < end:
		yield current
		current += step


def build_output_path(output_dir: Path, start: date, span_days: int) -> Path:
	return output_dir / f'{start.year}' / f'arrivetime_{start:%Y%m%d}_{span_days}.txt'


def load_credentials_from_netrc(machine: str) -> tuple[str, str]:
	auth = netrc().authenticators(machine)
	if auth is None:
		raise ValueError(f'no .netrc entry found for machine={machine!r}')

	login, _, password = auth
	if not login:
		raise ValueError(f'.netrc entry for machine={machine!r} has no login')
	if not password:
		raise ValueError(f'.netrc entry for machine={machine!r} has no password')
	if len(password) > 12:
		raise ValueError(
			'Hi-net password is longer than 12 characters; '
			'set the exact usable password in .netrc'
		)

	return login, password


def download_arrivaltime_range(
	*,
	machine: str,
	start: date,
	end: date,
	span_days: int,
	output_dir: Path,
	line_ending: str,
	overwrite: bool,
) -> None:
	username, password = load_credentials_from_netrc(machine)

	output_dir = output_dir.resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	client = Client(username, password)

	for request_start in iter_request_dates(start, end, span_days):
		output_path = build_output_path(output_dir, request_start, span_days)
		output_path.parent.mkdir(parents=True, exist_ok=True)

		if output_path.exists() and not overwrite:
			LOGGER.info('skip existing: %s', output_path)
			continue

		LOGGER.info(
			'download start=%s span_days=%d output=%s',
			request_start.isoformat(),
			span_days,
			output_path,
		)
		client.get_arrivaltime(
			request_start,
			span_days,
			filename=str(output_path),
			os=line_ending,
		)


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description='Download JMA arrival time catalogs from Hi-net using credentials from .netrc.'
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
		default='UNIX',
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
		default='INFO',
		help='logging level',
	)
	return parser


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()

	logging.basicConfig(
		level=getattr(logging, args.log_level),
		format='%(asctime)s %(levelname)s %(message)s',
	)

	download_arrivaltime_range(
		machine=args.netrc_machine,
		start=args.start_date,
		end=args.end_date,
		span_days=args.span_days,
		output_dir=args.output_dir,
		line_ending=args.os,
		overwrite=args.overwrite,
	)


if __name__ == '__main__':
	main()
