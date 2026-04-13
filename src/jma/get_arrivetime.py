"""Reusable helpers for downloading JMA arrival-time catalogs."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path

from HinetPy import Client

LOGGER = logging.getLogger(__name__)


def iter_request_dates(start: date, end: date, span_days: int) -> Iterator[date]:
	"""Yield request start dates over a half-open date range."""
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
	"""Build the output file path for one arrival-time request."""
	return output_dir / f'{start.year}' / f'arrivetime_{start:%Y%m%d}_{span_days}.txt'


def download_arrivaltime_range(
	client: Client,
	*,
	start: date,
	end: date,
	span_days: int,
	output_dir: Path,
	line_ending: str,
	overwrite: bool,
) -> None:
	"""Download arrival-time catalogs over a half-open date range."""
	output_dir = output_dir.resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

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
