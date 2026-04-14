# ruff: noqa: INP001
"""Build flow-scoped missing continuous targets for JMA model dataset events."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from jma.stationcode_mappingdb import load_mapping_db
from jma.stationcode_presence import load_presence_db
from jma_model_dataset.step1_missing_targets import build_missing_targets_for_event

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	"""Parse command-line arguments for missing continuous target generation."""
	parser = argparse.ArgumentParser(
		description=(
			'Create flow-scoped missing_continuous targets for the '
			'JMA model dataset Step1 workflow.'
		)
	)
	parser.add_argument(
		'event_dirs',
		nargs='+',
		type=Path,
		help=(
			'Event directory path(s) whose raw inputs are under raw/ and whose '
			'flow-scoped *_active.ch exists under '
			'flows/jma_model_dataset/active/'
		),
	)
	parser.add_argument(
		'--meas-csv',
		required=True,
		type=Path,
		help='Path to measurements.csv',
	)
	parser.add_argument(
		'--epi-csv',
		required=True,
		type=Path,
		help='Path to epicenters.csv',
	)
	parser.add_argument(
		'--pres-csv',
		required=True,
		type=Path,
		help='Path to monthly_presence.csv',
	)
	parser.add_argument(
		'--mapping-report-csv',
		required=True,
		type=Path,
		help='Path to mapping_report.csv',
	)
	parser.add_argument(
		'--near0-suggest-csv',
		required=True,
		type=Path,
		help='Path to near0_suggestions.csv',
	)
	parser.add_argument(
		'--skip-if-exists',
		action='store_true',
		help=(
			'Skip an event directory when both flow-scoped '
			'*_missing_continuous.txt and *_mapping_log.csv already exist'
		),
	)
	return parser.parse_args(argv)


def _configure_logging() -> None:
	if logging.getLogger().handlers:
		return
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s %(levelname)s %(name)s: %(message)s',
	)


def main(argv: list[str] | None = None) -> None:
	"""Process each event directory and continue past per-event failures."""
	_configure_logging()
	args = parse_args(argv)

	meas_df = pd.read_csv(args.meas_csv, low_memory=False)
	epi_df = pd.read_csv(args.epi_csv, low_memory=False)
	pdb = load_presence_db(args.pres_csv)
	mdb = load_mapping_db(args.mapping_report_csv, args.near0_suggest_csv)
	processed = 0
	succeeded = 0
	skipped = 0
	failed = 0

	for event_dir_arg in args.event_dirs:
		processed += 1
		event_dir = Path(event_dir_arg).resolve()
		event_name = event_dir.name
		try:
			result = build_missing_targets_for_event(
				event_dir,
				meas_df=meas_df,
				epi_df=epi_df,
				pdb=pdb,
				mdb=mdb,
				skip_if_exists=args.skip_if_exists,
			)
		except Exception as exc:
			skipped += 1
			exception_message = str(exc)
			logger.exception(
				(
					'skipping event after exception: event_dir=%s event_name=%s '
					'exception_class=%s exception_message=%s'
				),
				event_dir,
				event_name,
				exc.__class__.__name__,
				exception_message,
			)
			continue

		succeeded += 1
		if result.skipped:
			print(
				f'[skip] {result.evt_path.name}: '
				f'existing outputs -> {result.missing_path}, {result.mapping_log_path}'
			)
			continue
		if result.n_missing > 0:
			print(
				f'[missing] {result.evt_path.name}: '
				f'n_missing={result.n_missing} -> {result.missing_path}'
			)
		else:
			print(f'[missing] {result.evt_path.name}: n_missing=0')
		print(f'[mapping] {result.evt_path.name} -> {result.mapping_log_path}')

	logger.info(
		'summary: processed=%d succeeded=%d skipped=%d failed=%d',
		processed,
		succeeded,
		skipped,
		failed,
	)


if __name__ == '__main__':
	main()
