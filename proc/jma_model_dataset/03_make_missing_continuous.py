from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from jma.stationcode_mappingdb import load_mapping_db
from jma.stationcode_presence import load_presence_db
from jma_model_dataset.step1_missing_targets import build_missing_targets_for_event


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
			'flow-scoped *_active.ch exists under flows/jma_model_dataset/active/'
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
	return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
	args = parse_args(argv)

	meas_df = pd.read_csv(args.meas_csv, low_memory=False)
	epi_df = pd.read_csv(args.epi_csv, low_memory=False)
	pdb = load_presence_db(args.pres_csv)
	mdb = load_mapping_db(args.mapping_report_csv, args.near0_suggest_csv)

	for event_dir in args.event_dirs:
		result = build_missing_targets_for_event(
			event_dir,
			meas_df=meas_df,
			epi_df=epi_df,
			pdb=pdb,
			mdb=mdb,
		)
		if result.n_missing > 0:
			print(
				f'[missing] {result.evt_path.name}: '
				f'n_missing={result.n_missing} -> {result.missing_path}'
			)
		else:
			print(f'[missing] {result.evt_path.name}: n_missing=0')
		print(f'[mapping] {result.evt_path.name} -> {result.mapping_log_path}')


if __name__ == '__main__':
	main()
