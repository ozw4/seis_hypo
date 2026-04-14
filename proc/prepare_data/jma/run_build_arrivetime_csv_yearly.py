# ruff: noqa: D100, D103, E501
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from jma.arrivetime_reader import convert_epicenter_to_csv, convert_measure_to_csv

DEFAULT_INPUT_ROOT = Path('/workspace/data/arrivetime/JMA')
DEFAULT_GLOB_PATTERN = 'arrivetime_*.txt'
DEFAULT_EPICENTER_NAME = 'epicenter.csv'
DEFAULT_MEASUREMENT_NAME = 'measurement.csv'
DEFAULT_LOG_LEVEL = 'INFO'
LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description='Build yearly JMA arrivetime CSV files from daily txt files.'
	)
	parser.add_argument(
		'--input-root',
		type=Path,
		default=DEFAULT_INPUT_ROOT,
		help='root directory that contains yearly arrivetime directories',
	)
	parser.add_argument(
		'--years',
		nargs='+',
		help='target years under input-root',
	)
	parser.add_argument(
		'--glob-pattern',
		default=DEFAULT_GLOB_PATTERN,
		help='glob pattern for daily arrivetime txt files inside each year directory',
	)
	parser.add_argument(
		'--epicenter-name',
		default=DEFAULT_EPICENTER_NAME,
		help='output filename for epicenter CSV',
	)
	parser.add_argument(
		'--measurement-name',
		default=DEFAULT_MEASUREMENT_NAME,
		help='output filename for measurement CSV',
	)
	parser.add_argument(
		'--overwrite',
		action='store_true',
		help='overwrite existing yearly CSV outputs',
	)
	parser.add_argument(
		'--log-level',
		choices=('DEBUG', 'INFO', 'WARNING', 'ERROR'),
		default=DEFAULT_LOG_LEVEL,
		help='logging level',
	)
	return parser


def validate_input_root(input_root: Path) -> Path:
	input_root = input_root.resolve()
	if not input_root.exists():
		raise FileNotFoundError(f'input root does not exist: {input_root}')
	if not input_root.is_dir():
		raise NotADirectoryError(f'input root is not a directory: {input_root}')
	return input_root


def discover_year_directories(input_root: Path) -> list[Path]:
	return sorted(
		path
		for path in input_root.iterdir()
		if path.is_dir() and path.name.isdigit() and len(path.name) == 4
	)


def resolve_year_directories(input_root: Path, years: list[str] | None) -> list[Path]:
	if years is None:
		return discover_year_directories(input_root)

	year_directories: list[Path] = []
	for year in years:
		if not year.isdigit() or len(year) != 4:
			raise ValueError(f'year must be a 4-digit string: {year}')
		year_directory = input_root / year
		if not year_directory.exists():
			raise FileNotFoundError(f'year directory does not exist: {year_directory}')
		if not year_directory.is_dir():
			raise NotADirectoryError(f'year path is not a directory: {year_directory}')
		year_directories.append(year_directory)
	return year_directories


def collect_input_files(year_directory: Path, glob_pattern: str) -> list[Path]:
	return sorted(
		path for path in year_directory.glob(glob_pattern) if path.is_file()
	)


def build_yearly_csv(year_directory: Path, args: argparse.Namespace) -> None:
	input_files = collect_input_files(year_directory, args.glob_pattern)
	if not input_files:
		LOGGER.warning('skip year=%s: no input files matched %s', year_directory.name, args.glob_pattern)
		return

	epicenter_path = year_directory / args.epicenter_name
	measurement_path = year_directory / args.measurement_name
	if (epicenter_path.exists() or measurement_path.exists()) and not args.overwrite:
		LOGGER.warning(
			'skip year=%s: output exists and overwrite is disabled',
			year_directory.name,
		)
		return

	LOGGER.info(
		'build year=%s files=%d epicenter=%s measurement=%s',
		year_directory.name,
		len(input_files),
		epicenter_path,
		measurement_path,
	)
	convert_epicenter_to_csv(input_files, epicenter_path)
	convert_measure_to_csv(input_files, measurement_path)


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()

	logging.basicConfig(
		level=getattr(logging, args.log_level),
		format='%(asctime)s %(levelname)s %(message)s',
	)

	input_root = validate_input_root(args.input_root)
	year_directories = resolve_year_directories(input_root, args.years)
	for year_directory in year_directories:
		build_yearly_csv(year_directory, args)


if __name__ == '__main__':
	main()
