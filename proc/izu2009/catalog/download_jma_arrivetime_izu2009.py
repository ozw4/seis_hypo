# %%
"""Download JMA arrival-time catalogs for Izu 2009 from a YAML config."""

# file: proc/izu2009/catalog/download_jma_arrivetime_izu2009.py
#
# Purpose:
# - Keep the JMA arrival-time download parameters in a reviewable YAML file.
# - Reuse the existing jma.download and jma.get_arrivetime utilities.
# - Allow a dry run that lists the planned Hi-net requests without logging in.

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import yaml

if TYPE_CHECKING:
	from collections.abc import Iterator

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _REPO_ROOT / 'src'

for _path in (_REPO_ROOT, _SRC_DIR):
	_path_str = str(_path)
	if _path_str not in sys.path:
		sys.path.insert(0, _path_str)

DEFAULT_CONFIG = _REPO_ROOT / 'proc/izu2009/catalog/config/jma_arrivetime_download.yaml'
VALID_LINE_ENDINGS = {'UNIX', 'DOS'}
VALID_LOG_LEVELS = {'DEBUG', 'INFO', 'WARNING', 'ERROR'}


class _DownloadConfig(TypedDict):
	"""Normalized download configuration."""

	start_date: date
	end_date: date
	span_days: int
	output_dir: Path
	netrc_machine: str
	line_ending: str
	overwrite: bool
	log_level: str


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
	"""Build the output file path used by src.jma.get_arrivetime."""
	return output_dir / f'{start.year}' / f'arrivetime_{start:%Y%m%d}_{span_days}.txt'


def _parse_date(value: object, *, field: str) -> date:
	if isinstance(value, date):
		return value
	if isinstance(value, str):
		return date.fromisoformat(value)
	raise TypeError(f'{field} must be YYYY-MM-DD string, got {type(value).__name__}')


def _as_bool(value: object, *, field: str) -> bool:
	if isinstance(value, bool):
		return value
	raise TypeError(f'{field} must be boolean, got {type(value).__name__}')


def _as_int(value: object, *, field: str) -> int:
	if isinstance(value, bool) or not isinstance(value, int):
		raise TypeError(f'{field} must be integer, got {type(value).__name__}')
	return value


def _as_str(value: object, *, field: str) -> str:
	if isinstance(value, str):
		return value
	raise TypeError(f'{field} must be string, got {type(value).__name__}')


def _load_download_config(config_path: Path) -> _DownloadConfig:
	if not config_path.is_file():
		raise FileNotFoundError(f'config YAML not found: {config_path}')

	with config_path.open('r', encoding='utf-8') as f:
		obj: object = yaml.safe_load(f)

	if not isinstance(obj, dict):
		raise TypeError(f'config YAML root must be a mapping: {config_path}')

	download_obj = obj.get('download', obj)
	if not isinstance(download_obj, dict):
		raise TypeError('config YAML must contain a mapping at key="download"')
	download = {str(key): value for key, value in download_obj.items()}

	required = [
		'start_date',
		'end_date',
		'span_days',
		'output_dir',
		'netrc_machine',
		'line_ending',
		'overwrite',
		'log_level',
	]
	missing = [key for key in required if key not in download]
	if missing:
		raise ValueError(f'missing required download config key(s): {missing}')

	start = _parse_date(download['start_date'], field='download.start_date')
	end = _parse_date(download['end_date'], field='download.end_date')
	span_days = _as_int(download['span_days'], field='download.span_days')
	output_dir = Path(_as_str(download['output_dir'], field='download.output_dir'))
	netrc_machine = _as_str(download['netrc_machine'], field='download.netrc_machine')
	line_ending = _as_str(download['line_ending'], field='download.line_ending').upper()
	overwrite = _as_bool(download['overwrite'], field='download.overwrite')
	log_level = _as_str(download['log_level'], field='download.log_level').upper()

	if start >= end:
		raise ValueError(f'start_date must be earlier than end_date: {start} >= {end}')
	if span_days <= 0:
		raise ValueError(f'span_days must be positive: {span_days}')
	if line_ending not in VALID_LINE_ENDINGS:
		raise ValueError(f'line_ending must be one of {sorted(VALID_LINE_ENDINGS)}')
	if log_level not in VALID_LOG_LEVELS:
		raise ValueError(f'log_level must be one of {sorted(VALID_LOG_LEVELS)}')

	return {
		'start_date': start,
		'end_date': end,
		'span_days': span_days,
		'output_dir': output_dir,
		'netrc_machine': netrc_machine,
		'line_ending': line_ending,
		'overwrite': overwrite,
		'log_level': log_level,
	}


def build_parser() -> argparse.ArgumentParser:
	"""Build the command-line parser."""
	parser = argparse.ArgumentParser(
		description='Download JMA arrival-time catalogs for Izu 2009 using YAML.'
	)
	parser.add_argument(
		'--config',
		type=Path,
		default=DEFAULT_CONFIG,
		help='YAML config path',
	)
	parser.add_argument(
		'--dry-run',
		action='store_true',
		help=(
			'print planned requests and output files without logging in or downloading'
		),
	)
	return parser


def _print_plan(config: _DownloadConfig) -> None:
	print('JMA arrival-time download plan')
	print('  start_date    :', config['start_date'])
	print('  end_date      :', config['end_date'], '(exclusive)')
	print('  span_days     :', config['span_days'])
	print('  output_dir    :', config['output_dir'])
	print('  netrc_machine :', config['netrc_machine'])
	print('  line_ending   :', config['line_ending'])
	print('  overwrite     :', config['overwrite'])
	print('  requests:')
	for request_start in iter_request_dates(
		config['start_date'],
		config['end_date'],
		config['span_days'],
	):
		output_path = build_output_path(
			config['output_dir'],
			request_start,
			config['span_days'],
		)
		print(f'    - start={request_start} -> {output_path}')


def main() -> None:
	"""Run the downloader CLI."""
	parser = build_parser()
	args = parser.parse_args()

	config = _load_download_config(args.config)
	logging.basicConfig(
		level=getattr(logging, config['log_level']),
		format='%(asctime)s %(levelname)s %(message)s',
	)

	_print_plan(config)
	if args.dry_run:
		return

	# Import Hi-net dependencies only when actually downloading. This keeps
	# --dry-run usable even in environments where HinetPy is not installed yet.
	from jma.download import create_hinet_client  # noqa: PLC0415
	from jma.get_arrivetime import download_arrivaltime_range  # noqa: PLC0415

	client = create_hinet_client(machine=config['netrc_machine'])
	download_arrivaltime_range(
		client,
		start=config['start_date'],
		end=config['end_date'],
		span_days=config['span_days'],
		output_dir=config['output_dir'],
		line_ending=config['line_ending'],
		overwrite=config['overwrite'],
	)


if __name__ == '__main__':
	main()

# Examples:
# python proc/izu2009/catalog/download_jma_arrivetime_izu2009.py --dry-run
# python proc/izu2009/catalog/download_jma_arrivetime_izu2009.py
