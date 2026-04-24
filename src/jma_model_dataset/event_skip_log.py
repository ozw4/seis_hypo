"""Event-level skip logging for JMA model dataset flow scripts."""

from __future__ import annotations

import datetime as dt
from contextlib import closing
from pathlib import Path

from common.csv_util import open_dict_writer
from jma_model_dataset.paths import logs_dir, raw_root

__all__ = [
	'MISSING_INPUT_EXCEPTIONS',
	'append_event_skip_log',
	'is_expected_missing_input_error',
]

MISSING_INPUT_EXCEPTIONS = (FileNotFoundError, NotADirectoryError, ValueError)
_SKIP_LOG_FILENAME = 'skipped_events.csv'
_SKIP_LOG_FIELDS = [
	'time_utc',
	'step_name',
	'reason',
	'event_dir',
	'event_name',
	'evt_file',
	'exception_class',
	'message',
]
_EVT_COUNT_MESSAGE = '.evt must be exactly 1'
_EVT_COUNT_ZERO_MESSAGE = '(found 0)'
_NO_MISSING_STATIONS_MESSAGE = 'no missing station pairs found in'


def _error_mentions_event_dir(error: BaseException, event_dir: Path) -> bool:
	event_dir_text = str(Path(event_dir).resolve())
	parts = [str(error)]
	for name in ('filename', 'filename2'):
		value = getattr(error, name, None)
		if value is not None:
			parts.append(str(value))
	return any(event_dir_text in part for part in parts)


def is_expected_missing_input_error(
	error: BaseException,
	*,
	event_dir: Path | None = None,
) -> bool:
	"""Return true only for expected event-local input-missing errors."""
	if isinstance(error, (FileNotFoundError, NotADirectoryError)):
		return event_dir is None or _error_mentions_event_dir(error, event_dir)
	if isinstance(error, ValueError):
		message = str(error)
		if event_dir is not None and not _error_mentions_event_dir(error, event_dir):
			return False
		return (
			_EVT_COUNT_MESSAGE in message and _EVT_COUNT_ZERO_MESSAGE in message
		) or _NO_MISSING_STATIONS_MESSAGE in message
	return False


def _utc_now_text() -> str:
	return (
		dt.datetime.now(dt.timezone.utc)
		.replace(microsecond=0)
		.isoformat()
		.replace('+00:00', 'Z')
	)


def _resolve_evt_file_for_log(event_dir: Path) -> str:
	raw_dir = raw_root(event_dir)
	if not raw_dir.is_dir():
		return ''
	evt_paths = sorted(raw_dir.glob('*.evt'))
	if len(evt_paths) != 1:
		return ''
	return evt_paths[0].name


def append_event_skip_log(
	event_dir: Path,
	*,
	step_name: str,
	reason: str,
	error: BaseException,
) -> Path:
	"""Append one event-level skip record under flows/jma_model_dataset/logs/."""
	event_dir2 = Path(event_dir).resolve()
	step_name2 = str(step_name).strip()
	reason2 = str(reason).strip()
	if step_name2 == '':
		raise ValueError('step_name must be non-empty')
	if reason2 == '':
		raise ValueError('reason must be non-empty')

	log_path = logs_dir(event_dir2) / _SKIP_LOG_FILENAME
	write_header = not log_path.is_file() or log_path.stat().st_size == 0
	log_f, writer = open_dict_writer(
		log_path,
		fieldnames=_SKIP_LOG_FIELDS,
		mode='a',
		write_header=write_header,
	)
	with closing(log_f):
		writer.writerow(
			{
				'time_utc': _utc_now_text(),
				'step_name': step_name2,
				'reason': reason2,
				'event_dir': str(event_dir2),
				'event_name': event_dir2.name,
				'evt_file': _resolve_evt_file_for_log(event_dir2),
				'exception_class': error.__class__.__name__,
				'message': str(error),
			}
		)
	return log_path
