from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd

from common.config import LokiWaveformStackingPipelineConfig
from common.time_util import parse_cfg_time_utc

_EventContext = TypeVar('_EventContext', bound='EventContext')


@dataclass(frozen=True)
class EventContext:
	origin_utc: pd.Timestamp
	meta: dict[str, Any]
	meta_path: Path
	event_dir: Path | None = None


def _coerce_mag(v: Any) -> float | None:
	"""Coerce an input value into a floating-point magnitude.

	This helper converts common representations of numeric magnitudes to a `float`,
	treating missing/blank values as `None`.

	Rules:
	- `None` -> `None`
	- `int`/`float` -> `float(v)`
	- `str` -> stripped; empty string -> `None`; otherwise `float(stripped)`
	- Any other type -> attempts `float(v)`

	Parameters
	----------
	v : Any
		The value to coerce.

	Returns
	-------
	float | None
		The coerced floating-point value, or `None` if the input is `None` or a
		blank string.

	Raises
	------
	ValueError
		If a non-empty string cannot be parsed as a float, or if `float(v)` fails
		for other types.
	TypeError
		If `float(v)` is not supported for the given type.

	"""
	if v is None:
		return None
	if isinstance(v, (int, float)):
		return float(v)
	if isinstance(v, str):
		s = v.strip()
		if s == '':
			return None
		return float(s)
	return float(v)


def extract_event_magnitude(meta: dict[str, Any]) -> float | None:
	"""Extract an event magnitude value from an event metadata mapping.

	The function looks for common magnitude keys in both the top-level metadata and
	the nested ``meta["extra"]`` mapping, in the following priority order:
	``"mag1"``, ``"magnitude"``, ``"mag"``. For each key, it checks the top-level
	``meta`` first, then ``meta["extra"]``. Values are normalized via
	``_coerce_mag``; the first successfully coerced magnitude is returned.

	Args:
		meta: Event metadata dictionary. May contain an ``"extra"`` entry; if
			present and not a ``dict``, it is ignored.

	Returns:
		A ``float`` magnitude if found and coercible; otherwise ``None``.

	"""
	extra = meta.get('extra', {})
	if not isinstance(extra, dict):
		extra = {}
	for key in ('mag1', 'magnitude', 'mag'):
		if key in meta:
			m = _coerce_mag(meta[key])
			if m is not None:
				return m
		if key in extra:
			m = _coerce_mag(extra[key])
			if m is not None:
				return m
	return None


def filter_event_dirs(
	cfg: LokiWaveformStackingPipelineConfig,
	*,
	build_candidates: Callable[[Path], list[Path]],
	read_event: Callable[[Path], _EventContext | None],
	get_event_time: Callable[[_EventContext], pd.Timestamp],
	extra_filter: Callable[[_EventContext], bool] | None,
	empty_error: str,
	log_prefix: str,
) -> list[Path]:
	"""Filter and return event directories under ``cfg.base_input_dir`` based on origin time
	constraints, an optional extra predicate, and an optional maximum count.

	The caller provides strategy functions to (1) enumerate candidate event directories,
	(2) read/parse each event into a context object, and (3) extract an origin timestamp
	from that context.

	Args:
		cfg: Pipeline configuration. Uses:
			- ``base_input_dir``: Root directory containing event directories/files.
			- ``origin_time_start`` / ``origin_time_end``: Optional UTC time bounds
			(parsed via ``parse_cfg_time_utc``).
			- ``event_glob``: Used only for formatting the empty-results error.
			- ``max_events``: If set and > 0, truncates the kept list to this length.
		build_candidates: Callable that takes ``base`` and returns a list of candidate
			``Path`` objects to consider.
		read_event: Callable that takes a candidate ``Path`` and returns an event context
			instance, or ``None`` to skip the candidate.
		get_event_time: Callable that extracts the event origin time as a ``pd.Timestamp``
			(UTC) from the event context.
		extra_filter: Optional predicate applied to the event context. If provided and
			returns ``False``, the event is dropped.
		empty_error: Format string used when no events are kept. It is formatted with
			``base`` and ``glob`` keys.
		log_prefix: Prefix for the summary line printed to stdout.

	Returns:
		A list of ``Path`` objects for event directories that pass all filters, optionally
		truncated to ``cfg.max_events``.

	Raises:
		FileNotFoundError: If ``cfg.base_input_dir`` does not exist or is not a directory.
		ValueError: If ``origin_time_end`` is earlier than ``origin_time_start``, or if
			no event directories remain after filtering (using ``empty_error``).

	Side Effects:
		Prints a summary line:
		``"{log_prefix}: total=<candidates> kept=<kept> dropped=<dropped>"``.

	"""
	base = Path(cfg.base_input_dir)
	if not base.is_dir():
		raise FileNotFoundError(f'base_input_dir not found: {base}')

	t_start = parse_cfg_time_utc(cfg.origin_time_start)
	t_end = parse_cfg_time_utc(cfg.origin_time_end)
	if t_start is not None and t_end is not None and t_end < t_start:
		raise ValueError(
			f'origin_time_end must be >= origin_time_start: {t_end} < {t_start}'
		)

	candidates = build_candidates(base)

	dirs: list[Path] = []
	dropped = 0

	for p in candidates:
		ctx = read_event(p)
		if ctx is None:
			continue

		origin_utc = get_event_time(ctx)

		if t_start is not None and origin_utc < t_start:
			dropped += 1
			continue
		if t_end is not None and origin_utc > t_end:
			dropped += 1
			continue

		if extra_filter is not None and not extra_filter(ctx):
			dropped += 1
			continue

		dirs.append(p)

	if cfg.max_events is not None and cfg.max_events > 0:
		dirs = dirs[: int(cfg.max_events)]

	if not dirs:
		raise ValueError(empty_error.format(base=base, glob=cfg.event_glob))

	print(f'{log_prefix}: total={len(candidates)} kept={len(dirs)} dropped={dropped}')
	return dirs
