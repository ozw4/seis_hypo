from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

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


def extract_event_magnitude(meta: dict[str, Any]) -> float | None:
	extra = meta.get('extra', {})
	if not isinstance(extra, dict):
		extra = {}

	for key in ('mag1', 'magnitude', 'mag'):
		if key in meta:
			return float(meta[key])
		if key in extra:
			return float(extra[key])
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
