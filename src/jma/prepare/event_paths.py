from __future__ import annotations

from pathlib import Path
from typing import Literal


def _raise_evt_count(event_dir: Path, evt_files: list[Path]) -> None:
	raise ValueError(
		f'.evt must be exactly 1 in {event_dir} (found {len(evt_files)}): '
		+ ', '.join([p.name for p in evt_files])
	)


def resolve_single_evt(event_dir: Path, *, allow_none: bool = False) -> Path | None:
	evt_files = sorted(Path(event_dir).glob('*.evt'))
	if len(evt_files) != 1:
		if allow_none and len(evt_files) == 0:
			return None
		_raise_evt_count(Path(event_dir), evt_files)
	return evt_files[0]


def resolve_evt_and_txt(event_dir: Path) -> tuple[Path, Path]:
	evt_path = resolve_single_evt(event_dir, allow_none=False)
	return evt_path, resolve_txt_for_evt(evt_path)


def resolve_evt_and_ch(event_dir: Path) -> tuple[Path, Path]:
	evt_path = resolve_single_evt(event_dir, allow_none=False)
	ch_path = Path(event_dir) / f'{evt_path.stem}.ch'
	if not ch_path.is_file():
		raise FileNotFoundError(f'event .ch not found: {ch_path}')
	return evt_path, ch_path


def resolve_txt_for_evt(evt_path: Path) -> Path:
	txt_path = Path(evt_path).with_suffix('.txt')
	if not txt_path.is_file():
		raise FileNotFoundError(f'event txt not found: {txt_path}')
	return txt_path


def resolve_active_ch(
	event_dir: Path,
	*,
	stem: str | None = None,
	mode: Literal['stem', 'glob_single'] = 'stem',
) -> Path:
	event_dir = Path(event_dir)
	if mode == 'glob_single':
		active_files = sorted(event_dir.glob('*_active.ch'))
		if len(active_files) != 1:
			raise ValueError(
				f'*_active.ch must be exactly 1 in {event_dir} (found {len(active_files)}): '
				+ ', '.join([p.name for p in active_files])
			)
		return active_files[0]

	if stem is None:
		raise ValueError("stem is required when mode='stem'")
	active_path = event_dir / f'{stem}_active.ch'
	if not active_path.is_file():
		raise FileNotFoundError(f'event active .ch not found: {active_path}')
	return active_path


def resolve_missing_continuous(event_dir: Path, *, stem: str) -> Path | None:
	p = Path(event_dir) / f'{stem}_missing_continuous.txt'
	if not p.is_file():
		return None
	return p
