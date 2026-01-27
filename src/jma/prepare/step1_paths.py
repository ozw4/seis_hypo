from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EventStep1Files:
	stem: str
	evt_path: Path
	ch_path: Path
	txt_path: Path
	active_ch_path: Path


def build_step1_paths(event_dir: Path, *, stem: str | None = None) -> EventStep1Files:
	"""Step1で使うファイルパス一式を組み立てる.

	前提:
	- stem を指定しない場合、event_dir.name を stem とみなす（既存の運用ルール）。
	- event_dir 配下に {stem}.evt / {stem}.ch / {stem}.txt / {stem}_active.ch がある想定。
	"""
	event_dir = Path(event_dir)
	stem2 = str(stem).strip() if stem is not None else event_dir.name
	if stem2 == '':
		raise ValueError('stem must be non-empty')
	return EventStep1Files(
		stem=stem2,
		evt_path=event_dir / f'{stem2}.evt',
		ch_path=event_dir / f'{stem2}.ch',
		txt_path=event_dir / f'{stem2}.txt',
		active_ch_path=event_dir / f'{stem2}_active.ch',
	)
