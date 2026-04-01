from __future__ import annotations

from pathlib import Path

from jma.prepare.active_channel import make_active_ch_for_evt
from jma.prepare.event_paths import resolve_evt_and_ch
from jma_model_dataset.paths import active_ch_path

__all__ = ['build_active_ch_for_event']


def build_active_ch_for_event(
	event_dir: Path,
	*,
	target_sampling_rate_HZ: int = 100,
	scan_rate_blocks: int = 1000,
	skip_if_exists: bool = False,
) -> Path:
	event_dir = Path(event_dir).resolve()
	if not event_dir.is_dir():
		raise NotADirectoryError(f'event directory not found: {event_dir}')

	evt_path, ch_path = resolve_evt_and_ch(event_dir)
	out_path = active_ch_path(event_dir, evt_path.stem)
	if skip_if_exists and out_path.is_file():
		return out_path

	out_path.parent.mkdir(parents=True, exist_ok=True)
	return make_active_ch_for_evt(
		evt_path,
		ch_path,
		out_ch_path=out_path,
		target_sampling_rate_HZ=int(target_sampling_rate_HZ),
		scan_rate_blocks=int(scan_rate_blocks),
	)
