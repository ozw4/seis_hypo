from __future__ import annotations

import argparse
from pathlib import Path

from jma.download import create_hinet_client, download_event_waveform_directories
from jma_model_dataset.paths import raw_root

DEFAULT_OUTDIR = Path('/workspace/data/waveform/jma/event').resolve()
DEFAULT_REGION = '00'
DEFAULT_MIN_MAG = 1.0
DEFAULT_MAX_MAG = 99.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			'Download JMA event waveforms for the model-dataset Step 0 workflow ' 
			'and place raw inputs under each event_dir/raw/.'
		)
	)
	parser.add_argument(
		'--events-start-jst',
		required=True,
		help='Start time of events in JST, formatted as YYYYMMDDHHMM',
	)
	parser.add_argument(
		'--events-end-jst',
		required=True,
		help='End time of events in JST, formatted as YYYYMMDDHHMM',
	)
	parser.add_argument(
		'--outdir',
		type=Path,
		default=DEFAULT_OUTDIR,
		help=f'Base directory that will contain per-event directories (default: {DEFAULT_OUTDIR})',
	)
	parser.add_argument(
		'--region',
		default=DEFAULT_REGION,
		help=f'JMA region code for event selection (default: {DEFAULT_REGION})',
	)
	parser.add_argument(
		'--min-mag',
		type=float,
		default=DEFAULT_MIN_MAG,
		help=f'Minimum magnitude filter (default: {DEFAULT_MIN_MAG})',
	)
	parser.add_argument(
		'--max-mag',
		type=float,
		default=DEFAULT_MAX_MAG,
		help=f'Maximum magnitude filter (default: {DEFAULT_MAX_MAG})',
	)
	return parser.parse_args(argv)


def _resolve_downloaded_triplet(event_dir: Path) -> tuple[Path, Path, Path]:
	files = sorted(path for path in event_dir.iterdir() if path.is_file())
	if not files:
		raise FileNotFoundError(f'no downloaded files found in event directory: {event_dir}')

	unexpected_files = [
		path for path in files if path.suffix not in {'.evt', '.ch', '.txt'}
	]
	if unexpected_files:
		names = ', '.join(path.name for path in unexpected_files)
		raise ValueError(f'unexpected files found in event directory {event_dir}: {names}')

	evt_files = [path for path in files if path.suffix == '.evt']
	ch_files = [path for path in files if path.suffix == '.ch']
	txt_files = [path for path in files if path.suffix == '.txt']
	if len(evt_files) != 1 or len(ch_files) != 1 or len(txt_files) != 1:
		raise ValueError(
			'event directory must contain exactly one .evt, one .ch, and one .txt ' 
			f'file: {event_dir}'
		)

	evt_path = evt_files[0]
	ch_path = ch_files[0]
	txt_path = txt_files[0]
	stems = {evt_path.stem, ch_path.stem, txt_path.stem}
	if len(stems) != 1:
		raise ValueError(
			'downloaded event files must share the same stem: ' 
			f'{evt_path.name}, {ch_path.name}, {txt_path.name}'
		)

	unexpected_dirs = [path for path in event_dir.iterdir() if path.is_dir()]
	if unexpected_dirs:
		names = ', '.join(path.name for path in unexpected_dirs)
		raise ValueError(f'unexpected directories found in event directory {event_dir}: {names}')

	return evt_path, ch_path, txt_path


def _move_triplet_to_raw(event_dir: Path) -> tuple[Path, Path, Path]:
	evt_path, ch_path, txt_path = _resolve_downloaded_triplet(event_dir)
	raw_dir = raw_root(event_dir)
	raw_dir.mkdir(parents=True, exist_ok=True)

	moved_paths: list[Path] = []
	for src_path in [evt_path, ch_path, txt_path]:
		dst_path = raw_dir / src_path.name
		if dst_path.exists():
			raise FileExistsError(f'raw file already exists: {dst_path}')
		src_path.rename(dst_path)
		moved_paths.append(dst_path)

	unexpected_paths = [path for path in event_dir.iterdir() if path.name != 'raw']
	if unexpected_paths:
		names = ', '.join(path.name for path in unexpected_paths)
		raise ValueError(f'unexpected artifacts remain after raw move in {event_dir}: {names}')

	return moved_paths[0], moved_paths[1], moved_paths[2]


def main(argv: list[str] | None = None) -> None:
	args = parse_args(argv)
	args.outdir.mkdir(parents=True, exist_ok=True)

	client = create_hinet_client()
	event_dirs = download_event_waveform_directories(
		client,
		args.events_start_jst,
		args.events_end_jst,
		args.outdir,
		region=args.region,
		minmagnitude=args.min_mag,
		maxmagnitude=args.max_mag,
	)

	for event_dir in event_dirs:
		evt_path, ch_path, txt_path = _move_triplet_to_raw(event_dir)
		print(f'[raw] {event_dir}')
		print(f'  evt: {evt_path}')
		print(f'  ch : {ch_path}')
		print(f'  txt: {txt_path}')


if __name__ == '__main__':
	main()
