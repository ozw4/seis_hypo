from __future__ import annotations

import argparse
from pathlib import Path

from jma_model_dataset.paths import raw_root

DEFAULT_OUTDIR = Path('/workspace/data/waveform/jma/event').resolve()
DEFAULT_REGION = '00'
DEFAULT_MIN_MAG = 1.0
DEFAULT_MAX_MAG = 99.0
RAW_SUFFIXES = {'.evt', '.ch', '.txt'}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			'Download JMA event waveforms for the model-dataset Step 0 workflow '
			'and place raw inputs under each event_dir/raw/. Use --skip-download '
			'to only organize event directories that already exist under --outdir.'
		)
	)
	parser.add_argument(
		'--events-start-jst',
		help=(
			'Start time of events in JST, formatted as YYYYMMDDHHMM. '
			'Required unless --skip-download is set.'
		),
	)
	parser.add_argument(
		'--events-end-jst',
		help=(
			'End time of events in JST, formatted as YYYYMMDDHHMM. '
			'Required unless --skip-download is set.'
		),
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
	parser.add_argument(
		'--skip-download',
		action='store_true',
		help='Do not call Hi-net get_event_waveform; use existing D* event directories under --outdir.',
	)
	parser.add_argument(
		'--skip-incomplete',
		action='store_true',
		help='Skip event directories that do not contain a complete .evt/.ch/.txt triplet.',
	)

	args = parser.parse_args(argv)
	if not args.skip_download and (
		args.events_start_jst is None or args.events_end_jst is None
	):
		parser.error(
			'--events-start-jst and --events-end-jst are required '
			'unless --skip-download is set'
		)
	return args


def _names(paths: list[Path]) -> str:
	return ', '.join(path.name for path in paths) or '(none)'


def _resolve_triplet(directory: Path) -> tuple[Path, Path, Path]:
	if not directory.exists():
		raise FileNotFoundError(f'directory does not exist: {directory}')
	if not directory.is_dir():
		raise NotADirectoryError(f'not a directory: {directory}')

	files = sorted(path for path in directory.iterdir() if path.is_file())
	if not files:
		raise FileNotFoundError(f'no raw files found in directory: {directory}')

	unexpected_files = [path for path in files if path.suffix not in RAW_SUFFIXES]
	if unexpected_files:
		raise ValueError(
			f'unexpected files found in directory {directory}: {_names(unexpected_files)}'
		)

	evt_files = [path for path in files if path.suffix == '.evt']
	ch_files = [path for path in files if path.suffix == '.ch']
	txt_files = [path for path in files if path.suffix == '.txt']
	if len(evt_files) != 1 or len(ch_files) != 1 or len(txt_files) != 1:
		raise ValueError(
			'directory must contain exactly one .evt, one .ch, and one .txt file: '
			f'{directory}\n'
			f'  .evt ({len(evt_files)}): {_names(evt_files)}\n'
			f'  .ch  ({len(ch_files)}): {_names(ch_files)}\n'
			f'  .txt ({len(txt_files)}): {_names(txt_files)}'
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

	return evt_path, ch_path, txt_path


def _resolve_existing_raw_triplet(event_dir: Path) -> tuple[Path, Path, Path] | None:
	raw_dir = raw_root(event_dir)
	if not raw_dir.exists():
		return None
	if not raw_dir.is_dir():
		raise NotADirectoryError(f'raw path is not a directory: {raw_dir}')
	if not any(raw_dir.iterdir()):
		return None
	return _resolve_triplet(raw_dir)


def _resolve_downloaded_triplet(event_dir: Path) -> tuple[Path, Path, Path]:
	return _resolve_triplet(event_dir)


def _prepare_raw_triplet(event_dir: Path) -> tuple[str, Path, Path, Path]:
	raw_triplet = _resolve_existing_raw_triplet(event_dir)
	if raw_triplet is not None:
		evt_path, ch_path, txt_path = raw_triplet
		return 'raw-exists', evt_path, ch_path, txt_path

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

	return 'raw-moved', moved_paths[0], moved_paths[1], moved_paths[2]


def _existing_event_dirs(outdir: Path) -> list[Path]:
	if not outdir.exists():
		raise FileNotFoundError(f'event output directory does not exist: {outdir}')
	if not outdir.is_dir():
		raise NotADirectoryError(f'event output path is not a directory: {outdir}')

	event_dirs = sorted(
		path for path in outdir.iterdir() if path.is_dir() and path.name.startswith('D')
	)
	if not event_dirs:
		raise FileNotFoundError(f'no existing D* event directories found under: {outdir}')
	return event_dirs


def _download_event_dirs(args: argparse.Namespace) -> list[Path]:
	from jma.download import create_hinet_client, download_event_waveform_directories

	args.outdir.mkdir(parents=True, exist_ok=True)
	client = create_hinet_client()
	return download_event_waveform_directories(
		client,
		args.events_start_jst,
		args.events_end_jst,
		args.outdir,
		region=args.region,
		minmagnitude=args.min_mag,
		maxmagnitude=args.max_mag,
	)


def main(argv: list[str] | None = None) -> None:
	args = parse_args(argv)

	if args.skip_download:
		event_dirs = _existing_event_dirs(args.outdir)
		print(
			f'[skip-download] use existing event directories: '
			f'{len(event_dirs)} under {args.outdir}'
		)
	else:
		event_dirs = _download_event_dirs(args)

	prepared = 0
	skipped = 0
	for event_dir in event_dirs:
		try:
			status, evt_path, ch_path, txt_path = _prepare_raw_triplet(event_dir)
		except (
			FileNotFoundError,
			FileExistsError,
			NotADirectoryError,
			ValueError,
		) as exc:
			if not args.skip_incomplete:
				raise
			skipped += 1
			print(f'[skip-incomplete] {event_dir}: {exc}')
			continue

		prepared += 1
		print(f'[{status}] {event_dir}')
		print(f'  evt: {evt_path}')
		print(f'  ch : {ch_path}')
		print(f'  txt: {txt_path}')

	if prepared == 0:
		raise RuntimeError('no event directories were prepared')
	if skipped:
		print(f'[summary] prepared={prepared}, skipped_incomplete={skipped}')
	else:
		print(f'[summary] prepared={prepared}')


if __name__ == '__main__':
	main()
