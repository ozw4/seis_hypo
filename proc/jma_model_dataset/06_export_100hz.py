from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from jma_model_dataset.event_skip_log import (
	MISSING_INPUT_EXCEPTIONS,
	append_event_skip_log,
	is_expected_missing_input_error,
)
from jma_model_dataset.export_100hz import export_event_100hz

DEFAULT_TARGET_FS_HZ = 100
STEP_NAME = '06_export_100hz'


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			'Export flow-scoped JMA model dataset waveforms to a '
			'waveforms_<fs>hz.npz file under flows/jma_model_dataset/export/ '
			'and append monthly aggregate manifests under '
			'flows/jma_model_dataset/export_manifests/YYYY-MM/.'
		)
	)
	parser.add_argument(
		'event_dirs',
		nargs='+',
		type=Path,
		help=(
			'Event directory path(s) whose raw inputs are under raw/ and whose '
			'flow-scoped active/continuous inputs are under '
			'flows/jma_model_dataset/'
		),
	)
	parser.add_argument(
		'--epi-csv',
		required=True,
		type=Path,
		help='Path to the epicenters.csv used to resolve catalog event_id from ORIGIN_JST',
	)
	parser.add_argument(
		'--target-fs-hz',
		type=int,
		default=DEFAULT_TARGET_FS_HZ,
		help=(
			'Target sampling rate for export waveform resampling '
			f'(default: {DEFAULT_TARGET_FS_HZ})'
		),
	)
	parser.add_argument(
		'--skip-if-exists',
		action=argparse.BooleanOptionalAction,
		default=True,
		help=(
			'Skip an event when waveforms_<fs>hz.npz already exists and the '
			'matching monthly manifest entries are already present '
			'(default: true)'
		),
	)
	parser.add_argument(
		'--skip-missing-inputs',
		action=argparse.BooleanOptionalAction,
		default=False,
		help=(
			'Skip the current event and append a flow-scoped skip log when '
			'required per-event input files are missing (default: false)'
		),
	)
	return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
	args = parse_args(argv)
	epi_csv = args.epi_csv.resolve()
	if not epi_csv.is_file():
		raise FileNotFoundError(epi_csv)
	epi_df = pd.read_csv(epi_csv, low_memory=False)
	skipped_events: list[tuple[Path, str, Path]] = []
	ok_count = 0

	for event_dir in args.event_dirs:
		event_dir2 = Path(event_dir).resolve()
		try:
			result = export_event_100hz(
				event_dir2,
				epi_df=epi_df,
				epi_source=str(epi_csv),
				target_fs_hz=args.target_fs_hz,
				skip_if_exists=args.skip_if_exists,
			)
		except MISSING_INPUT_EXCEPTIONS as error:
			if not (
				args.skip_missing_inputs
				and is_expected_missing_input_error(error, event_dir=event_dir2)
			):
				raise
			log_path = append_event_skip_log(
				event_dir2,
				step_name=STEP_NAME,
				reason='missing_input',
				error=error,
			)
			message = str(error)
			print(
				f'[skip] step=06 event_dir={event_dir2} '
				f'error={error.__class__.__name__}: {message} log={log_path}'
			)
			skipped_events.append((event_dir2, message, log_path))
			continue

		ok_count += 1
		if result.skipped:
			print(
				f'[step6] {result.event_id}: '
				f'skipped existing waveform and verified manifests -> '
				f'{result.waveforms_path}'
			)
			continue

		print(
			f'[step6] {result.event_id}: '
			f'stations={result.station_count} '
			f'samples={result.n_samples} '
			f'fs={result.target_fs_hz}Hz '
			f'window={result.start_time}..{result.end_time_exclusive} '
			f'-> {result.outdir}'
		)
		print(f'[export] {result.event_id} -> {result.waveforms_path}')
		print(f'[manifest] {result.event_id} -> {result.event_manifest_path}')
		print(f'[manifest] {result.event_id} -> {result.event_station_manifest_path}')

	print(f'[summary] ok={ok_count} skipped_missing_input={len(skipped_events)}')
	if skipped_events:
		for event_dir, message, log_path in skipped_events:
			print(f'[skipped] event_dir={event_dir} error={message} log={log_path}')


if __name__ == '__main__':
	main()
