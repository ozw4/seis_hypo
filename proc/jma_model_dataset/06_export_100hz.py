from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from jma_model_dataset.export_100hz import export_event_100hz

DEFAULT_TARGET_FS_HZ = 100


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
	return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
	args = parse_args(argv)
	epi_csv = args.epi_csv.resolve()
	if not epi_csv.is_file():
		raise FileNotFoundError(epi_csv)
	epi_df = pd.read_csv(epi_csv, low_memory=False)

	for event_dir in args.event_dirs:
		result = export_event_100hz(
			event_dir,
			epi_df=epi_df,
			epi_source=str(epi_csv),
			target_fs_hz=args.target_fs_hz,
			skip_if_exists=args.skip_if_exists,
		)
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
		print(
			f'[manifest] {result.event_id} -> '
			f'{result.event_station_manifest_path}'
		)


if __name__ == '__main__':
	main()
