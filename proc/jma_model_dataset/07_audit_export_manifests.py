from __future__ import annotations

import argparse
from pathlib import Path

from jma_model_dataset.manifest_audit import (
	audit_export_manifest_dirs,
	format_manifest_audit_summary,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			'Audit one or more monthly export manifest directories under '
			'flows/jma_model_dataset/export_manifests/YYYY-MM/.'
		)
	)
	parser.add_argument(
		'manifest_month_dirs',
		nargs='+',
		type=Path,
		help=(
			'Monthly manifest directory path(s), each containing '
			'event_manifest.jsonl and event_station_manifest.csv'
		),
	)
	return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
	args = parse_args(argv)
	summary = audit_export_manifest_dirs(args.manifest_month_dirs)
	print(format_manifest_audit_summary(summary))


if __name__ == '__main__':
	main()
