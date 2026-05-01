# file: src/loki_tools/build_loki.py
from __future__ import annotations

from pathlib import Path

from loki.loki import Loki

from common.config import LokiWaveformStackingPipelineConfig
from loki_tools.loki_parse import LokiHeader, parse_loki_header


def build_loki_with_header(
	cfg: LokiWaveformStackingPipelineConfig,
	*,
	data_path: str | Path | None = None,
	output_path: str | Path | None = None,
) -> tuple[Loki, LokiHeader, Path]:
	db_path = Path(cfg.loki_db_path).resolve()
	hdr_arg = Path(cfg.loki_hdr_filename)

	if hdr_arg.is_absolute():
		header_path = hdr_arg.resolve()
		try:
			hdr_filename_for_loki = str(header_path.relative_to(db_path))
		except ValueError:
			raise ValueError(
				'cfg.loki_hdr_filename is absolute but is not under loki_db_path: '
				f'loki_db_path={db_path}, loki_hdr_filename={header_path}'
			)
	else:
		hdr_filename_for_loki = hdr_arg.as_posix()
		header_path = (db_path / hdr_arg).resolve()

	if not header_path.is_file():
		raise FileNotFoundError(f'header not found: {header_path}')
	header = parse_loki_header(header_path)
	data_root = Path(data_path) if data_path is not None else Path(cfg.loki_data_path)
	output_root = (
		Path(output_path) if output_path is not None else Path(cfg.loki_output_path)
	)
	print(f'Loki DB path: {db_path}')
	print(f'Loki header path: {header_path}')
	print(f'Loki header filename argument: {hdr_filename_for_loki}')
	loki = Loki(
		str(data_root),
		str(output_root),
		str(db_path),
		hdr_filename_for_loki,
		mode='locator',
	)
	return loki, header, header_path
