# file: src/loki_tools/loki_build.py
from __future__ import annotations

from pathlib import Path

from loki.loki import Loki

from common.config import LokiWaveformStackingPipelineConfig
from loki_tools.loki_parse import LokiHeader, parse_loki_header


def build_loki_with_header(
	cfg: LokiWaveformStackingPipelineConfig,
	*,
	data_path: Path | None = None,
	output_path: Path | None = None,
) -> tuple[Loki, LokiHeader, Path]:
	header_path = Path(cfg.loki_db_path) / Path(cfg.loki_hdr_filename)
	if not header_path.is_file():
		raise FileNotFoundError(f'header not found: {header_path}')
	header = parse_loki_header(header_path)
	loki = Loki(
		str(data_path or cfg.loki_data_path),
		str(output_path or cfg.loki_output_path),
		str(cfg.loki_db_path),
		str(header_path),
		mode='locator',
	)
	return loki, header, header_path
