from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from loki.loki import Loki
from obspy import Stream

from common.config import (
	LokiWaveformStackingInputs,
	LokiWaveformStackingPipelineConfig,
)
from io_util.stream import build_stream_from_downloaded_win32


def _list_event_dirs(cfg: LokiWaveformStackingPipelineConfig) -> list[Path]:
	base = Path(cfg.base_input_dir)
	if not base.is_dir():
		raise FileNotFoundError(f'base_input_dir not found: {base}')

	dirs = []
	for p in sorted(base.glob(cfg.event_glob)):
		if not p.is_dir():
			continue
		if (p / 'event.json').is_file():
			dirs.append(p)

	if cfg.max_events is not None and cfg.max_events > 0:
		dirs = dirs[: int(cfg.max_events)]

	if not dirs:
		raise ValueError(f'No event dirs found under: {base} (glob={cfg.event_glob})')
	return dirs


def pipeline_loki_waveform_stacking(
	cfg: LokiWaveformStackingPipelineConfig, inputs: LokiWaveformStackingInputs
) -> None:
	cfg.loki_data_path.mkdir(parents=True, exist_ok=True)
	cfg.loki_output_path.mkdir(parents=True, exist_ok=True)

	event_dirs = _list_event_dirs(cfg)

	streams_by_event: dict[str, Stream] = {}

	for event_dir in event_dirs:
		event_name = event_dir.name

		# LOKIが os.walk(data_path) の leaf dir をイベントとして列挙するので空dirを作る
		(cfg.loki_data_path / event_name).mkdir(parents=True, exist_ok=True)

		st = build_stream_from_downloaded_win32(
			event_dir,
			base_sampling_rate_hz=int(inputs.base_sampling_rate_hz),
			components_order=('U', 'N', 'E'),
		)
		streams_by_event[event_name] = st
		print(f'prepared stream: event={event_name} n_traces={len(st)} dir={event_dir}')

	l1 = Loki(
		str(cfg.loki_data_path),
		str(cfg.loki_output_path),
		str(cfg.loki_db_path),
		str(cfg.loki_hdr_filename),
		mode='locator',
	)

	inputs_dict = asdict(inputs)

	l1.location(
		extension=cfg.extension,
		comp=list(cfg.comp),
		precision=cfg.precision,
		search=cfg.search,
		streams_by_event=streams_by_event,
		**inputs_dict,
	)
