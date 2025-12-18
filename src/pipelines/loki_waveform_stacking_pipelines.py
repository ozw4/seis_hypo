# file: src/pipelines/loki_waveform_stacking_pipelines.py
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
from waveform.preprocess import DetrendBandpassSpec, preprocess_stream_detrend_bandpass

_DEFAULT_PRE = {
	'pre_detrend': 'linear',
	'pre_fstop_lo': 0.5,
	'pre_fpass_lo': 1.0,
	'pre_fpass_hi': 20.0,
	'pre_fstop_hi': 30.0,
	'pre_gpass': 1.0,
	'pre_gstop': 40.0,
	'pre_mad_scale': False,
	'pre_mad_eps': 1.0,
	'pre_mad_c': 6.0,
}

_PRE_KEYS = {
	'pre_enable',
	'pre_detrend',
	'pre_fstop_lo',
	'pre_fpass_lo',
	'pre_fpass_hi',
	'pre_fstop_hi',
	'pre_gpass',
	'pre_gstop',
}


def _list_event_dirs(cfg: LokiWaveformStackingPipelineConfig) -> list[Path]:
	base = Path(cfg.base_input_dir)
	if not base.is_dir():
		raise FileNotFoundError(f'base_input_dir not found: {base}')

	dirs: list[Path] = []
	for p in sorted(base.glob(cfg.event_glob)):
		if p.is_dir() and (p / 'event.json').is_file():
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

	# 前処理specは inputs から作る（yaml管理）。属性が無い場合はデフォルト値を使用。
	pre_spec = DetrendBandpassSpec(
		detrend=getattr(inputs, 'pre_detrend', _DEFAULT_PRE['pre_detrend']),
		fstop_lo=getattr(inputs, 'pre_fstop_lo', _DEFAULT_PRE['pre_fstop_lo']),
		fpass_lo=getattr(inputs, 'pre_fpass_lo', _DEFAULT_PRE['pre_fpass_lo']),
		fpass_hi=getattr(inputs, 'pre_fpass_hi', _DEFAULT_PRE['pre_fpass_hi']),
		fstop_hi=getattr(inputs, 'pre_fstop_hi', _DEFAULT_PRE['pre_fstop_hi']),
		gpass=getattr(inputs, 'pre_gpass', _DEFAULT_PRE['pre_gpass']),
		gstop=getattr(inputs, 'pre_gstop', _DEFAULT_PRE['pre_gstop']),
		mad_scale=bool(getattr(inputs, 'pre_mad_scale', _DEFAULT_PRE['pre_mad_scale'])),
		mad_eps=float(getattr(inputs, 'pre_mad_eps', _DEFAULT_PRE['pre_mad_eps'])),
		mad_c=float(getattr(inputs, 'pre_mad_c', _DEFAULT_PRE['pre_mad_c'])),
	)
	fs_expected = float(inputs.base_sampling_rate_hz)

	for event_dir in event_dirs:
		event_name = event_dir.name
		(cfg.loki_data_path / event_name).mkdir(parents=True, exist_ok=True)

		st = build_stream_from_downloaded_win32(
			event_dir,
			base_sampling_rate_hz=int(inputs.base_sampling_rate_hz),
			components_order=('U', 'N', 'E'),
		)

		if inputs.pre_enable:
			preprocess_stream_detrend_bandpass(
				st,
				spec=pre_spec,
				fs_expected=fs_expected,
			)

		streams_by_event[event_name] = st
		print(
			f'prepared stream: event={event_name} n_traces={len(st)} dir={event_dir} '
			f'(pre={"on" if inputs.pre_enable else "off"})'
		)

	l1 = Loki(
		str(cfg.loki_data_path),
		str(cfg.loki_output_path),
		str(cfg.loki_db_path),
		str(cfg.loki_hdr_filename),
		mode='locator',
	)

	inputs_dict = asdict(inputs)
	# ★ 前処理キーは LOKI に渡さない
	loki_kwargs = {k: v for k, v in inputs_dict.items() if k not in _PRE_KEYS}

	l1.location(
		extension=cfg.extension,
		comp=list(cfg.comp),
		precision=cfg.precision,
		search=cfg.search,
		streams_by_event=streams_by_event,
		**loki_kwargs,
	)
