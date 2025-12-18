# %%
# %%
from __future__ import annotations

import datetime as dt
import shutil
from datetime import timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import Stream

from common.config import LokiWaveformStackingInputs, LokiWaveformStackingPipelineConfig
from common.core import load_event_json
from common.load_config import load_config
from common.run_snapshot import save_many_yaml_and_effective
from io_util.stream import build_stream_from_downloaded_win32
from loki_tools.loki_parse import parse_loki_header, parse_phs_absolute_times
from pipelines.loki_waveform_stacking_pipelines import pipeline_loki_waveform_stacking
from viz.gather import plot_gather
from waveform.preprocess import DetrendBandpassSpec, preprocess_stream_detrend_bandpass


def _as_utc_aware(d: dt.datetime) -> dt.datetime:
	if d.tzinfo is None:
		return d.replace(tzinfo=timezone.utc)
	return d.astimezone(timezone.utc)


def _station_key(network: str | None, station: str) -> str:
	return f'{network}.{station}' if network else station


def _trace_station_comp(tr) -> tuple[str, str]:
	net = getattr(tr.stats, 'network', None)
	sta = getattr(tr.stats, 'station', None)
	cha = getattr(tr.stats, 'channel', None)
	if sta is None or cha is None:
		raise ValueError('trace.stats.station/channel missing')

	sta_full = _station_key(str(net) if net is not None else None, str(sta))
	comp = str(cha)[-1]  # ...U / ...N / ...E
	return sta_full, comp


def _build_gather_matrix(
	st: Stream, comp: str
) -> tuple[np.ndarray, list[str], float, dt.datetime]:
	trs: list[tuple[str, object]] = []
	for tr in st:
		sta_full, c = _trace_station_comp(tr)
		if c == comp:
			trs.append((sta_full, tr))

	if not trs:
		raise ValueError(f'no traces for comp={comp}')

	fs = float(trs[0][1].stats.sampling_rate)
	t0 = trs[0][1].stats.starttime.datetime
	t_start_utc = _as_utc_aware(t0)

	n = min(int(t.stats.npts) for _, t in trs)
	stations = [sta for sta, _ in trs]
	data = np.vstack([t.data[:n].astype(np.float32, copy=False) for _, t in trs])
	return data, stations, fs, t_start_utc


def _picks_to_sample_idx(
	stations: list[str],
	phs_df: pd.DataFrame,
	*,
	fs: float,
	t_start_utc: dt.datetime,
) -> tuple[np.ndarray, np.ndarray]:
	phs_df = phs_df.set_index('station')
	p_idx = np.full(len(stations), np.nan, dtype=float)
	s_idx = np.full(len(stations), np.nan, dtype=float)

	t_start_utc = _as_utc_aware(t_start_utc)

	for i, sta in enumerate(stations):
		if sta not in phs_df.index:
			continue
		tp = phs_df.loc[sta, 'tp']
		ts = phs_df.loc[sta, 'ts']

		tp_dt = _as_utc_aware(tp.to_pydatetime())
		ts_dt = _as_utc_aware(ts.to_pydatetime())

		p_idx[i] = (tp_dt - t_start_utc).total_seconds() * fs
		s_idx[i] = (ts_dt - t_start_utc).total_seconds() * fs

	return p_idx, s_idx


def _list_event_dirs(
	base_input_dir: Path, event_glob: str, max_events: int | None
) -> list[Path]:
	if not base_input_dir.is_dir():
		raise FileNotFoundError(f'base_input_dir not found: {base_input_dir}')

	dirs: list[Path] = []
	for p in sorted(base_input_dir.glob(event_glob)):
		if p.is_dir() and (p / 'event.json').is_file():
			dirs.append(p)

	if max_events is not None and max_events > 0:
		dirs = dirs[: int(max_events)]

	if not dirs:
		raise ValueError(
			f'No event dirs found under: {base_input_dir} (glob={event_glob})'
		)
	return dirs


def _spec_from_inputs(inputs: LokiWaveformStackingInputs) -> DetrendBandpassSpec:
	# inputs 側に pre_* が無ければデフォルトでOK
	return DetrendBandpassSpec(
		detrend=inputs.pre_detrend,
		fstop_lo=float(inputs.pre_fstop_lo),
		fpass_lo=float(inputs.pre_fpass_lo),
		fpass_hi=float(inputs.pre_fpass_hi),
		fstop_hi=float(inputs.pre_fstop_hi),
		gpass=float(inputs.pre_gpass),
		gstop=float(inputs.pre_gstop),
		mad_scale=bool(inputs.pre_mad_scale),
		mad_eps=float(inputs.pre_mad_eps),
		mad_c=float(inputs.pre_mad_c),
	)


def _plot_waveforms_with_picks_for_event(
	*,
	event_dir: Path,
	loki_output_dir: Path,
	header_path: Path,
	base_sampling_rate_hz: int,
	components_order: tuple[str, str, str],
	plot_components: tuple[str, ...],
	y_time: str,
	pre_spec: DetrendBandpassSpec,
) -> None:
	event_id = event_dir.name
	ev_out_dir = loki_output_dir / event_id

	phs_paths = sorted(ev_out_dir.glob('*trial0.phs'))
	if not phs_paths:
		return

	st = build_stream_from_downloaded_win32(
		event_dir,
		base_sampling_rate_hz=base_sampling_rate_hz,
		components_order=components_order,
	)
	preprocess_stream_detrend_bandpass(
		st, spec=pre_spec, fs_expected=float(base_sampling_rate_hz)
	)

	ev = load_event_json(event_dir)
	origin = ev.get('origin_time_jst', None) or ev.get('origin_time', None)
	if origin is None:
		raise ValueError(f'origin_time missing in {event_dir / "event.json"}')

	event_time_utc = pd.to_datetime(origin).tz_convert('UTC').to_pydatetime()
	event_time_utc = _as_utc_aware(event_time_utc)

	stations_df = parse_loki_header(header_path).stations_df
	phs_df = parse_phs_absolute_times(phs_paths[0])  # station,tp,ts（UTC想定）

	for comp in plot_components:
		data, stations, fs, t_start_utc = _build_gather_matrix(st, comp=comp)
		p_idx, s_idx = _picks_to_sample_idx(
			stations, phs_df, fs=fs, t_start_utc=t_start_utc
		)

		sta_meta = stations_df.set_index('station').reindex(stations).reset_index()

		fig, ax = plt.subplots(figsize=(max(10.0, 0.18 * len(stations)), 8))
		plot_gather(
			data,
			station_df=sta_meta.rename(
				columns={'station': 'station', 'lat': 'lat', 'lon': 'lon'}
			),
			# 前処理済みなので、ここでは「表示用」のzscoreだけ入れるのはアリ
			scaling='zscore',
			amp=1.0,
			title=f'event={event_id} comp={comp} (LOKI P/S picks)',
			p_idx=p_idx,
			s_idx=s_idx,
			order_mode='pca',
			ax=ax,
			decim=1,
			detrend=None,  # 二重detrendしない
			taper_frac=0.02,
			y_time=y_time,
			fs=fs if y_time != 'samples' else None,
			t_start=t_start_utc if y_time != 'samples' else None,
			event_time=event_time_utc if y_time == 'relative' else None,
		)

		out_png = ev_out_dir / f'waveform_with_loki_picks_{comp}.png'
		out_png.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(out_png, dpi=200)
		plt.close(fig)


def main() -> None:
	pipeline_yaml = Path('/workspace/data/config/loki_waveform_pipeline.yaml')
	pipeline_preset = 'mobara'

	cfg = load_config(
		LokiWaveformStackingPipelineConfig, pipeline_yaml, pipeline_preset
	)
	print(cfg)

	if not cfg.inputs_yaml.is_file():
		raise FileNotFoundError(f'inputs_yaml not found: {cfg.inputs_yaml}')

	inputs = load_config(LokiWaveformStackingInputs, cfg.inputs_yaml, cfg.inputs_preset)

	save_many_yaml_and_effective(
		out_dir=cfg.loki_output_path,
		items=[
			('pipeline', pipeline_yaml, pipeline_preset, cfg),
			('inputs', cfg.inputs_yaml, cfg.inputs_preset, inputs),
		],
	)

	# ---- LOKI 本体 ----
	pipeline_loki_waveform_stacking(cfg, inputs)

	# ---- 追加: プロット（全イベント）----
	# 直書き運用：必要ならここだけいじる
	plot_components = ('U', 'N')
	y_time = 'relative'  # "samples" | "absolute" | "relative"

	header_path = Path(cfg.loki_hdr_filename)
	if not header_path.is_file():
		raise FileNotFoundError(f'header not found: {header_path}')

	pre_spec = _spec_from_inputs(inputs)

	event_dirs = _list_event_dirs(
		Path(cfg.base_input_dir), cfg.event_glob, cfg.max_events
	)
	for event_dir in event_dirs:
		_plot_waveforms_with_picks_for_event(
			event_dir=event_dir,
			loki_output_dir=Path(cfg.loki_output_path),
			header_path=header_path,
			base_sampling_rate_hz=int(inputs.base_sampling_rate_hz),
			components_order=('U', 'N', 'E'),
			plot_components=plot_components,
			y_time=y_time,
			pre_spec=pre_spec,
		)
	print(f'Waveform plots written under: {cfg.loki_output_path}')

	# ---- cleanup ----
	if cfg.loki_data_path.is_dir():
		shutil.rmtree(cfg.loki_data_path)


if __name__ == '__main__':
	main()
