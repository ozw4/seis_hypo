from __future__ import annotations

from pathlib import Path

from common.core import load_event_json
from common.time_util import get_event_origin_utc
from io_util.stream import build_stream_from_downloaded_win32
from loki_tools.loki_parse import parse_loki_header, parse_phs_absolute_times
from viz.gather_util import build_gather_matrix, picks_to_sample_idx
from viz.loki.waveforms_with_picks import save_gather_with_loki_picks

# EqT backend（あなたが src/pick などに追加した想定）
# backend_eqt_probs(x_3cn, fs, weights=..., in_samples=..., overlap=..., batch_size=...)
from waveform.preprocess import DetrendBandpassSpec, preprocess_stream_detrend_bandpass


def plot_waveforms_with_picks_for_event(
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
	event_time_utc = get_event_origin_utc(
		ev, event_json_path=event_dir / 'event.json'
	).to_pydatetime()

	stations_df = parse_loki_header(header_path).stations_df
	phs_df = parse_phs_absolute_times(phs_paths[0])  # station,tp,ts（UTC想定）

	for comp in plot_components:
		data, stations, fs, t_start_utc = build_gather_matrix(st, comp=comp)
		p_idx, s_idx = picks_to_sample_idx(
			stations, phs_df, fs=fs, t_start_utc=t_start_utc
		)

		sta_meta = stations_df.set_index('station').reindex(stations).reset_index()

		out_png = ev_out_dir / f'waveform_with_loki_picks_{comp}.png'

		save_gather_with_loki_picks(
			data,
			station_df=sta_meta.rename(
				columns={'station': 'station', 'lat': 'lat', 'lon': 'lon'}
			),
			event_id=event_id,
			comp=comp,
			p_idx=p_idx,
			s_idx=s_idx,
			out_png=out_png,
			y_time=y_time,
			fs=fs,
			t_start_utc=t_start_utc,
			event_time_utc=event_time_utc,
		)
