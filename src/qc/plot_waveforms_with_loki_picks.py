from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from common.core import load_event_json
from common.time_util import origin_to_utc
from io_util.stream import build_stream_from_downloaded_win32
from loki_tools.loki_parse import parse_loki_header, parse_phs_absolute_times
from loki_tools.plot_waveforms_with_loki_picks import plot_gather
from viz.gather_util import build_gather_matrix, picks_to_sample_idx

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
	origin = ev.get('origin_time_jst', None) or ev.get('origin_time', None)
	if origin is None:
		raise ValueError(f'origin_time missing in {event_dir / "event.json"}')

	event_time_utc = origin_to_utc(origin).to_pydatetime()

	stations_df = parse_loki_header(header_path).stations_df
	phs_df = parse_phs_absolute_times(phs_paths[0])  # station,tp,ts（UTC想定）

	for comp in plot_components:
		data, stations, fs, t_start_utc = build_gather_matrix(st, comp=comp)
		p_idx, s_idx = picks_to_sample_idx(
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
