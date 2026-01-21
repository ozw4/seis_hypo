# %%
#!/usr/bin/env python3
# proc/loki_hypo/run_plot_waveforms_with_loki_picks.py
#
# 目的:
# - event_dir から Stream を毎回生成（保存しない）
# - LOKI出力の .phs (P/S arrival time) を読み
# - plot_gather の p_idx/s_idx に載せて可視化
#
# 出力:
# - <loki_output_dir>/<event_id>/waveform_with_loki_picks_U.png
# - <loki_output_dir>/<event_id>/waveform_with_loki_picks_N.png

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from common.core import load_event_json
from common.time_util import get_event_origin_utc
from io_util.stream import build_stream_from_downloaded_win32
from loki_tools.loki_parse import parse_loki_header, parse_phs_absolute_times
from viz.gather import (
	plot_gather,  # ←あなたの plot_gather があるモジュールに合わせてimport先修正して
)
from viz.gather_util import build_gather_matrix, picks_to_sample_idx
from waveform.preprocess import (
	DetrendBandpassSpec,
	preprocess_stream_detrend_bandpass,
)

def main() -> None:
	# =========================
	# 直書き設定
	# =========================
	event_id = '3163344'

	base_input_dir = Path('/workspace/data/waveform/jma')
	loki_output_dir = Path(
		'/workspace/proc/loki_hypo/mobara/loki_output_mobara_w_preprocess'
	)
	header_path = Path('/workspace/proc/loki_hypo/mobara_traveltime/db/header.hdr')

	base_sampling_rate_hz = 100
	components_order = ('U', 'N', 'E')

	# どの成分を描く？
	# - Pの初動確認なら U
	# - Sの初動確認なら N/E（まずはNを推奨）
	plot_components = ('U', 'N')

	# y軸の時間表示
	y_time = 'relative'  # "samples" | "absolute" | "relative"
	# =========================

	event_dir = base_input_dir / str(event_id)
	if not event_dir.is_dir():
		raise FileNotFoundError(f'event_dir not found: {event_dir}')

	# Streamは毎回生成（保存しない）
	st = build_stream_from_downloaded_win32(
		event_dir,
		base_sampling_rate_hz=base_sampling_rate_hz,
		components_order=components_order,
	)
	pre_spec = DetrendBandpassSpec()
	preprocess_stream_detrend_bandpass(
		st,
		spec=pre_spec,
		fs_expected=float(base_sampling_rate_hz),
	)

	# event_time（相対表示の0基準）
	ev = load_event_json(event_dir)
	event_time_utc = get_event_origin_utc(
		ev, event_json_path=event_dir / 'event.json'
	).to_pydatetime()

	# header stations（並び替え用）
	stations_df = parse_loki_header(header_path).stations_df

	# .phs 読み
	ev_out_dir = loki_output_dir / str(event_id)
	phs_paths = sorted(ev_out_dir.glob('*trial0.phs'))
	if not phs_paths:
		raise FileNotFoundError(f'no *trial0.phs under: {ev_out_dir}')
	phs_df = parse_phs_absolute_times(phs_paths[0])

	for comp in plot_components:
		data, stations, fs, t_start_utc = build_gather_matrix(
			st,
			comp=comp,
			start_time_mode='first',
			length_mode='min',
			align_on_start=False,
		)
		p_idx, s_idx = picks_to_sample_idx(
			stations,
			phs_df,
			fs=fs,
			t_start_utc=t_start_utc,
			dropna=False,
			deduplicate=False,
		)

		# station_df を trace順に合わせる（無い局は落ちるので NaNのまま）
		sta_meta = stations_df.set_index('station').reindex(stations).reset_index()
		title = f'event={event_id} comp={comp} (LOKI P/S picks)'

		fig, ax = plt.subplots(figsize=(max(10.0, 0.18 * len(stations)), 8))
		plot_gather(
			data,
			station_df=sta_meta.rename(
				columns={'station': 'station', 'lat': 'lat', 'lon': 'lon'}
			),
			scaling='zscore',
			amp=1.0,
			title=title,
			p_idx=p_idx,
			s_idx=s_idx,
			order_mode='pca',
			ax=ax,
			decim=1,
			detrend='linear',
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
		print(f'Wrote: {out_png}')


if __name__ == '__main__':
	main()
