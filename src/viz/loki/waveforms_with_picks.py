from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from loki_tools.plot_waveforms_with_loki_picks import plot_gather
from viz.core.fig_io import save_figure


def save_gather_with_loki_picks(
	data: np.ndarray,
	*,
	station_df: Any,
	event_id: str,
	comp: str,
	p_idx: np.ndarray | None,
	s_idx: np.ndarray | None,
	out_png: Path,
	y_time: str,
	fs: float,
	t_start_utc: Any,
	event_time_utc: Any,
	scaling: str = 'zscore',
	amp: float = 1.0,
	order_mode: str = 'pca',
	decim: int = 1,
	taper_frac: float = 0.02,
	dpi: int = 200,
) -> Path:
	fig_w = max(10.0, 0.18 * len(station_df))
	fig, ax = plt.subplots(figsize=(fig_w, 8))

	plot_gather(
		data,
		station_df=station_df,
		scaling=scaling,
		amp=amp,
		title=f'event={event_id} comp={comp} (LOKI P/S picks)',
		p_idx=p_idx,
		s_idx=s_idx,
		order_mode=order_mode,
		ax=ax,
		decim=decim,
		detrend=None,
		taper_frac=taper_frac,
		y_time=y_time,
		fs=fs if y_time != 'samples' else None,
		t_start=t_start_utc if y_time != 'samples' else None,
		event_time=event_time_utc if y_time == 'relative' else None,
	)

	return save_figure(fig, out_png, dpi=dpi)
