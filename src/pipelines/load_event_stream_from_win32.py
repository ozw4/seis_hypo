# %%
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from HinetPy import Client
from obspy import UTCDateTime

from io_util.stream import build_stream_from_array
from jma.download import download_win_for_event
from jma.win32_reader import read_event_win32_window_as_array
from loki_tools.create_events_dir import create_event_input_dir


def prepare_single_event_stream(
	event_row: pd.Series,
	client: Client,
	station_list: Sequence[str],
	*,
	base_input_dir: str | Path,
	network_code: str = '0101',
	pre_sec: int = 20,
	post_sec: int = 120,
	base_sampling_rate_HZ: int = 100,
):
	event_dir = create_event_input_dir(
		event_row,
		base_input_dir=base_input_dir,
		pre_sec=pre_sec,
		post_sec=post_sec,
	)

	cnt_path_list, ch_path = download_win_for_event(
		client,
		station_list=station_list,
		event_row=event_row,
		event_dir=event_dir,
		network_code=network_code,
		pre_sec=pre_sec,
		post_sec=post_sec,
	)
	arr_event, station_df, t_start, t_end = read_event_win32_window_as_array(
		event_row,
		ch_path,
		cnt_path_list,
		base_sampling_rate_HZ=base_sampling_rate_HZ,
		pre_sec=pre_sec,
		post_sec=post_sec,
	)
	starttime_utc = UTCDateTime(t_start.isoformat())

	stream = build_stream_from_array(
		data=arr_event,
		df_selected=station_df,
		starttime=starttime_utc,
		sampling_rate_HZ=base_sampling_rate_HZ,
	)

	return event_dir, stream, t_start, t_end


# %%
