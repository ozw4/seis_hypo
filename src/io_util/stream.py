import numpy as np
import pandas as pd
from obspy import Stream, Trace, UTCDateTime


def infer_net_sta_loc(station_field: str) -> tuple[str, str, str]:
	"""channel_table の station 表記から network/station/location を推定。"""
	if '.' in station_field:
		net, sta = station_field.split('.', 1)
		return net, sta, ''
	return '', station_field, ''


def build_stream_from_array(
	data: np.ndarray,
	df_selected: pd.DataFrame,
	*,
	starttime: UTCDateTime,
	sampling_rate_HZ: int,
) -> Stream:
	"""Ndarray + チャネルメタから ObsPy Stream を構築する純粋関数。"""
	if data.ndim != 2:
		raise ValueError(f'expected 2D array, got shape={data.shape}')

	if len(df_selected) != data.shape[0]:
		raise ValueError(
			'channel count mismatch between ndarray and df_selected: '
			f'{data.shape[0]} vs {len(df_selected)}'
		)

	traces: list[Trace] = []
	fs = float(sampling_rate_HZ)

	for i in range(len(df_selected)):
		row = df_selected.iloc[i]

		net, sta, loc = infer_net_sta_loc(str(row['station']))
		component = str(row['component'])

		stats = {
			'network': net,
			'station': sta,
			'location': loc,
			'channel': component,
			'starttime': starttime,
			'sampling_rate': fs,
		}

		tr = Trace(data=data[i].astype(np.float32, copy=False), header=stats)

		tr.stats.hinet = {
			'ch_hex': str(row.get('ch_hex', '')),
			'ch_int': int(row['ch_int']) if 'ch_int' in row else None,
			'input_unit': str(row.get('input_unit', '')),
			'conv_coeff': float(row['conv_coeff']) if 'conv_coeff' in row else None,
		}

		if 'lat' in row and 'lon' in row:
			tr.stats.coordinates = {
				'latitude': float(row.get('lat')),
				'longitude': float(row.get('lon')),
				'elevation': float(row.get('elevation_m', 0.0)),
			}

		traces.append(tr)

	return Stream(traces=traces)
