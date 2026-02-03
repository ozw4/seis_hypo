from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from obspy import UTCDateTime

from common.config import (
	JmaDtPickErrorBandpassConfig,
	JmaDtPickErrorPreprocessConfig,
)


def _preprocess_cfg(
	*, detrend: str | None = 'linear'
) -> JmaDtPickErrorPreprocessConfig:
	return JmaDtPickErrorPreprocessConfig(
		preprocess_preset='jma_snr_pick_table_v1',
		fs_target_hz=100.0,
		detrend=detrend,
		bandpass=JmaDtPickErrorBandpassConfig(
			fstop_lo=0.5,
			fpass_lo=1.0,
			fpass_hi=20.0,
			fstop_hi=25.0,
			gpass=1.0,
			gstop=40.0,
		),
	)


@dataclass(frozen=True)
class _Src:
	source_id: str
	data_path: str
	ch_path: str


def _stub_inventory(
	*,
	start_time: str = '2026-02-03T00:00:00',
	span_seconds: int = 10,
	include_second_source: bool = False,
	second_start_time: str | None = None,
	second_span_seconds: int | None = None,
):
	sources = [
		_Src(source_id='S1', data_path='/dummy/S1', ch_path='/dummy/S1.ch'),
	]
	sources_meta = {
		'S1': {'start_time': start_time, 'span_seconds': span_seconds},
	}
	station_meta = {
		'STA1': {'U': {'source_id': 'S1', 'ch_int': 1}},
		'STA2': {'U': {'source_id': 'S1', 'ch_int': 2}},
	}
	if include_second_source:
		sources.append(
			_Src(source_id='S2', data_path='/dummy/S2', ch_path='/dummy/S2.ch')
		)
		sources_meta['S2'] = {
			'start_time': second_start_time or start_time,
			'span_seconds': int(
				second_span_seconds if second_span_seconds is not None else span_seconds
			),
		}
		station_meta['STA3'] = {'U': {'source_id': 'S2', 'ch_int': 3}}

	class _Inv:
		def __init__(self):
			self.sources = sources
			self.sources_meta = sources_meta
			self.station_meta = station_meta

	return _Inv()


def _stub_channel_table() -> pd.DataFrame:
	return pd.DataFrame(
		{
			'ch_hex': ['0001', '0002', '0003'],
			'ch_int': [1, 2, 3],
			'conv_coeff': [1.0, 1.0, 1.0],
			'station': ['STA1', 'STA2', 'STA3'],
			'component': ['U', 'U', 'U'],
		}
	)


def test_load_u_stream_for_event_happy_path_batches_read_win32(
	monkeypatch, tmp_path
) -> None:
	from jma.prepare import jma_waveform_loader as wl

	event_dir = tmp_path / 'event'
	event_dir.mkdir()

	inv = _stub_inventory()
	monkeypatch.setattr(wl, 'build_inventory', lambda _: inv)
	monkeypatch.setattr(wl, 'read_hinet_channel_table', lambda _: _stub_channel_table())

	def scan_fs(_data_path, *, channel_filter, on_mixed):
		assert on_mixed == 'drop'
		return {int(ch): 200 for ch in channel_filter}

	monkeypatch.setattr(wl, 'scan_channel_sampling_rate_map_win32', scan_fs)

	read_calls: list[dict[str, object]] = []

	def read_win32(
		data_path, ch_table, *, base_sampling_rate_HZ, duration_SECOND, channels_hex
	):
		read_calls.append(
			{
				'data_path': str(data_path),
				'fs': int(base_sampling_rate_HZ),
				'dur': int(duration_SECOND),
				'channels_hex': channels_hex,
				'ch_hex': list(ch_table['ch_hex']),
			}
		)
		n = int(base_sampling_rate_HZ) * int(duration_SECOND)
		out = []
		for i in range(len(ch_table)):
			out.append(np.arange(n, dtype=np.float32) + float(i))
		return out

	monkeypatch.setattr(wl, 'read_win32', read_win32)

	def resample_window_poly(x2d, *, fs_in, fs_out, out_len):
		x = np.asarray(x2d[0], dtype=np.float64)
		if x.size == int(out_len):
			y = x
		else:
			xi = np.linspace(0.0, float(x.size - 1), num=int(out_len))
			y = np.interp(xi, np.arange(x.size), x)
		return np.asarray([y], dtype=np.float64)

	monkeypatch.setattr(wl, 'resample_window_poly', resample_window_poly)
	monkeypatch.setattr(wl, 'bandpass_iir_filtfilt', lambda x, **_: np.asarray(x))
	monkeypatch.setattr(wl, 'sp_detrend', lambda x, **_: np.asarray(x))

	cfg = _preprocess_cfg(detrend='linear')
	res = wl.load_u_stream_for_event(event_dir, stations=None, preprocess_cfg=cfg)

	# return contract
	assert res.fs_hz == 100.0
	assert res.t0 == datetime.fromisoformat(inv.sources_meta['S1']['start_time'])
	assert set(res.stations_used) == {'STA1', 'STA2'}
	assert len(res.stream) == 2

	# batching: STA1/STA2 are same (source_id, fs_in) => read_win32 once with 2 chans
	assert len(read_calls) == 1
	assert read_calls[0]['channels_hex'] is None
	assert read_calls[0]['ch_hex'] == ['0001', '0002']

	# trace contract: station/channel/starttime/delta/len
	for tr in res.stream:
		assert tr.stats.channel == 'HHU'
		assert tr.stats.station in {'STA1', 'STA2'}
		assert tr.stats.starttime == UTCDateTime(res.t0 - timedelta(hours=9))
		assert tr.stats.delta == pytest.approx(0.01)
		assert len(tr.data) == 10 * 100  # span_seconds * fs_target_hz


def test_load_u_stream_for_event_window_mismatch_skips(monkeypatch, tmp_path) -> None:
	from jma.prepare import jma_waveform_loader as wl

	event_dir = tmp_path / 'event'
	event_dir.mkdir()

	inv = _stub_inventory(
		include_second_source=True,
		second_start_time='2026-02-03T00:00:01',
		second_span_seconds=10,
	)
	monkeypatch.setattr(wl, 'build_inventory', lambda _: inv)
	monkeypatch.setattr(wl, 'read_hinet_channel_table', lambda _: _stub_channel_table())
	monkeypatch.setattr(
		wl,
		'scan_channel_sampling_rate_map_win32',
		lambda *_args, **kwargs: {int(ch): 200 for ch in kwargs['channel_filter']},
	)

	def read_win32(
		_data_path, ch_table, *, base_sampling_rate_HZ, duration_SECOND, channels_hex
	):
		n = int(base_sampling_rate_HZ) * int(duration_SECOND)
		out = []
		for i in range(len(ch_table)):
			out.append(np.arange(n, dtype=np.float32) + float(i))
		return out

	monkeypatch.setattr(wl, 'read_win32', read_win32)
	monkeypatch.setattr(
		wl,
		'resample_window_poly',
		lambda x2d, **kw: np.asarray([np.zeros(kw['out_len'])], dtype=np.float64),
	)
	monkeypatch.setattr(wl, 'bandpass_iir_filtfilt', lambda x, **_: np.asarray(x))
	monkeypatch.setattr(wl, 'sp_detrend', lambda x, **_: np.asarray(x))

	cfg = _preprocess_cfg(detrend='linear')
	res = wl.load_u_stream_for_event(event_dir, stations=None, preprocess_cfg=cfg)

	# STA3 should be skipped due to mismatched window; others remain
	assert set(res.stations_used) == {'STA1', 'STA2'}
	assert all(tr.stats.station in {'STA1', 'STA2'} for tr in res.stream)

	reasons = {(s['station'], s['reason']) for s in res.skips}
	assert ('STA3', 'window_mismatch') in reasons


def test_load_u_stream_for_event_rejects_unsupported_detrend(
	monkeypatch, tmp_path
) -> None:
	from jma.prepare import jma_waveform_loader as wl

	event_dir = tmp_path / 'event'
	event_dir.mkdir()

	inv = _stub_inventory()
	monkeypatch.setattr(wl, 'build_inventory', lambda _: inv)
	monkeypatch.setattr(wl, 'read_hinet_channel_table', lambda _: _stub_channel_table())
	monkeypatch.setattr(
		wl,
		'scan_channel_sampling_rate_map_win32',
		lambda *_args, **kwargs: {int(ch): 200 for ch in kwargs['channel_filter']},
	)

	def read_win32(
		_data_path, ch_table, *, base_sampling_rate_HZ, duration_SECOND, channels_hex
	):
		n = int(base_sampling_rate_HZ) * int(duration_SECOND)
		out = []
		for i in range(len(ch_table)):
			out.append(np.arange(n, dtype=np.float32) + float(i))
		return out

	monkeypatch.setattr(wl, 'read_win32', read_win32)
	monkeypatch.setattr(
		wl,
		'resample_window_poly',
		lambda x2d, **kw: np.asarray([np.zeros(kw['out_len'])], dtype=np.float64),
	)
	monkeypatch.setattr(wl, 'bandpass_iir_filtfilt', lambda x, **_: np.asarray(x))

	cfg = _preprocess_cfg(detrend='constant')
	with pytest.raises(ValueError, match='unsupported detrend'):
		wl.load_u_stream_for_event(event_dir, stations=None, preprocess_cfg=cfg)


def test_load_u_stream_for_event_missing_dir_raises(tmp_path) -> None:
	from jma.prepare.jma_waveform_loader import load_u_stream_for_event

	cfg = _preprocess_cfg(detrend='linear')
	with pytest.raises(FileNotFoundError, match='event_dir not found'):
		load_u_stream_for_event(
			tmp_path / 'no_such_event', stations=None, preprocess_cfg=cfg
		)
