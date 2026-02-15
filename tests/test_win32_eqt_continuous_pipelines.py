from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pipelines.win32_eqt_continuous_pipelines as pl


def _basic_ch_table() -> pd.DataFrame:
	return pd.DataFrame(
		{
			'ch_hex': ['0001', '0002', '0003', '0004', '0005', '0006'],
			'ch_int': [1, 2, 3, 4, 5, 6],
			'conv_coeff': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
			'station': ['STA1', 'STA1', 'STA1', 'STA2', 'STA2', 'STA2'],
			'component': ['U', 'N', 'E', 'U', 'N', 'E'],
		}
	)


def test_parse_win32_cnt_filename_ok_and_ng():
	info = pl.parse_win32_cnt_filename('win_0301_200912170000_10m_4dd999af.cnt')
	assert info.network_code == '0301'
	assert info.start_jst == dt.datetime(2009, 12, 17, 0, 0)
	assert info.span_min == 10

	with pytest.raises(ValueError):
		pl.parse_win32_cnt_filename('bad_name.cnt')
	with pytest.raises(ValueError):
		pl.parse_win32_cnt_filename('win_0301_200912170000_0m_4dd999af.cnt')


def test_iter_win32_station_windows_hop_and_boundary(monkeypatch):
	def fake_read_win32(
		file_path,
		channel_table,
		*,
		base_sampling_rate_HZ,
		duration_SECOND,
		channels_hex=None,
		station=None,
		components=None,
	):
		info = pl.parse_win32_cnt_filename(file_path)
		ref = dt.datetime(2020, 1, 1, 0, 0, 0)
		offset_s = int((info.start_jst - ref).total_seconds())
		n_ch = len(channel_table)
		n_t = int(base_sampling_rate_HZ) * int(duration_SECOND)
		t = np.arange(n_t, dtype=np.float32) + float(offset_s)
		out = np.zeros((n_ch, n_t), dtype=np.float32)
		for i in range(n_ch):
			out[i, :] = t + float(i) * 1000.0
		return out

	monkeypatch.setattr(pl, 'read_win32', fake_read_win32)

	cnt_paths = [
		Path('win_0301_202001010000_1m_aaaaaaaa.cnt'),
		Path('win_0301_202001010001_1m_bbbbbbbb.cnt'),
	]
	windows = list(
		pl.iter_win32_station_windows(
			cnt_paths=cnt_paths,
			ch_table=_basic_ch_table(),
			target_fs_hz=1.0,
			eqt_in_samples=8,
			eqt_overlap=4,
			use_resampled=False,
		)
	)

	assert len(windows) == 29

	w0, m0 = windows[0]
	assert w0.shape == (2, 3, 8)
	assert m0.window_start_jst == dt.datetime(2020, 1, 1, 0, 0, 0)

	w1, m1 = windows[1]
	assert m1.window_start_jst == dt.datetime(2020, 1, 1, 0, 0, 4)
	assert np.allclose(w1[0, 0, :], np.array([4, 5, 6, 7, 8, 9, 10, 11], dtype=float))

	w_cross, m_cross = windows[14]
	assert m_cross.window_start_jst == dt.datetime(2020, 1, 1, 0, 0, 56)
	assert np.allclose(
		w_cross[0, 0, :],
		np.array([56, 57, 58, 59, 60, 61, 62, 63], dtype=float),
	)


def test_pick_time_jst_iso_has_plus09(monkeypatch, tmp_path: Path):
	class DummyRunner:
		def __init__(self, weights: str, in_samples: int, batch_stations: int):
			self.in_samples = int(in_samples)

		def predict_window(self, wave: np.ndarray):
			b, _c, l = wave.shape
			det = np.ones((b, l), dtype=np.float32)
			p = np.zeros((b, l), dtype=np.float32)
			s = np.zeros((b, l), dtype=np.float32)
			p[:, 5] = 0.9
			s[:, 7] = 0.8
			return det, p, s

	def fake_prepare(_ch_table):
		df = pd.DataFrame(
			{
				'station': ['STA1', 'STA1', 'STA1'],
				'component': ['U', 'N', 'E'],
			}
		)
		return df, ['STA1']

	def fake_iter(**kwargs):
		t0 = dt.datetime(2020, 1, 1, 0, 0, 0)
		meta = pl.Win32WindowMeta(
			network_code='0301',
			window_start_jst=t0,
			window_start_epoch_ms=pl._jst_to_epoch_ms(t0),
		)
		yield np.zeros((1, 3, 10), dtype=np.float32), meta

	monkeypatch.setattr(
		pl,
		'_build_eqt_runner_3c',
		lambda **kwargs: DummyRunner(
			weights=str(kwargs['weights']),
			in_samples=int(kwargs['in_samples']),
			batch_stations=int(kwargs['batch_stations']),
		),
	)
	monkeypatch.setattr(pl, 'prepare_win32_ch_table_une', fake_prepare)
	monkeypatch.setattr(pl, 'iter_win32_station_windows', fake_iter)

	out_csv = tmp_path / 'pick.csv'
	stats = pl.pipeline_win32_eqt_pick_to_csv(
		cnt_paths=[Path('win_0301_202001010000_1m_aaaaaaaa.cnt')],
		ch_table=tmp_path / 'dummy.ch',
		out_csv=out_csv,
		eqt_weights='dummy',
		eqt_in_samples=10,
		eqt_overlap=5,
		eqt_batch_stations=4,
		target_fs_hz=1.0,
		det_threshold=0.3,
		p_threshold=0.1,
		s_threshold=0.1,
		min_pick_sep_samples=1,
	)

	df = pd.read_csv(out_csv)
	assert stats.picks_written == 2
	assert list(df.columns) == [
		'station_code',
		'Phase',
		'pick_time',
		'w_conf',
		'network_code',
	]
	assert df['pick_time'].str.endswith('+09:00').all()
	assert '2020-01-01T00:00:05.000+09:00' in df['pick_time'].tolist()
