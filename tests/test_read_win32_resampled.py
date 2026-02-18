import numpy as np
import pandas as pd
import pytest

import jma.win32_reader as wr


def test_read_win32_resampled_groups_by_fs(monkeypatch, tmp_path):
	fp = tmp_path / 'x.cnt'
	fp.write_bytes(b'\x00\x00\x00\x00')  # 存在チェックだけ通す

	df = pd.DataFrame(
		{
			'ch_hex': ['0003', '0004'],
			'ch_int': [3, 4],
			'conv_coeff': [1.0, 1.0],
			'station': ['N.TEST', 'N.TEST'],
			'component': ['U', 'U'],
		}
	)

	monkeypatch.setattr(
		wr,
		'scan_channel_sampling_rate_map_win32',
		lambda *args, **kwargs: {3: 50, 4: 100},
	)

	def fake_read_win32(
		file_path,
		channel_table,
		base_sampling_rate_HZ,
		duration_SECOND,
		channels_hex=None,
		station=None,
		components=None,
	):
		n_ch = len(channel_table)
		nt = int(duration_SECOND) * int(base_sampling_rate_HZ)
		out = np.zeros((n_ch, nt), dtype=np.float32)
		out[:] = float(base_sampling_rate_HZ)
		return out

	monkeypatch.setattr(wr, 'read_win32', fake_read_win32)

	def fake_resample_window_poly(y, fs_in, fs_out, out_len):
		out = np.zeros((y.shape[0], out_len), dtype=np.float32)
		out[:] = float(fs_in)
		return out

	monkeypatch.setattr(wr, 'resample_window_poly', fake_resample_window_poly)

	out = wr.read_win32_resampled(
		fp,
		df,
		target_sampling_rate_HZ=20,
		duration_SECOND=2,
		station='N.TEST',
		components=['U'],
	)

	assert out.shape == (2, 40)
	assert out[0, 0] == 50.0
	assert out[1, 0] == 100.0


def test_read_win32_resampled_missing_channel_zero_policy(monkeypatch, tmp_path):
	fp = tmp_path / 'x.cnt'
	fp.write_bytes(b'\x00\x00\x00\x00')

	df = pd.DataFrame(
		{
			'ch_hex': ['0003', '0004'],
			'ch_int': [3, 4],
			'conv_coeff': [1.0, 1.0],
			'station': ['N.TEST', 'N.TEST'],
			'component': ['U', 'N'],
		}
	)

	monkeypatch.setattr(
		wr,
		'scan_channel_sampling_rate_map_win32',
		lambda *args, **kwargs: {3: 50},
	)

	def fake_read_win32(
		file_path,
		channel_table,
		base_sampling_rate_HZ,
		duration_SECOND,
		channels_hex=None,
		station=None,
		components=None,
	):
		n_ch = len(channel_table)
		nt = int(duration_SECOND) * int(base_sampling_rate_HZ)
		out = np.zeros((n_ch, nt), dtype=np.float32)
		out[:] = float(base_sampling_rate_HZ)
		return out

	monkeypatch.setattr(wr, 'read_win32', fake_read_win32)

	def fake_resample_window_poly(y, fs_in, fs_out, out_len):
		out = np.zeros((y.shape[0], out_len), dtype=np.float32)
		out[:] = float(fs_in)
		return out

	monkeypatch.setattr(wr, 'resample_window_poly', fake_resample_window_poly)

	out = wr.read_win32_resampled(
		fp,
		df,
		target_sampling_rate_HZ=20,
		duration_SECOND=2,
		missing_channel_policy='zero',
	)

	assert out.shape == (2, 40)
	assert np.allclose(out[0, :], 50.0)
	assert np.allclose(out[1, :], 0.0)


def test_read_win32_resampled_missing_channel_drop_policy(monkeypatch, tmp_path):
	fp = tmp_path / 'x.cnt'
	fp.write_bytes(b'\x00\x00\x00\x00')

	df = pd.DataFrame(
		{
			'ch_hex': ['0003', '0004'],
			'ch_int': [3, 4],
			'conv_coeff': [1.0, 1.0],
			'station': ['N.TEST', 'N.TEST'],
			'component': ['U', 'N'],
		}
	)

	monkeypatch.setattr(
		wr,
		'scan_channel_sampling_rate_map_win32',
		lambda *args, **kwargs: {3: 50},
	)

	def fake_read_win32(
		file_path,
		channel_table,
		base_sampling_rate_HZ,
		duration_SECOND,
		channels_hex=None,
		station=None,
		components=None,
	):
		n_ch = len(channel_table)
		nt = int(duration_SECOND) * int(base_sampling_rate_HZ)
		out = np.zeros((n_ch, nt), dtype=np.float32)
		out[:] = float(base_sampling_rate_HZ)
		return out

	monkeypatch.setattr(wr, 'read_win32', fake_read_win32)

	def fake_resample_window_poly(y, fs_in, fs_out, out_len):
		out = np.zeros((y.shape[0], out_len), dtype=np.float32)
		out[:] = float(fs_in)
		return out

	monkeypatch.setattr(wr, 'resample_window_poly', fake_resample_window_poly)

	out = wr.read_win32_resampled(
		fp,
		df,
		target_sampling_rate_HZ=20,
		duration_SECOND=2,
		missing_channel_policy='drop',
	)

	assert out.shape == (2, 40)
	assert np.allclose(out[0, :], 50.0)
	assert np.allclose(out[1, :], 0.0)


def test_read_win32_resampled_missing_channel_policy_validation(tmp_path):
	fp = tmp_path / 'x.cnt'
	fp.write_bytes(b'\x00\x00\x00\x00')

	df = pd.DataFrame(
		{
			'ch_hex': ['0003'],
			'ch_int': [3],
			'conv_coeff': [1.0],
			'station': ['N.TEST'],
			'component': ['U'],
		}
	)

	with pytest.raises(ValueError, match='missing_channel_policy'):
		wr.read_win32_resampled(
			fp,
			df,
			target_sampling_rate_HZ=20,
			duration_SECOND=2,
			missing_channel_policy='bad_policy',
		)
