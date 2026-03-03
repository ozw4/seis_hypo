from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hypo.synth_eval.builders import build_station_df, build_truth_df


def test_build_station_df_elevation_from_depth_positive() -> None:
	recv_xyz_m = np.zeros((9, 3), dtype=float)
	z_m = np.array(
		[10.4, 10.6, -10.4, -10.6, 0.0, 1.2, -1.2, 99.9, -99.9],
		dtype=float,
	)
	recv_xyz_m[:, 2] = z_m
	receiver_indices = np.arange(0, 9, dtype=int)
	station_codes_all = np.array([f'G{i:04d}' for i in range(1, 10)], dtype=str)

	df = build_station_df(
		recv_xyz_m,
		receiver_indices,
		station_codes_all,
		35.0,
		140.0,
		z_is_depth_positive=True,
	)

	expected = (-z_m).round().astype(int)
	got = df['Elevation_m'].to_numpy()
	assert np.array_equal(got, expected)


def test_build_station_df_elevation_from_up_positive() -> None:
	recv_xyz_m = np.zeros((9, 3), dtype=float)
	z_m = np.array(
		[10.4, 10.6, -10.4, -10.6, 0.0, 1.2, -1.2, 99.9, -99.9],
		dtype=float,
	)
	recv_xyz_m[:, 2] = z_m
	receiver_indices = np.arange(0, 9, dtype=int)
	station_codes_all = np.array([f'G{i:04d}' for i in range(1, 10)], dtype=str)

	df = build_station_df(
		recv_xyz_m,
		receiver_indices,
		station_codes_all,
		35.0,
		140.0,
		z_is_depth_positive=False,
	)

	expected = (z_m).round().astype(int)
	got = df['Elevation_m'].to_numpy()
	assert np.array_equal(got, expected)


def _write_index_csv(path: Path) -> None:
	rows = []
	eid = 0
	for k in range(4):
		for j in range(4):
			for i in range(4):
				rows.append(
					{
						'event_id': f'ev_{eid:06d}',
						'x_m': float(i),
						'y_m': float(j),
						'z_m': float(k),
					}
				)
				eid += 1
	pd.DataFrame(rows).to_csv(path, index=False)


def test_build_truth_df_stride_subsample_and_max_events_order(tmp_path: Path) -> None:
	index_csv = tmp_path / 'index.csv'
	_write_index_csv(index_csv)

	df = build_truth_df(
		index_csv,
		35.0,
		140.0,
		pd.Timestamp('2020-01-01T00:00:00Z'),
		0.1,
		2,
		event_stride_ijk=[2, 2, 2],
	)

	assert len(df) == 2
	assert df['event_id'].tolist() == [0, 2]


def test_build_truth_df_z_range_filter_bounds(tmp_path: Path) -> None:
	index_csv = tmp_path / 'index.csv'
	_write_index_csv(index_csv)

	df = build_truth_df(
		index_csv,
		35.0,
		140.0,
		pd.Timestamp('2020-01-01T00:00:00Z'),
		0.1,
		0,
		event_z_range_m=[2.0, None],
	)
	assert len(df) == 32
	assert df['z_m_true'].min() >= 2.0

	df = build_truth_df(
		index_csv,
		35.0,
		140.0,
		pd.Timestamp('2020-01-01T00:00:00Z'),
		0.1,
		0,
		event_z_range_m=[None, 1.0],
	)
	assert len(df) == 32
	assert df['z_m_true'].max() <= 1.0


def test_build_truth_df_order_z_filter_then_subsample_then_max_events(
	tmp_path: Path,
) -> None:
	index_csv = tmp_path / 'index.csv'
	_write_index_csv(index_csv)

	df = build_truth_df(
		index_csv,
		35.0,
		140.0,
		pd.Timestamp('2020-01-01T00:00:00Z'),
		0.1,
		3,
		event_z_range_m=[2.0, None],
		event_stride_ijk=[2, 2, 2],
	)

	assert len(df) == 3
	assert df['event_id'].tolist() == [32, 34, 40]


def test_build_truth_df_keep_n_xyz_subsample(tmp_path: Path) -> None:
	index_csv = tmp_path / 'index.csv'
	_write_index_csv(index_csv)

	df = build_truth_df(
		index_csv,
		35.0,
		140.0,
		pd.Timestamp('2020-01-01T00:00:00Z'),
		0.1,
		0,
		event_keep_n_xyz=[2, 2, 2],
	)

	assert len(df) == 8
	assert set(np.rint(df['x_m_true'].to_numpy()).astype(int).tolist()) == {0, 3}
	assert set(np.rint(df['y_m_true'].to_numpy()).astype(int).tolist()) == {0, 3}
	assert set(np.rint(df['z_m_true'].to_numpy()).astype(int).tolist()) == {0, 3}


def test_build_truth_df_rejects_invalid_event_subsample(tmp_path: Path) -> None:
	index_csv = tmp_path / 'index.csv'
	_write_index_csv(index_csv)

	with pytest.raises(ValueError, match='cannot be specified at the same time'):
		build_truth_df(
			index_csv,
			35.0,
			140.0,
			pd.Timestamp('2020-01-01T00:00:00Z'),
			0.1,
			0,
			event_stride_ijk=[2, 2, 2],
			event_keep_n_xyz=[2, 2, 2],
		)

	with pytest.raises(ValueError, match='>= 1'):
		build_truth_df(
			index_csv,
			35.0,
			140.0,
			pd.Timestamp('2020-01-01T00:00:00Z'),
			0.1,
			0,
			event_stride_ijk=[1, 0, 1],
		)

	with pytest.raises(ValueError, match='exceeds unique grid counts'):
		build_truth_df(
			index_csv,
			35.0,
			140.0,
			pd.Timestamp('2020-01-01T00:00:00Z'),
			0.1,
			0,
			event_keep_n_xyz=[5, 1, 1],
		)

	with pytest.raises(ValueError, match='zmin_m must be <= zmax_m'):
		build_truth_df(
			index_csv,
			35.0,
			140.0,
			pd.Timestamp('2020-01-01T00:00:00Z'),
			0.1,
			0,
			event_z_range_m=[3.0, 1.0],
		)
