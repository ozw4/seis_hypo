from __future__ import annotations

import numpy as np

from hypo.synth_eval.builders import build_station_df


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
