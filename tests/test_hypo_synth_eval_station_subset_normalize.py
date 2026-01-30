# tests/test_hypo_synth_eval_station_subset_normalize.py
from __future__ import annotations

import numpy as np
import pytest

from hypo.synth_eval.station_subset import normalize_station_subset


def test_normalize_station_subset_all_all_returns_all_sorted_unique() -> None:
	codes = np.array(['G0000', 'd0001', 'X0002', 'D0003'], dtype=object)
	ss = {'surface_indices': 'all', 'das_indices': 'ALL'}
	out = normalize_station_subset(ss, codes=codes, min_points=1)
	assert out.tolist() == [0, 1, 2, 3]


def test_normalize_station_subset_whitespace_all_is_accepted() -> None:
	codes = np.array(['G0', 'D1', 'G2', 'D3'], dtype=object)
	ss = {'surface_indices': '  all  '}
	out = normalize_station_subset(ss, codes=codes, min_points=1)
	assert out.tolist() == [0, 2]


def test_normalize_station_subset_device_indices_map_to_global() -> None:
	# surface_global = [0, 2], das_global = [1, 3]
	codes = np.array(['G0', 'D1', 'G2', 'D3'], dtype=object)

	ss = {'surface_indices': [1]}  # surface内 index=1 -> global=2
	out = normalize_station_subset(ss, codes=codes, min_points=1)
	assert out.tolist() == [2]

	ss = {'das_indices': [0, 1]}  # das内 index=0,1 -> global=1,3
	out = normalize_station_subset(ss, codes=codes, min_points=1)
	assert out.tolist() == [1, 3]


@pytest.mark.parametrize(
	'station_subset',
	[
		{},  # keysなし => 空選択
		{'surface_indices': [], 'das_indices': []},  # 空リスト => 空選択
	],
)
def test_normalize_station_subset_rejects_empty_selection(station_subset) -> None:
	codes = np.array(['G0', 'D1', 'G2', 'D3'], dtype=object)
	with pytest.raises(ValueError):
		normalize_station_subset(station_subset, codes=codes)


@pytest.mark.parametrize(
	'station_subset',
	[
		{'surface_indices': [-1]},  # negative
		{'surface_indices': [2]},  # surface_globalは2要素なので out-of-range
		{'das_indices': [2]},  # das_globalも2要素なので out-of-range
	],
)
def test_normalize_station_subset_rejects_out_of_range_device_indices(
	station_subset,
) -> None:
	codes = np.array(['G0', 'D1', 'G2', 'D3'], dtype=object)
	with pytest.raises(ValueError):
		normalize_station_subset(station_subset, codes=codes, min_points=1)


def test_normalize_station_subset_rejects_codes_not_1d() -> None:
	codes = np.array([['G0', 'D1']], dtype=object)  # ndim=2
	ss = {'surface_indices': 'all'}
	with pytest.raises(ValueError):
		normalize_station_subset(ss, codes=codes, min_points=1)


def test_normalize_station_subset_enforces_min_points_default_4() -> None:
	# surface_global=[0], das_global=[1,2,3]
	codes = np.array(['G0', 'D1', 'D2', 'D3'], dtype=object)

	# 3点選択 => min_points=4(default)で落ちる
	ss = {'surface_indices': 'all', 'das_indices': [0, 1]}
	with pytest.raises(ValueError):
		normalize_station_subset(ss, codes=codes)

	# 4点選択 => 通る
	ss = {'surface_indices': 'all', 'das_indices': 'all'}
	out = normalize_station_subset(ss, codes=codes)
	assert out.tolist() == [0, 1, 2, 3]
