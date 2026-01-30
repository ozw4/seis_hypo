# tests/test_hypo_synth_eval_station_subset.py
from __future__ import annotations

import numpy as np
import pytest

from hypo.synth_eval.station_subset import normalize_station_subset


def _default_codes() -> list[str]:
	return ['G0001', 'G0002', 'G0003', 'D0001', 'D0002', 'D0003', 'D0004']


def test_normalize_station_subset_all() -> None:
	station_subset = {'surface_indices': 'ALL', 'das_indices': 'all'}

	result = normalize_station_subset(station_subset, codes=_default_codes())

	np.testing.assert_array_equal(result, np.arange(7))


def test_normalize_station_subset_lists() -> None:
	station_subset = {'surface_indices': [0, 2], 'das_indices': [1, 3]}

	result = normalize_station_subset(station_subset, codes=_default_codes())

	np.testing.assert_array_equal(result, np.array([0, 2, 4, 6]))


def test_normalize_station_subset_das_only_all() -> None:
	station_subset = {'das_indices': 'all'}

	result = normalize_station_subset(station_subset, codes=_default_codes())

	np.testing.assert_array_equal(result, np.array([3, 4, 5, 6]))


@pytest.mark.parametrize(
	'station_subset',
	[
		{},
		{'surface_indices': [], 'das_indices': []},
	],
)
def test_normalize_station_subset_empty_raises(station_subset: dict) -> None:
	with pytest.raises(ValueError):
		normalize_station_subset(station_subset, codes=_default_codes())


def test_normalize_station_subset_too_few_points_raises() -> None:
	station_subset = {'surface_indices': [0, 1, 2]}

	with pytest.raises(ValueError):
		normalize_station_subset(station_subset, codes=_default_codes())


def test_normalize_station_subset_out_of_range_raises() -> None:
	station_subset = {'surface_indices': [3]}

	with pytest.raises(ValueError):
		normalize_station_subset(station_subset, codes=_default_codes())


def test_normalize_station_subset_duplicate_indices_raises() -> None:
	station_subset = {'surface_indices': [0, 0]}

	with pytest.raises(ValueError):
		normalize_station_subset(station_subset, codes=_default_codes())


@pytest.mark.parametrize('indices', [[True], [0, True]])
def test_normalize_station_subset_rejects_bool(indices: list[object]) -> None:
	station_subset = {'surface_indices': indices}

	with pytest.raises(ValueError):
		normalize_station_subset(station_subset, codes=_default_codes())


def test_normalize_station_subset_rejects_invalid_string() -> None:
	station_subset = {'surface_indices': 'everything'}

	with pytest.raises(ValueError):
		normalize_station_subset(station_subset, codes=_default_codes())


def test_normalize_station_subset_expected_len_mismatch_raises() -> None:
	station_subset = {'surface_indices': 'all', 'das_indices': 'all'}

	with pytest.raises(ValueError):
		normalize_station_subset(
			station_subset,
			codes=_default_codes(),
			expected_len=10,
		)
