from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hypo.synth_eval.event_subsample import (
	event_subsample_mask_from_xyz,
	parse_event_subsample_3ints,
	validate_event_subsample_config,
)


def test_parse_event_subsample_3ints_requires_three_positive_ints() -> None:
	assert parse_event_subsample_3ints([2, 3, 4], key='stride_ijk', min_value=1) == (
		2,
		3,
		4,
	)

	with pytest.raises(ValueError, match='exactly 3'):
		parse_event_subsample_3ints([2, 3], key='stride_ijk', min_value=1)

	with pytest.raises(ValueError, match='>= 1'):
		parse_event_subsample_3ints([2, 0, 4], key='stride_ijk', min_value=1)

	with pytest.raises(ValueError, match='>= 1'):
		parse_event_subsample_3ints([2, True, 4], key='stride_ijk', min_value=1)


def test_validate_event_subsample_config_enforces_stride_keep_exclusivity() -> None:
	with pytest.raises(ValueError, match='cannot be specified at the same time'):
		validate_event_subsample_config(
			{'stride_ijk': [2, 2, 2], 'keep_n_xyz': [1, 1, 1]}
		)

	with pytest.raises(ValueError, match='keep_n_xyz is not supported'):
		validate_event_subsample_config(
			{'keep_n_xyz': [1, 1, 1]},
			field='uncertainty_plot.event_subsample',
			allow_keep_n_xyz=False,
		)

	assert validate_event_subsample_config({'stride_ijk': [2, 2, 1]}) == {
		'stride_ijk': [2, 2, 1]
	}


def test_event_subsample_mask_from_xyz_stride_and_keep_match_expected_grid() -> None:
	df = pd.DataFrame(
		{
			'x_m': [0.0, 0.0, 3.0, 3.0, 0.0, 0.0, 3.0, 3.0],
			'y_m': [0.0, 3.0, 0.0, 3.0, 0.0, 3.0, 0.0, 3.0],
			'z_m': [0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 3.0],
		}
	)

	mask_stride = event_subsample_mask_from_xyz(
		df,
		stride_ijk=(2, 1, 1),
		keep_n_xyz=None,
	)
	assert np.array_equal(
		mask_stride,
		np.array([True, True, False, False, True, True, False, False]),
	)

	mask_keep = event_subsample_mask_from_xyz(
		df,
		stride_ijk=None,
		keep_n_xyz=(1, 2, 2),
	)
	assert np.array_equal(
		mask_keep,
		np.array([True, True, False, False, True, True, False, False]),
	)
