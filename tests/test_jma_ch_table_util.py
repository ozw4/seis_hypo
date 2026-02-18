from __future__ import annotations

import pandas as pd
import pytest

from jma.ch_table_util import normalize_ch_table_components_to_une


def _partial_component_table() -> pd.DataFrame:
	return pd.DataFrame(
		{
			'station': ['STA1', 'STA2', 'STA2', 'STA3'],
			'component': ['U', 'N', 'E', 'Q'],
		}
	)


def test_normalize_ch_table_components_to_une_strict_by_default():
	with pytest.raises(ValueError, match='stations missing U/N/E after normalization'):
		normalize_ch_table_components_to_une(_partial_component_table())


def test_normalize_ch_table_components_to_une_allow_partial():
	out = normalize_ch_table_components_to_une(
		_partial_component_table(),
		require_full_une=False,
	)
	pairs = out[['station', 'component']].values.tolist()
	assert pairs == [['STA1', 'U'], ['STA2', 'N'], ['STA2', 'E']]
	assert 'STA3' not in out['station'].tolist()
