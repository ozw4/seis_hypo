from __future__ import annotations

import pandas as pd
import pytest

from gamma_workflow.normalize import normalize_picks, normalize_stations


def test_normalize_picks_renames_columns_and_filters_types():
	df = pd.DataFrame(
		{
			'station_id': ['STA01', 'STA02', 'STA03'],
			'phase_time': [
				'2020-01-01T00:00:00Z',
				'2020-01-01T00:00:01Z',
				'2020-01-01T00:00:02Z',
			],
			'phase_type': ['p', 'S', 'X'],
			'phase_score': [0.9, 0.8, 0.1],
			'phase_amplitude': [1.0, 2.0, 3.0],
		}
	)

	out = normalize_picks(df)

	assert list(out.columns) == ['id', 'timestamp', 'type', 'prob', 'amp']
	assert out['id'].tolist() == ['STA01', 'STA02']
	assert out['type'].tolist() == ['P', 'S']


def test_normalize_stations_raises_when_required_columns_missing():
	df = pd.DataFrame({'station_id': ['STA01'], 'x_km': [1.0], 'y_km': [2.0]})

	with pytest.raises(ValueError, match='stations needs z\\(km\\) or z_depth_km'):
		normalize_stations(df)
