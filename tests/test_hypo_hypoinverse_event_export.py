from __future__ import annotations

import pandas as pd

from hypo.hypoinverse_event_export import build_hypoinverse_event_export_df


def test_build_hypoinverse_event_export_df_merges_initial_and_hypo_rows() -> None:
	initial_df = pd.DataFrame(
		{
			'event_id': [11, 22],
			'origin_time': ['2026-01-01 00:00:00', '2026-01-01 00:01:00'],
			'latitude_deg': [35.1, 35.2],
			'longitude_deg': [140.1, 140.2],
			'depth_km': [5.0, 6.0],
		}
	)
	hyp_df = pd.DataFrame(
		{
			'sequence_no_prt': [7, 9],
			'id_no_prt': [22, 11],
			'seq': [1, 2],
			'origin_time_hyp': pd.to_datetime(
				['2026-01-01 00:01:02', '2026-01-01 00:00:03']
			),
			'lat_deg_hyp': [35.21, 35.11],
			'lon_deg_hyp': [140.21, 140.11],
			'depth_km_hyp': [6.2, 5.2],
			'ERH': [0.2, 0.3],
			'ERZ': [0.4, 0.5],
			'NSTA': [10, 11],
		}
	)

	out = build_hypoinverse_event_export_df(initial_df, hyp_df)

	assert out['event_id'].tolist() == [22, 11]
	assert out['sequence_no_prt'].tolist() == [7, 9]
	assert out['seq'].tolist() == [1, 2]
	assert out['origin_time_init'].tolist() == [
		'2026-01-01 00:01:00',
		'2026-01-01 00:00:00',
	]
	assert out['lat_deg_init'].tolist() == [35.2, 35.1]
	assert out['lon_deg_init'].tolist() == [140.2, 140.1]
	assert out['depth_km_init'].tolist() == [6.0, 5.0]
	assert out['lat_deg_hyp'].tolist() == [35.21, 35.11]
