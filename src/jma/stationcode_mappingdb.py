# src/jma/stationcode_mappingdb.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from jma.stationcode_common import normalize_code


@dataclass(frozen=True)
class MappingDB:
	report: pd.DataFrame
	near0: pd.DataFrame


def load_mapping_db(mapping_report_csv: str | Path, near0_csv: str | Path) -> MappingDB:
	rp = Path(mapping_report_csv)
	if not rp.is_file():
		raise FileNotFoundError(rp)
	report = pd.read_csv(rp, low_memory=False)

	req = {'mea_norm', 'match_status', 'all_hit_stations_norm', 'all_hit_rules'}
	if not req.issubset(report.columns):
		raise ValueError(
			f'mapping_report missing columns: {sorted(req - set(report.columns))}'
		)

	if report['mea_norm'].isna().any():
		raise ValueError('mapping_report has NaN mea_norm')

	report = report.copy()
	report['mea_norm'] = report['mea_norm'].astype(str).map(normalize_code)
	report = report.drop_duplicates(subset=['mea_norm']).set_index(
		'mea_norm', drop=True
	)

	np0 = Path(near0_csv)
	if np0.is_file():
		near0 = pd.read_csv(np0, low_memory=False)
		nreq = {
			'mea_norm',
			'suggest_ch_station_norm',
			'nearest_distance_km',
			'overlap_first',
			'overlap_last',
		}
		if not nreq.issubset(near0.columns):
			raise ValueError(
				f'near0_suggestions missing columns: {sorted(nreq - set(near0.columns))}'
			)

		near0 = near0.copy()
		near0['mea_norm'] = near0['mea_norm'].astype(str).map(normalize_code)
		near0['suggest_ch_station_norm'] = (
			near0['suggest_ch_station_norm'].astype(str).map(normalize_code)
		)
	else:
		near0 = pd.DataFrame(
			columns=[
				'mea_norm',
				'suggest_ch_station_norm',
				'nearest_distance_km',
				'overlap_first',
				'overlap_last',
			]
		)

	return MappingDB(report=report, near0=near0)
