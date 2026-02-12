# %%
from __future__ import annotations

from pathlib import Path

from jma.monthly_presence_selection import (
	stations_within_radius_from_monthly_presence,
	write_station_lists,
)

if __name__ == '__main__':
	monthly_presence_csv = '/workspace/proc/prepare_data/jma/stationcode_match/v1/snapshots/monthly/monthly_presence.csv'

	# 対象期間（ざっくり月単位で判定するので、ここは期間が2009-12を含めばOK）
	start_time = '2009-12-17 00:00:00'
	end_time = '2009-12-20 23:59:59'

	# 中心点（well_coord）
	center_lat = 34.97
	center_lon = 139.13

	# 条件
	radius_km = 50.0
	target_networks = {'0101', '0203', '0207', '0301'}

	out_dir = Path('./profile/stations47')
	out_dir.mkdir(parents=True, exist_ok=True)

	# 2009-12に観測中の局から、50 km以内だけ拾う（dist_km付きで返る）
	df = stations_within_radius_from_monthly_presence(
		monthly_presence_csv,
		start_time=start_time,
		end_time=end_time,
		center_lat=center_lat,
		center_lon=center_lon,
		radius_km=radius_km,
	)

	df = df[df['network_code'].isin(target_networks)].copy()
	df = df.sort_values(['network_code', 'dist_km', 'station']).reset_index(drop=True)

	if len(df) != 47:
		raise RuntimeError(f'expected 47 stations, got {len(df)}')

	df.to_csv(out_dir / 'stations_47.csv', index=False, encoding='utf-8')

	(out_dir / 'stations_47.tsv').write_text(
		'\n'.join(f'{r.network_code}\t{r.station}' for r in df.itertuples(index=False))
		+ '\n',
		encoding='utf-8',
	)

	write_station_lists(df, out_dir=out_dir, pick_n=None)

# %%
