# %%
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from jma.download import create_hinet_client, download_win_for_stations
from jma.station_reader import stations_within_radius_from_jma_compact


def check_network_downloadable_by_probe(
	client,
	*,
	when: dt.datetime,
	outdir: str | Path,
	network_codes: list[str],
	station_candidates: list[str],
	span_min: int = 1,
	threads: int = 4,
	max_station_trials_per_network: int = 30,
) -> pd.DataFrame:
	"""get_station_list() を使わず、station_candidates を順に試して network が落とせるか調べる。

	- 各networkにつき station を1局ずつ試す（最大 max_station_trials_per_network）
	- 成功した最初のstationでOK判定
	- どのstationでも成功しなければ FAIL として記録（例外は握って継続）
	"""
	if not network_codes:
		raise ValueError('network_codes is empty')
	if not station_candidates:
		raise ValueError('station_candidates is empty')

	outdir = Path(outdir)
	outdir.mkdir(parents=True, exist_ok=True)

	rows: list[dict[str, str]] = []
	for code in network_codes:
		code = str(code)
		net_dir = outdir / code
		net_dir.mkdir(parents=True, exist_ok=True)

		ok = False
		last_err = None
		tried = 0

		for sta in station_candidates[: int(max_station_trials_per_network)]:
			tried += 1
			sta = str(sta)

			try:
				cnt_path, ch_path = download_win_for_stations(
					client,
					stations=[sta],
					when=when,
					network_code=code,
					span_min=int(span_min),
					outdir=net_dir,
					threads=int(threads),
					cleanup=True,
					clear_selection=True,
					skip_if_exists=False,
				)
			except Exception as e:
				last_err = e
				continue

			rows.append(
				{
					'network_code': code,
					'status': 'OK',
					'station_success': sta,
					'n_station_trials': str(tried),
					'cnt_path': str(cnt_path),
					'ch_path': str(ch_path),
				}
			)
			ok = True
			break

		if not ok:
			msg = repr(last_err) if last_err is not None else 'no error captured'
			rows.append(
				{
					'network_code': code,
					'status': 'FAIL',
					'station_success': '',
					'n_station_trials': str(tried),
					'cnt_path': '',
					'ch_path': '',
					'last_error': msg,
				}
			)

	return pd.DataFrame(rows)


def build_station_candidates_from_jma_compact(
	*,
	jma_stations_path: str | Path,
	site_lat: float,
	site_lon: float,
	radius_km: float,
) -> list[str]:
	"""JMA compact stations から半径内の station_code を作る（候補局）。"""
	return stations_within_radius_from_jma_compact(
		lat=float(site_lat),
		lon=float(site_lon),
		radius_km=float(radius_km),
		jma_compact_path=jma_stations_path,
		output='list',
	)


if __name__ == '__main__':
	NETWORK_CODES = [
		'0101',
		'0103',
		'0103A',
		'0120',
		'0120A',
		'0120B',
		'0120C',
		'0131',
		'0301',
		'0701',
		'0702',
		'0703',
		'0705',
		'0801',
	]

	# clientの作り方は既存の run_prepare_event.py と同じやつを使ってください
	client = create_hinet_client()

	station_candidates = build_station_candidates_from_jma_compact(
		jma_stations_path='/workspace/data/station/stations',
		site_lat=35.4,
		site_lon=140.2,
		radius_km=80.0,
	)
	when = dt.datetime(2025, 1, 1, 0, 0, 0)
	df = check_network_downloadable_by_probe(
		client=client,
		when=when,
		outdir='network_test_downloads',
		network_codes=NETWORK_CODES,
		station_candidates=station_candidates,
		max_station_trials_per_network=30,
	)
	print(df)
