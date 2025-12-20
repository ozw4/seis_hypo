# %%
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from jma.download import create_hinet_client, download_win_for_stations


def probe_stations_from_station_csv(
	client,
	*,
	station_csv_path: str | Path,
	network_codes: list[str],
	when: dt.datetime,
	outdir: str | Path,
	span_min: int = 1,
	threads: int = 4,
	max_trials_per_network: int = 500,
	max_success_per_network: int = 10,
	start_row: int = 0,
) -> pd.DataFrame:
	"""station.csv の station_code を先頭から順に、各network_codeへ1局ずつprobeする。

	- 半径フィルタなし（CSV全体を候補にする）
	- 1 station / 1 network / 1 minute で download を試す
	- 失敗は検査目的なので握って継続（※フォールバック扱い：警告を出す）
	"""
	station_csv_path = Path(station_csv_path)
	if not station_csv_path.is_file():
		raise FileNotFoundError(f'station_csv_path not found: {station_csv_path}')

	if not network_codes:
		raise ValueError('network_codes is empty')

	df = pd.read_csv(station_csv_path)
	if df.empty:
		raise ValueError(f'station csv is empty: {station_csv_path}')

	if 'station_code' not in df.columns:
		raise ValueError(
			f'station csv missing station_code column: have={df.columns.tolist()}'
		)

	stations_all = df['station_code'].astype(str).tolist()
	if start_row < 0 or start_row >= len(stations_all):
		raise ValueError(f'start_row out of range: {start_row} (n={len(stations_all)})')

	outdir = Path(outdir)
	outdir.mkdir(parents=True, exist_ok=True)

	rows: list[dict[str, str]] = []

	for code in network_codes:
		code = str(code)
		net_dir = outdir / code
		net_dir.mkdir(parents=True, exist_ok=True)

		success = 0
		trials = 0

		for sta in stations_all[start_row:]:
			if trials >= int(max_trials_per_network):
				break
			if success >= int(max_success_per_network):
				break

			trials += 1
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
				# [WARN] 検査目的の継続（フォールバック扱い）
				print(
					f'[WARN] probe failed: network={code} station={sta} err={type(e).__name__}: {e}'
				)
				continue

			rows.append(
				{
					'network_code': code,
					'station_success': sta,
					'trial_index': str(trials),
					'cnt_path': str(cnt_path),
					'ch_path': str(ch_path),
				}
			)
			success += 1

		rows.append(
			{
				'network_code': code,
				'station_success': '',
				'trial_index': str(trials),
				'cnt_path': '',
				'ch_path': '',
				'summary': f'success={success} trials={trials}',
			}
		)

	return pd.DataFrame(rows)


if __name__ == '__main__':
	client = create_hinet_client()

	# 観測が確実にありそうな時刻にする（JSTの naive を渡して良いならそのまま）
	when = dt.datetime(2025, 1, 1, 0, 0, 0)

	NETWORK_CODES = [
		'0201',
	]

	df = probe_stations_from_station_csv(
		client,
		station_csv_path='/workspace/data/station/station.csv',
		network_codes=NETWORK_CODES,
		when=when,
		outdir='network_test_downloads',
		span_min=1,
		threads=4,
		max_trials_per_network=10,
		max_success_per_network=5,
		start_row=0,
	)

	print(df)
	df.to_csv('network_probe_results.csv', index=False, encoding='utf-8')
	print('wrote network_probe_results.csv')
