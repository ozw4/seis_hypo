# %%
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
from dateutil.relativedelta import relativedelta

from jma.chk_network_station import (
	export_channels_from_probe_ch_dirs,
	export_station_summary_from_channels,
)
from jma.chk_newwork_downloadable import probe_networks_by_get_continuous_waveform


def probe_networks_monthly_retry(
	*,
	network_info: dict[str, str],
	when: dt.datetime,
	download_folder: str | Path,
	max_tries: int = 24,
	span_min: int = 1,
	threads: int = 4,
	cleanup: bool = True,
	keep_cnt: bool = False,
) -> pd.DataFrame:
	"""失敗したネットワークだけ、whenを1か月ずつ「過去方向」にずらして再probeする。

	- 保存先は download_folder に固定（時期別に分けない）
	- 返り値: networkごとの最終結果（成功した試行の情報が残る）
	"""
	if not network_info:
		raise ValueError('network_info is empty')
	if max_tries <= 0:
		raise ValueError('max_tries must be > 0')

	download_folder = Path(download_folder)
	download_folder.mkdir(parents=True, exist_ok=True)

	remaining = dict(network_info)
	best: dict[str, dict[str, str]] = {}

	last_when_i: dt.datetime = when

	for i in range(max_tries):
		when_i = when - relativedelta(months=i)
		last_when_i = when_i

		df = probe_networks_by_get_continuous_waveform(
			network_info=remaining,
			when=when_i,
			base_outdir=download_folder,
			span_min=span_min,
			threads=threads,
			cleanup=cleanup,
			keep_cnt=keep_cnt,
		)

		for _, r in df.iterrows():
			code = str(r['network_code'])
			ok = str(r['ok']).lower() == 'true'
			if ok:
				best[code] = {
					'network_code': code,
					'network_name': str(r.get('network_name', '')),
					'ok': 'True',
					'when': when_i.strftime('%Y-%m-%d %H:%M:%S'),
					'outdir': str(r.get('outdir', '')),
					'cnt_path': str(r.get('cnt_path', '')),
					'ch_path': str(r.get('ch_path', '')),
					'error_type': '',
					'error_msg': '',
				}

		remaining = {k: v for k, v in remaining.items() if k not in best}
		if not remaining:
			break

	# 全試行失敗のネット
	for code, name in remaining.items():
		best[str(code)] = {
			'network_code': str(code),
			'network_name': str(name),
			'ok': 'False',
			'when': last_when_i.strftime('%Y-%m-%d %H:%M:%S'),
			'outdir': '',
			'cnt_path': '',
			'ch_path': '',
			'error_type': 'ProbeFailedAllTries',
			'error_msg': f'failed for all tries: {max_tries} (month step backward)',
		}

	out_df = (
		pd.DataFrame(
			list(best.values()),
			columns=[
				'network_code',
				'network_name',
				'ok',
				'when',
				'outdir',
				'cnt_path',
				'ch_path',
				'error_type',
				'error_msg',
			],
		)
		.sort_values(['ok', 'network_code'], ascending=[False, True])
		.reset_index(drop=True)
	)

	out_csv = download_folder / 'network_probe_results_final.csv'
	out_df.to_csv(out_csv, index=False, encoding='utf-8')
	print(f'[INFO] wrote: {out_csv}')

	return out_df


if __name__ == '__main__':
	probe_download = True
	# pasted.txt のこれをそのままコピペでOK
	start = dt.datetime(2004, 4, 1, 0, 0, 0)
	end = dt.datetime(2025, 12, 1, 0, 0, 0)

	NETWORK_INFO = {
		'0301': 'JMA Seismometer Network',
	}

	for when in pd.date_range(start=start, end=end, freq='MS'):
		print(f'=== Probing for {when.strftime("%Y-%m")} ===')
		download_folder = '/workspace/data/station/jma/newwork_test_downloads_jma'

		if probe_download:
			df = probe_networks_monthly_retry(
				network_info=NETWORK_INFO,
				when=when,
				download_folder=download_folder,
				max_tries=300,  # 4年分
				span_min=1,
				threads=4,
				cleanup=True,
				keep_cnt=False,
			)
			print(df)

	df_ch = export_channels_from_probe_ch_dirs(
		base_probe_dir=download_folder,
		out_csv=Path(download_folder) / 'channels_by_network.csv',
	)
	print(df_ch.head())

	df_sta = export_station_summary_from_channels(
		df_ch,
		out_csv=Path(download_folder) / 'stations_by_network.csv',
	)
	print(df_sta.head())
