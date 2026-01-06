# %%
from __future__ import annotations

import os
from pathlib import Path

from jma.download import create_hinet_client
from jma.win32_reader import get_evt_info

# =========================
# 設定（ここを直書きでOK）
# =========================

# 取得対象の「イベント発生時刻の範囲」（JST）
EVENTS_START_JST = '202201010000'
EVENTS_END_JST = '202301010000'

# イベントの絞り込み（必要なら）
REGION = '00'  # 00: 日本全域 :contentReference[oaicite:1]{index=1}
MIN_MAG = 1.0
MAX_MAG = 99.0


# 出力先（この中に zip が落ちる）
OUTDIR = Path('/workspace/data/waveform/jma/event').resolve()


def main() -> None:
	OUTDIR.mkdir(parents=True, exist_ok=True)

	client = create_hinet_client()

	# get_event_waveform は「カレントディレクトリ」に落ちる想定なので移動
	os.chdir(OUTDIR)

	before = {p.name for p in OUTDIR.glob('*')}

	client.get_event_waveform(
		EVENTS_START_JST,
		EVENTS_END_JST,
		region=REGION,
		minmagnitude=MIN_MAG,
		maxmagnitude=MAX_MAG,
	)

	after = {p.name for p in OUTDIR.glob('*')}
	new_zips = sorted(after - before)

	if not new_zips:
		raise RuntimeError(
			'新しい file が見つかりませんでした（取得結果 0 の可能性もあります）。'
		)

	for zip_name in new_zips:
		zpath = OUTDIR / zip_name
		event_dir = OUTDIR / zpath.stem
		if event_dir.exists():
			print(
				f'イベントディレクトリ {event_dir} は既に存在します。スキップします。'
			)
			continue
		event_dir.mkdir(exist_ok=True)

		evt_files = sorted(event_dir.glob('*.evt'))
		ch_files = sorted(event_dir.glob('*.ch'))
		if not evt_files or not ch_files:
			continue

		for evt_path, ch_path in zip(evt_files, ch_files):
			info = get_evt_info(evt_path, scan_rate_blocks=1000)

			print(f'start_time          : {info.start_time}')
			print(f'end_time_exclusive  : {info.end_time_exclusive}')
			print(f'n_second_blocks     : {info.n_second_blocks}')
			print(f'span_seconds        : {info.span_seconds}')
			print(f'missing_seconds_est : {info.missing_seconds_est}')
			print(f'timestamp_anomalies : {info.timestamp_anomalies}')
			print(f'sampling_rates_hz   : {info.sampling_rates_hz}')
			print(f'base_sampling_rate  : {info.base_sampling_rate_hz}')
			print(f'Processing event file: {evt_path.name}')
			# win32_data = read_win32(evt_path, channel_table=ch_path)
			# print(win32_data.shape)


if __name__ == '__main__':
	main()
