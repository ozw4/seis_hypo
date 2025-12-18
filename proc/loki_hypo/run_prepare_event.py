# %%
#!/usr/bin/env python
# proc/loki_hypo/run_prepare_events.py
#
# 指定観測期間・観測範囲・マグニチュード条件を満たすイベントについて、
# Hi-net WIN32 から ObsPy Stream をまとめて準備するバッチ。
#
# 設定は YAML から common.load_config.load_config() で読み込む。
#


from __future__ import annotations

from netrc import netrc
from pathlib import Path

import pandas as pd
from HinetPy import Client

from catalog.selection import extract_events_in_region
from common.config import PrepareEventsConfig
from common.load_config import load_config
from jma.download import download_win_for_event
from jma.station_reader import stations_within_radius

# あなたの環境に合わせて書き換えればOK
YAML_PATH = Path('/workspace/data/config/prepare_events.yaml')
PRESET = 'mobara'


def create_hinet_client() -> Client:
	login, _, password = netrc().authenticators('hinet')
	return Client(login, password)


def select_stations_for_site(cfg: PrepareEventsConfig) -> list[str]:
	station_list = stations_within_radius(
		lat=cfg.station_site_lat,
		lon=cfg.station_site_lon,
		radius_km=cfg.station_radius_km,
	)
	if not station_list:
		msg = 'stations_within_radius で観測点が1つも選ばれていません。'
		raise ValueError(msg)
	return station_list


def filter_events_with_existing_logic(
	df: pd.DataFrame,
	cfg: PrepareEventsConfig,
) -> pd.DataFrame:
	"""既存の catalog.selection.extract_events_in_region を使ってイベントを絞る."""
	epic_sub, _ = extract_events_in_region(
		epic_df=df,
		meas_df=None,
		start_time=cfg.start_time,
		end_time=cfg.end_time,
		mag_min=cfg.min_mag,
		mag_max=cfg.max_mag,
		center_lat=cfg.station_site_lat,
		center_lon=cfg.station_site_lon,
		radius_km=cfg.event_radius_km,
	)
	if cfg.max_events is not None and cfg.max_events > 0:
		epic_sub = epic_sub.sort_values('origin_time').head(cfg.max_events)
	return epic_sub


def main() -> None:
	cfg = load_config(PrepareEventsConfig, YAML_PATH, PRESET)

	if not cfg.catalog_csv.is_file():
		msg = f'catalog CSV not found: {cfg.catalog_csv}'
		raise FileNotFoundError(msg)

	df_catalog = pd.read_csv(cfg.catalog_csv)

	# 既存の selection ロジックをそのまま利用してイベントを選ぶ
	df_events = filter_events_with_existing_logic(df_catalog, cfg)
	print(f'selected events: {len(df_events)}')

	# 観測点リストも既存の stations_within_radius を使う
	station_list = select_stations_for_site(cfg)
	print(f'selected stations: {len(station_list)}')

	client = create_hinet_client()

	base_input_dir = cfg.base_input_dir
	base_input_dir.mkdir(parents=True, exist_ok=True)

	for _, event_row in df_events.iterrows():
		event_id = int(event_row['event_id'])
		print(f'prepare event_id={event_id}')

		event_dir, cnt_paths, ch_path = download_win_for_event(
			client,
			station_list=station_list,
			event_row=event_row,
			base_input_dir=cfg.base_input_dir,
			network_code=cfg.network_code,
			pre_sec=cfg.pre_sec,
			post_sec=cfg.post_sec,
			span_min_default=1,
			threads=cfg.hinet_threads,
			save_catalog_fields=True,
		)

		print(f'  event_dir: {event_dir}')
		print(
			f'  cnt_files: {len(cnt_paths)} (first={cnt_paths[0].name if cnt_paths else "N/A"})'
		)
		print(f'  ch_file: {ch_path.name}')
		print('  done')


if __name__ == '__main__':
	main()
