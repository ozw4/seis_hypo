# %%
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

import logging
from pathlib import Path

import pandas as pd

from catalog.selection import extract_events_in_region
from common.config import PrepareEventsConfig
from common.load_config import load_config
from jma.download import create_hinet_client, download_win_for_event
from jma.station_reader import stations_within_radius

logger = logging.getLogger(__name__)

# あなたの環境に合わせて書き換えればOK
YAML_PATH = Path('/workspace/data/config/prepare_events.yaml')
PRESET = 'mobara'


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


def _configure_logging() -> None:
	if logging.getLogger().handlers:
		return
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s %(levelname)s %(name)s: %(message)s',
	)


def main() -> None:
	_configure_logging()
	cfg = load_config(PrepareEventsConfig, YAML_PATH, PRESET)

	if not cfg.catalog_csv.is_file():
		msg = f'catalog CSV not found: {cfg.catalog_csv}'
		raise FileNotFoundError(msg)

	df_catalog = pd.read_csv(cfg.catalog_csv)

	# 既存の selection ロジックをそのまま利用してイベントを選ぶ
	df_events = filter_events_with_existing_logic(df_catalog, cfg)
	logger.info('selected events: %d', len(df_events))

	# 観測点リストも既存の stations_within_radius を使う
	station_list = select_stations_for_site(cfg)
	logger.info('selected stations: %d', len(station_list))

	client = create_hinet_client()

	base_input_dir = cfg.base_input_dir
	base_input_dir.mkdir(parents=True, exist_ok=True)

	for _, event_row in df_events.iterrows():
		event_id = int(event_row['event_id'])
		logger.info('prepare event_id=%d', event_id)

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

		logger.info('  event_dir: %s', event_dir)
		logger.info(
			'  cnt_files: %d (first=%s)',
			len(cnt_paths),
			cnt_paths[0].name if cnt_paths else 'N/A',
		)
		logger.info('  ch_file: %s', ch_path.name)
		logger.info('  done')


if __name__ == '__main__':
	main()
