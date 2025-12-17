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

from dataclasses import dataclass
from netrc import netrc
from pathlib import Path

import pandas as pd
from HinetPy import Client

from catalog.selection import extract_events_in_region
from common.load_config import load_config
from jma.station_reader import stations_within_radius

# あなたの環境に合わせて書き換えればOK
YAML_PATH = Path('/workspace/data/config/prepare_events.yaml')
PRESET = 'mobara'


@dataclass(frozen=True)
class PrepareEventsConfig:
	# カタログとイベント出力
	catalog_csv: Path
	base_input_dir: Path = Path('proc/inputs/events')

	# Hi-net
	network_code: str = '0101'
	fs: float = 100.0
	pre_sec: int = 20
	post_sec: int = 120
	hinet_threads: int = 8

	# 観測点選択（局配置）
	station_site_lat: float = 0.0
	station_site_lon: float = 0.0
	station_radius_km: float = 50.0

	# イベント選択（期間・マグ・震央距離）
	start_time: str | None = None
	end_time: str | None = None
	min_mag: float | None = None
	max_mag: float | None = None
	event_radius_km: float | None = None

	# 上限
	max_events: int | None = None

	# 任意メタ
	config_name: str | None = None


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

		event_dir, stream, t_start, t_end = prepare_single_event(
			event_row,
			client=client,
			station_list=station_list,
			base_input_dir=base_input_dir,
			network_code=cfg.network_code,
			pre_sec=cfg.pre_sec,
			post_sec=cfg.post_sec,
			base_sampling_rate_HZ=int(cfg.fs),
		)

		print(
			f'  event_dir={event_dir}, '
			f'n_traces={len(stream)}, '
			f't_start={t_start.isoformat()}, '
			f't_end={t_end.isoformat()}'
		)


if __name__ == '__main__':
	main()
