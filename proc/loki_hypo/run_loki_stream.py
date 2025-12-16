# %%
#!/usr/bin/env python
# run_loki_stream.py
#
# 目的:
# - 既存パイプラインで作った ObsPy Stream を LOKI に「ファイルI/Oなし」で渡して stacking/locator を回す
# 前提:
# - LOKI側の waveforms.Waveforms が traces=Stream を受け取れるように改造済み
# - LOKI側の loki.Loki.location が streams_by_event を受け取れるように改造済み
#
# 注意:
# - LOKI は data_path 配下の「末端ディレクトリ」をイベントとして列挙する実装なので、
#   Stream入力でも data_path にはイベント名の空ディレクトリ（leaf）を用意する（ここで作る）

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from netrc import netrc
from pathlib import Path

import pandas as pd
from HinetPy import Client
from loki.loki import Loki
from obspy import Stream

from catalog.selection import extract_events_in_region
from common.load_config import load_config
from jma.station_reader import stations_within_radius
from pipelines.load_event_stream_from_win32 import prepare_single_event_stream


@dataclass(frozen=True)
class RunLokiStreamConfig:
	# ---- event selection ----
	catalog_csv: Path
	start_time: str | None = None
	end_time: str | None = None
	min_mag: float | None = None
	max_mag: float | None = None
	station_site_lat: float = 0.0
	station_site_lon: float = 0.0
	event_radius_km: float | None = None
	max_events: int | None = None

	# ---- station selection ----
	station_radius_km: float = 50.0

	# ---- prepare stream ----
	base_input_dir: Path = Path('proc/inputs/events')
	network_code: str = '0101'
	fs: float = 100.0
	pre_sec: int = 20
	post_sec: int = 120

	# ---- LOKI paths ----
	loki_data_path: Path = Path(
		'proc/inputs/loki_events'
	)  # 空イベントディレクトリを置く場所
	loki_output_path: Path = Path('proc/outputs/loki')
	loki_db_path: Path = Path('proc/inputs/loki_db')
	loki_hdr_filename: str = 'hdr.hdr'  # 実際のファイル名に合わせて

	# ---- LOKI params ----
	comp: tuple[str, ...] = ('U', 'N', 'E')  # tr.stats.channel の末尾と一致させる
	precision: str = 'single'
	search: str = 'classic'

	# ---- LOKI inputs(**inputs) ----
	model: str = 'iasp91'
	ntrial: int = 1
	npr: int = 1
	freq: tuple[float, ...] | None = None
	opsf: bool = False

	# STA/LTA を使うなら以下を YAML に入れる（入れなければ使わない）
	tshortp_min: float | None = None
	tlongp_min: float | None = None
	tshorts_min: float | None = None
	tlongs_min: float | None = None
	trigger_p: float | None = None
	trigger_s: float | None = None


def create_hinet_client() -> Client:
	login, _, password = netrc().authenticators('hinet')
	return Client(login, password)


def select_stations(cfg: RunLokiStreamConfig) -> Sequence[str]:
	stations = stations_within_radius(
		lat=cfg.station_site_lat,
		lon=cfg.station_site_lon,
		radius_km=cfg.station_radius_km,
	)
	if not stations:
		raise ValueError('stations_within_radius で観測点が1つも選ばれていません。')
	return stations


def select_events(df_catalog: pd.DataFrame, cfg: RunLokiStreamConfig) -> pd.DataFrame:
	epic_sub, _ = extract_events_in_region(
		epic_df=df_catalog,
		meas_df=None,
		start_time=cfg.start_time,
		end_time=cfg.end_time,
		mag_min=cfg.min_mag,
		mag_max=cfg.max_mag,
		center_lat=cfg.station_site_lat,
		center_lon=cfg.station_site_lon,
		radius_km=cfg.event_radius_km,
	)
	epic_sub = epic_sub.sort_values('origin_time')
	if cfg.max_events is not None and cfg.max_events > 0:
		epic_sub = epic_sub.head(cfg.max_events)
	if epic_sub.empty:
		raise ValueError('指定条件を満たすイベントが 0 件です。')
	return epic_sub


def build_loki_inputs(cfg: RunLokiStreamConfig) -> dict:
	inputs: dict = {
		'ntrial': int(cfg.ntrial),
		'npr': int(cfg.npr),
		'model': str(cfg.model),
		'freq': list(cfg.freq) if cfg.freq is not None else None,
		'opsf': bool(cfg.opsf),
	}
	# STALTAは tshortp_min の有無で Loki.location 側が判定する
	if cfg.tshortp_min is not None:
		inputs['tshortp_min'] = float(cfg.tshortp_min)
	if cfg.tlongp_min is not None:
		inputs['tlongp_min'] = float(cfg.tlongp_min)
	if cfg.tshorts_min is not None:
		inputs['tshorts_min'] = float(cfg.tshorts_min)
	if cfg.tlongs_min is not None:
		inputs['tlongs_min'] = float(cfg.tlongs_min)
	if cfg.trigger_p is not None:
		inputs['trigger_p'] = float(cfg.trigger_p)
	if cfg.trigger_s is not None:
		inputs['trigger_s'] = float(cfg.trigger_s)

	# None を消す（LOKI側に渡さない）
	return {k: v for k, v in inputs.items() if v is not None}


def main() -> None:
	if len(sys.argv) < 3:
		raise SystemExit('Usage: run_loki_stream.py CONFIG_YAML PRESET_NAME')

	config_path = Path(sys.argv[1])
	preset = sys.argv[2]
	cfg = load_config(RunLokiStreamConfig, config_path, preset)

	if not cfg.catalog_csv.is_file():
		raise FileNotFoundError(f'catalog_csv not found: {cfg.catalog_csv}')

	cfg.base_input_dir.mkdir(parents=True, exist_ok=True)
	cfg.loki_data_path.mkdir(parents=True, exist_ok=True)
	cfg.loki_output_path.mkdir(parents=True, exist_ok=True)

	df_catalog = pd.read_csv(cfg.catalog_csv)
	df_events = select_events(df_catalog, cfg)
	stations = select_stations(cfg)
	client = create_hinet_client()

	streams_by_event: dict[str, Stream] = {}

	for _, event_row in df_events.iterrows():
		event_id = int(event_row['event_id'])
		event_name = f'{event_id:06d}'

		# LOKI が data_path を os.walk でイベント列挙するので、leaf dir を作る（中身は空でOK）
		(cfg.loki_data_path / event_name).mkdir(parents=True, exist_ok=True)

		event_dir, stream, t_start, t_end = prepare_single_event_stream(
			event_row,
			client=client,
			station_list=stations,
			base_input_dir=cfg.base_input_dir,
			network_code=cfg.network_code,
			pre_sec=cfg.pre_sec,
			post_sec=cfg.post_sec,
			base_sampling_rate_HZ=int(cfg.fs),
			catalog_file=cfg.catalog_csv.name,
		)

		streams_by_event[event_name] = stream
		print(
			f'prepared stream: event={event_name} '
			f'n_traces={len(stream)} t_start={t_start} t_end={t_end} dir={event_dir}'
		)

	l1 = Loki(
		str(cfg.loki_data_path),
		str(cfg.loki_output_path),
		str(cfg.loki_db_path),
		str(cfg.loki_hdr_filename),
		mode='locator',
	)

	inputs = build_loki_inputs(cfg)

	l1.location(
		extension='*',  # 使わない（Stream入力時はreadしない）けど互換のため渡す
		comp=list(cfg.comp),
		precision=cfg.precision,
		search=cfg.search,
		streams_by_event=streams_by_event,
		**inputs,
	)


if __name__ == '__main__':
	main()
