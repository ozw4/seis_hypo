from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from common.geo import haversine_distance_km


@dataclass(frozen=True)
class EventStationSelectionSpec:
	# イベント窓（あなたの仕様）
	pre_sec: int = 30
	post_sec: int = 150

	# ここが修正点：最小 station 数
	min_stations: int = 64

	# measurements で「検測あり」と見なす列
	phase_time_cols: tuple[str, str] = ('phase1_time', 'phase2_time')

	# epicenters / stations の列名
	epic_lat_col: str = 'latitude_deg'
	epic_lon_col: str = 'longitude_deg'
	station_code_col: str = 'station_code'
	station_lat_col: str = 'latitude_deg'
	station_lon_col: str = 'longitude_deg'


def _unique_keep_order(items: Iterable[str]) -> list[str]:
	out: list[str] = []
	seen: set[str] = set()
	for x in items:
		s = str(x).strip()
		if not s:
			continue
		if s in seen:
			continue
		seen.add(s)
		out.append(s)
	return out


def required_stations_from_measurements(
	meas_df_event: pd.DataFrame,
	*,
	spec: EventStationSelectionSpec = EventStationSelectionSpec(),
) -> list[str]:
	"""検測（phase timeが入ってる） station_code を必須局として返す。"""
	for c in (spec.station_code_col, *spec.phase_time_cols):
		if c not in meas_df_event.columns:
			raise ValueError(f'measurements DF missing column: {c}')

	mask = (
		meas_df_event[spec.phase_time_cols[0]].notna()
		| meas_df_event[spec.phase_time_cols[1]].notna()
	)
	req = (
		meas_df_event.loc[mask, spec.station_code_col].astype(str).str.strip().tolist()
	)
	return _unique_keep_order(req)


def select_stations_nearest_fill(
	*,
	event_lat: float,
	event_lon: float,
	required_stations: list[str],
	station_geo_df: pd.DataFrame,
	station_to_network: dict[str, str],
	spec: EventStationSelectionSpec = EventStationSelectionSpec(),
) -> list[str]:
	"""必須局を含め、足りない分を震源から距離順で足して min_stations を満たす station_code を返す。"""
	required = _unique_keep_order(required_stations)

	missing_req = [s for s in required if s not in station_to_network]
	if missing_req:
		raise ValueError(
			f'required stations not found in station_to_network (first 30): {missing_req[:30]}'
		)

	if spec.min_stations <= 0:
		raise ValueError('min_stations must be > 0')

	# 必須だけで満たせるならそれで終わり
	if len(required) >= spec.min_stations:
		return required

	# station_geo_df 必須列
	for c in (spec.station_code_col, spec.station_lat_col, spec.station_lon_col):
		if c not in station_geo_df.columns:
			raise ValueError(f'station_geo_df missing column: {c}')

	df = station_geo_df[
		[spec.station_code_col, spec.station_lat_col, spec.station_lon_col]
	].copy()
	df[spec.station_code_col] = df[spec.station_code_col].astype(str).str.strip()

	# “ダウンロード可能な station” のみに絞る（network未判定は除外）
	df = df[df[spec.station_code_col].isin(station_to_network.keys())].copy()
	if df.empty:
		raise ValueError(
			'station_geo_df has no downloadable stations after filtering by station_to_network'
		)

	selected = list(required)
	selected_set = set(selected)

	df_rest = df[~df[spec.station_code_col].isin(selected_set)].copy()
	if df_rest.empty:
		raise ValueError('no additional stations available to fill min_stations')

	dist_km = haversine_distance_km(
		lat0_deg=float(event_lat),
		lon0_deg=float(event_lon),
		lat_deg=df_rest[spec.station_lat_col].to_numpy(dtype=float),
		lon_deg=df_rest[spec.station_lon_col].to_numpy(dtype=float),
	)

	order = np.argsort(dist_km)
	rest_codes = df_rest.iloc[order][spec.station_code_col].tolist()

	for sta in rest_codes:
		selected.append(sta)
		if len(selected) >= spec.min_stations:
			break

	if len(selected) < spec.min_stations:
		raise ValueError(
			f'insufficient stations even after selecting all candidates: '
			f'n_selected={len(selected)} < min_stations={spec.min_stations}'
		)

	return selected


def group_stations_by_network(
	stations: list[str],
	*,
	station_to_network: dict[str, str],
) -> dict[str, list[str]]:
	"""station_code を network_code -> station_code[] に分解。"""
	out: dict[str, list[str]] = {}
	for s in _unique_keep_order(stations):
		if s not in station_to_network:
			raise ValueError(f'station_to_network missing: {s}')
		net = station_to_network[s]
		out.setdefault(net, []).append(s)
	return out


def build_stations_by_network_for_event(
	*,
	event_id: int,
	epic_df: pd.DataFrame,
	meas_df: pd.DataFrame,
	station_geo_df: pd.DataFrame,
	station_to_network: dict[str, str],
	spec: EventStationSelectionSpec = EventStationSelectionSpec(),
) -> dict[str, list[str]]:
	"""epicenters/measurements CSVから、1イベントぶんの stations_by_network を構築する。"""
	if 'event_id' not in epic_df.columns:
		raise ValueError('epic_df missing column: event_id')
	if 'event_id' not in meas_df.columns:
		raise ValueError('meas_df missing column: event_id')

	for c in (spec.epic_lat_col, spec.epic_lon_col):
		if c not in epic_df.columns:
			raise ValueError(f'epic_df missing column: {c}')

	erows = epic_df[epic_df['event_id'] == int(event_id)]
	if erows.empty:
		raise ValueError(f'event_id not found in epic_df: {event_id}')
	erow = erows.iloc[0]

	event_lat = float(erow[spec.epic_lat_col])
	event_lon = float(erow[spec.epic_lon_col])

	m = meas_df[meas_df['event_id'] == int(event_id)]
	required = required_stations_from_measurements(m, spec=spec)

	selected = select_stations_nearest_fill(
		event_lat=event_lat,
		event_lon=event_lon,
		required_stations=required,
		station_geo_df=station_geo_df,
		station_to_network=station_to_network,
		spec=spec,
	)

	return group_stations_by_network(selected, station_to_network=station_to_network)
