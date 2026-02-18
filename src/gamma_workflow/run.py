"""Shared GaMMA execution flow from CSV inputs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from gamma.utils import association

from gamma_workflow.config import build_gamma_config
from gamma_workflow.normalize import (
	estimate_station_origin_m,
	normalize_picks,
	normalize_stations,
)
from gamma_workflow.velocity import load_velocity_json

if TYPE_CHECKING:
	from pathlib import Path


def run_gamma_from_csvs(  # noqa: PLR0913
	*,
	picks_csv: Path,
	stations_csv: Path,
	vel_json: Path,
	out_dir: Path,
	method: str,
	use_dbscan: bool,
	use_amplitude: bool,
	oversample_factor_bgmm: int,
	use_eikonal_1d: bool,
	eikonal_h_km: float,
	xy_margin_km: float,
	z_range_km: tuple[float, float],
	dbscan_eps_sec: float | None,
	dbscan_eps_sigma: float,
	dbscan_eps_mult: float,
	dbscan_min_samples: int,
	dbscan_min_cluster_size: int,
	dbscan_max_time_space_ratio: float,
	ncpu: int,
	min_picks_per_eq: int,
	min_p_picks_per_eq: int,
	min_s_picks_per_eq: int,
	max_sigma11_sec: float,
	max_sigma22_log10_ms: float,
	max_sigma12_cov: float,
) -> dict:
	"""Run GaMMA and write config/events/picks CSV outputs."""
	if not picks_csv.exists():
		raise FileNotFoundError(f'PICKS_CSV not found: {picks_csv}')
	if not stations_csv.exists():
		raise FileNotFoundError(f'STATIONS_CSV not found: {stations_csv}')
	if use_eikonal_1d and not vel_json.exists():
		raise FileNotFoundError(f'VEL_MODEL_JSON not found: {vel_json}')

	out_dir.mkdir(parents=True, exist_ok=True)

	picks_raw = pd.read_csv(picks_csv)
	stations_raw = pd.read_csv(stations_csv)

	picks = normalize_picks(picks_raw)
	stations = normalize_stations(stations_raw)
	origin_e_m, origin_n_m = estimate_station_origin_m(stations_raw)

	picks = picks[picks['id'].isin(set(stations['id']))].reset_index(drop=True)
	if use_amplitude:
		picks = picks[picks['amp'] != -1].reset_index(drop=True)

	vel = load_velocity_json(vel_json) if use_eikonal_1d else None
	config = build_gamma_config(
		stations_df=stations,
		vel=vel,
		method=method,
		use_dbscan=use_dbscan,
		use_amplitude=use_amplitude,
		oversample_factor_bgmm=oversample_factor_bgmm,
		use_eikonal_1d=use_eikonal_1d,
		eikonal_h_km=eikonal_h_km,
		xy_margin_km=xy_margin_km,
		z_range_km=z_range_km,
		dbscan_eps_sec=dbscan_eps_sec,
		dbscan_eps_sigma=dbscan_eps_sigma,
		dbscan_eps_mult=dbscan_eps_mult,
		dbscan_min_samples=dbscan_min_samples,
		dbscan_min_cluster_size=dbscan_min_cluster_size,
		dbscan_max_time_space_ratio=dbscan_max_time_space_ratio,
		ncpu=ncpu,
		min_picks_per_eq=min_picks_per_eq,
		min_p_picks_per_eq=min_p_picks_per_eq,
		min_s_picks_per_eq=min_s_picks_per_eq,
		max_sigma11_sec=max_sigma11_sec,
		max_sigma22_log10_ms=max_sigma22_log10_ms,
		max_sigma12_cov=max_sigma12_cov,
	)

	config_path = out_dir / 'gamma_config.json'
	config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')

	event_idx0 = 0
	events, assignments = association(
		picks, stations, config, event_idx0, config['method']
	)

	events_df = pd.DataFrame(events)
	events_path = out_dir / 'gamma_events.csv'
	if len(events_df) == 0:
		events_path.write_text('', encoding='utf-8')
	else:
		if not np.isnan(origin_e_m) and not np.isnan(origin_n_m):
			events_df['E_m'] = (
				origin_e_m + events_df['x(km)'].to_numpy(dtype=float) * 1000.0
			)
			events_df['N_m'] = (
				origin_n_m + events_df['y(km)'].to_numpy(dtype=float) * 1000.0
			)
		events_df.to_csv(
			events_path,
			index=False,
			float_format='%.6f',
			date_format='%Y-%m-%dT%H:%M:%S.%f',
		)

	assign_df = pd.DataFrame(
		assignments, columns=['pick_index', 'event_index', 'gamma_score']
	)
	picks_out = picks.copy()
	picks_out = picks_out.join(assign_df.set_index('pick_index')).fillna(-1)
	picks_out['event_index'] = picks_out['event_index'].astype(int)

	picks_out = picks_out.rename(
		columns={
			'id': 'station_id',
			'timestamp': 'phase_time',
			'type': 'phase_type',
			'prob': 'phase_score',
			'amp': 'phase_amplitude',
		}
	)

	picks_path = out_dir / 'gamma_picks.csv'
	picks_out.to_csv(
		picks_path,
		index=False,
		date_format='%Y-%m-%dT%H:%M:%S.%f',
	)

	return {
		'stations_count': len(stations),
		'picks_count': len(picks_out),
		'assigned_count': int((picks_out['event_index'] >= 0).sum()),
		'events_count': len(events_df),
		'config': config,
		'events_path': events_path,
		'picks_path': picks_path,
		'config_path': config_path,
	}
