from __future__ import annotations

import logging
import traceback
import warnings
from collections.abc import Callable
from datetime import datetime, timedelta
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common.config import JmaDtPickErrorConfigV1
from common.geo import haversine_distance_km
from common.run_snapshot import save_yaml_and_effective
from common.time_util import _as_jst, _format_jst_iso, iso_to_ns
from jma.picks import build_pick_table_for_event
from jma.prepare.event_txt import read_event_txt_meta, read_origin_jst_iso
from jma.prepare.inventory import build_inventory
from jma.prepare.jma_waveform_loader import load_u_stream_for_event
from jma.stationcode_mappingdb import load_mapping_db
from jma.stationcode_presence import load_presence_db
from pick.dt_eval import eval_dt_row
from pick.picks_from_probs import extract_pick_near_ref
from pick.prob_picker import build_probs_by_station
from pick.stalta_probs import StaltaProbSpec
_DT_TOL_REQUIRED = [0.05, 0.10, 0.20]
_DT_TABLE_COLUMNS = [
	'run_id',
	'picker_name',
	'picker_preset',
	'preprocess_preset',
	'phase',
	'component',
	'fs_hz',
	'search_pre_sec',
	'search_post_sec',
	'thr',
	'min_sep_sec',
	'choose',
	'tie_break',
	'event_id',
	'station',
	'mag',
	'mag1_type',
	'distance_hypo_km',
	't0_iso',
	't_ref_iso',
	'ref_pick_idx',
	'search_i0',
	'search_i1',
	'found_peak',
	't_est_iso',
	'est_pick_idx',
	'dt_sec',
	'score_at_pick',
	'n_peaks',
	'good_0p05',
	'good_0p10',
	'good_0p20',
	'fail_reason',
	'stalta_transform',
	'stalta_sta_sec',
	'stalta_lta_sec',
]


def _make_run_logger(log_path: Path) -> logging.Logger:
	"""File-only logger (no stdout/stderr handlers)."""
	logger = logging.getLogger(f'jma_dt_pick_error_table::{log_path!s}')
	logger.setLevel(logging.INFO)
	logger.propagate = False
	for h in list(logger.handlers):
		logger.removeHandler(h)

	log_path.parent.mkdir(parents=True, exist_ok=True)
	fh = logging.FileHandler(log_path, encoding='utf-8')
	fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
	fh.setFormatter(fmt)
	logger.addHandler(fh)
	return logger


def _install_warning_logger(logger: logging.Logger) -> Callable[[], None]:
	"""Redirect warnings (including pandas UserWarning) to the file logger only."""
	prev_showwarning = warnings.showwarning

	def _showwarning(
		message: object,
		category: type[Warning],
		filename: str,
		lineno: int,
		file: object | None = None,
		line: str | None = None,
	) -> None:
		logger.warning(f'{category.__name__}: {message} ({filename}:{lineno})')

	warnings.showwarning = _showwarning
	warnings.simplefilter('default')

	def _restore() -> None:
		warnings.showwarning = prev_showwarning

	return _restore


def _prepare_epicenters(epi_df: pd.DataFrame) -> pd.DataFrame:
	req = {
		'event_id',
		'origin_time',
		'latitude_deg',
		'longitude_deg',
		'depth_km',
		'mag1',
		'mag1_type',
	}
	miss = sorted(req - set(epi_df.columns))
	if miss:
		raise ValueError(f'epicenters csv missing columns: {miss}')

	origin_str = epi_df['origin_time'].astype(str)
	dt64 = pd.to_datetime(origin_str, format='ISO8601', errors='raise').to_numpy(
		dtype='datetime64[ns]'
	)
	out = epi_df.copy()
	out['_origin_ns'] = dt64.astype('int64')
	out['_mag1_type_norm'] = out['mag1_type'].astype(str).str.strip()
	return out


def _resolve_epicenters_row(
	epi_df: pd.DataFrame,
	*,
	origin_iso: str,
	txt_lat: float,
	txt_lon: float,
	mag1_types_allowed: set[str],
	dist_tie_km: float = 0.1,
) -> tuple[pd.Series | None, str]:
	ns = int(iso_to_ns(origin_iso))
	cands = epi_df.loc[epi_df['_origin_ns'] == ns]
	if cands.empty:
		return None, 'epicenters_origin_not_found'

	allowed = {str(x).strip() for x in mag1_types_allowed if str(x).strip()}
	if not allowed:
		raise ValueError('mag1_types_allowed is empty')

	cands2 = cands.loc[cands['_mag1_type_norm'].isin(sorted(allowed))]
	if cands2.empty:
		return None, 'mag1_type_not_allowed'

	if len(cands2) == 1:
		return cands2.iloc[0], ''

	d = haversine_distance_km(
		lat0_deg=float(txt_lat),
		lon0_deg=float(txt_lon),
		lat_deg=cands2['latitude_deg'].astype(float).to_numpy(),
		lon_deg=cands2['longitude_deg'].astype(float).to_numpy(),
	)
	order = np.argsort(d)
	best_i = int(order[0])
	best_d = float(d[best_i])

	if len(order) >= 2:
		second_d = float(d[int(order[1])])
		if abs(second_d - best_d) <= float(dist_tie_km):
			sub = cands2.copy()
			sub['_dist_km'] = d
			cols = ['latitude_deg', 'longitude_deg', 'depth_km', 'mag1']
			uniq = sub[cols].nunique(dropna=False).max()
			if int(uniq) == 1:
				sub2 = sub.sort_values('event_id', ascending=True)
				return sub2.iloc[0], ''

			ids = sub.sort_values('_dist_km')[
				[
					'event_id',
					'latitude_deg',
					'longitude_deg',
					'depth_km',
					'mag1',
					'mag1_type',
				]
			].head(10)
			raise ValueError(
				'ambiguous epicenters rows for same origin_time (and allowed mag1_type set). '
				f'origin={origin_iso} txt_latlon=({txt_lat},{txt_lon}) '
				f'top_candidates=\n{ids.to_string(index=False)}'
			)

	return cands2.iloc[best_i], ''


def _resolve_out_path(out_dir: Path, path: Path) -> Path:
	p = Path(path)
	if p.is_absolute():
		return p
	return Path(out_dir) / p


def _append_skip(
	skips: list[dict[str, object]],
	*,
	event_dir: Path,
	reason: str,
	event_id: int | None = None,
	station: str | None = None,
) -> None:
	skips.append(
		{
			'event_dir': str(event_dir),
			'event_id': '' if event_id is None else int(event_id),
			'station': '' if station is None else str(station),
			'reason': str(reason),
		}
	)


def run_jma_dt_pick_error_table(
	cfg: JmaDtPickErrorConfigV1,
	*,
	yaml_path: Path,
	preset: str = 'v1',
	continue_on_event_error: bool = False,
	log_warnings: bool = False,
	log_filename: str = 'run.log',
) -> tuple[pd.DataFrame, pd.DataFrame]:
	out_dir = Path(cfg.run.out_dir)
	if out_dir.exists() and not cfg.run.overwrite:
		raise FileExistsError(f'out_dir exists: {out_dir}')
	out_dir.mkdir(parents=True, exist_ok=True)

	logger: logging.Logger | None = None
	restore_warnings: Callable[[], None] | None = None
	if continue_on_event_error or log_warnings:
		logger = _make_run_logger(out_dir / str(log_filename))
		if log_warnings:
			restore_warnings = _install_warning_logger(logger)

	try:
		if cfg.output.save_config_snapshot:
			save_yaml_and_effective(
				out_dir=out_dir,
				yaml_path=yaml_path,
				preset=preset,
				cfg_obj=cfg,
				label='pipeline',
			)

		inputs = cfg.inputs
		for p in [
			inputs.event_root,
			inputs.epicenters_csv,
			inputs.measurements_csv,
			inputs.mapping_report_csv,
			inputs.near0_csv,
			inputs.monthly_presence_csv,
		]:
			if not Path(p).exists():
				raise FileNotFoundError(p)

		tol_sec = [float(x) for x in cfg.eval.tol_sec]
		if [float(x) for x in _DT_TOL_REQUIRED] != [float(x) for x in tol_sec]:
			raise ValueError(f'eval.tol_sec must be {_DT_TOL_REQUIRED}, got {tol_sec}')

		mag1_types_allowed = {str(x).strip() for x in inputs.mag1_types_allowed}
		if not mag1_types_allowed:
			raise ValueError('inputs.mag1_types_allowed is empty')

		p_phases = set(inputs.phase_defs.get('P', []))
		if not p_phases:
			raise ValueError('inputs.phase_defs.P is empty')
		p_phases = {str(x).strip().upper() for x in p_phases if str(x).strip()}

		s_phases = set(inputs.phase_defs.get('S', []))
		s_phases = {str(x).strip().upper() for x in s_phases if str(x).strip()}

		epi_df = pd.read_csv(Path(inputs.epicenters_csv), low_memory=False)
		epi_df2 = _prepare_epicenters(epi_df)
		meas_df = pd.read_csv(Path(inputs.measurements_csv), low_memory=False)
		mdb = load_mapping_db(inputs.mapping_report_csv, inputs.near0_csv)
		pdb = load_presence_db(inputs.monthly_presence_csv)

		allowlist = None
		if inputs.event_id_allowlist is not None:
			allowlist = {int(x) for x in inputs.event_id_allowlist}

		stations_allow = None
		if inputs.stations_allowlist is not None:
			stations_allow = {str(x) for x in inputs.stations_allowlist}

		rows: list[dict[str, object]] = []
		skips: list[dict[str, object]] = []

		stalta_cfg = cfg.picker.stalta
		stalta_spec = None
		stalta_transform = ''
		stalta_sta_sec = float('nan')
		stalta_lta_sec = float('nan')
		if stalta_cfg is not None:
			stalta_transform = str(stalta_cfg.transform)
			stalta_sta_sec = float(stalta_cfg.sta_sec)
			stalta_lta_sec = float(stalta_cfg.lta_sec)
			stalta_spec = StaltaProbSpec(
				transform=stalta_transform,
				sta_sec=stalta_sta_sec,
				lta_sec=stalta_lta_sec,
			)

		base_row = {
			'run_id': str(cfg.run.run_id),
			'picker_name': str(cfg.picker.picker_name),
			'picker_preset': str(cfg.picker.picker_preset),
			'preprocess_preset': str(cfg.preprocess.preprocess_preset),
			'phase': str(cfg.picker.phase),
			'component': str(cfg.picker.component),
			'fs_hz': float(cfg.preprocess.fs_target_hz),
			'search_pre_sec': float(cfg.pick_extract.search_pre_sec),
			'search_post_sec': float(cfg.pick_extract.search_post_sec),
			'thr': float(cfg.pick_extract.thr),
			'min_sep_sec': float(cfg.pick_extract.min_sep_sec),
			'choose': str(cfg.pick_extract.choose),
			'tie_break': str(cfg.pick_extract.tie_break),
			'stalta_transform': stalta_transform,
			'stalta_sta_sec': stalta_sta_sec,
			'stalta_lta_sec': stalta_lta_sec,
		}

		event_dirs = sorted(
			[p for p in Path(inputs.event_root).iterdir() if p.is_dir()]
		)
		if not event_dirs:
			raise ValueError(f'no event dirs under {inputs.event_root}')

		for i_ev, ev_dir in enumerate(event_dirs):
			evt_paths = sorted(ev_dir.glob('*.evt'))
			if len(evt_paths) != 1:
				_append_skip(
					skips,
					event_dir=ev_dir,
					reason=f'evt_not_single:{len(evt_paths)}',
				)
				continue
			evt_path = evt_paths[0]
			txt_path = evt_path.with_suffix('.txt')
			if not txt_path.is_file():
				_append_skip(skips, event_dir=ev_dir, reason='txt_missing')
				continue
			ch_path = ev_dir / f'{evt_path.stem}.ch'
			if not ch_path.is_file():
				_append_skip(skips, event_dir=ev_dir, reason='ch_missing')
				continue

			origin_iso = read_origin_jst_iso(txt_path)
			meta = read_event_txt_meta(txt_path)

			epi_row, epi_reason = _resolve_epicenters_row(
				epi_df2,
				origin_iso=origin_iso,
				txt_lat=float(meta.lat),
				txt_lon=float(meta.lon),
				mag1_types_allowed=mag1_types_allowed,
			)
			if epi_row is None:
				_append_skip(
					skips,
					event_dir=ev_dir,
					reason=epi_reason,
				)
				continue

			event_id = int(epi_row['event_id'])
			if allowlist is not None and event_id not in allowlist:
				_append_skip(
					skips,
					event_dir=ev_dir,
					event_id=event_id,
					reason='event_id_not_allowed',
				)
				continue

			mag1_type = str(epi_row['mag1_type']).strip()
			if mag1_type not in mag1_types_allowed:
				_append_skip(
					skips,
					event_dir=ev_dir,
					event_id=event_id,
					reason=f'mag1_type_not_allowed:{mag1_type}',
				)
				continue

			mag1 = float(epi_row['mag1'])
			if not np.isfinite(mag1):
				_append_skip(
					skips,
					event_dir=ev_dir,
					event_id=event_id,
					reason='mag1_nan',
				)
				continue

			depth_km = float(epi_row['depth_km'])
			if not np.isfinite(depth_km):
				_append_skip(
					skips,
					event_dir=ev_dir,
					event_id=event_id,
					reason='depth_km_nan',
				)
				continue

			pick_df, _log_rows = build_pick_table_for_event(
				meas_df,
				event_id=event_id,
				event_month=meta.event_month,
				mdb=mdb,
				pdb=pdb,
				p_phases=p_phases,
				s_phases=s_phases,
			)
			if pick_df.empty or 'p_time' not in pick_df.columns:
				_append_skip(
					skips,
					event_dir=ev_dir,
					event_id=event_id,
					reason='no_p_pick_table',
				)
				continue

			pick_df_p = pick_df.loc[pd.notna(pick_df['p_time'])].copy()
			if pick_df_p.empty:
				_append_skip(
					skips,
					event_dir=ev_dir,
					event_id=event_id,
					reason='no_p_picks',
				)
				continue

			station_list = pick_df_p.index.astype(str).tolist()
			if stations_allow is not None:
				station_list = [s for s in station_list if s in stations_allow]
			if not station_list:
				_append_skip(
					skips,
					event_dir=ev_dir,
					event_id=event_id,
					reason='no_stations_after_allowlist',
				)
				continue

			try:
				inv = build_inventory(ev_dir)
			except Exception as e:
				_append_skip(
					skips,
					event_dir=ev_dir,
					event_id=event_id,
					reason=f'inventory_error:{type(e).__name__}',
				)
				if logger is not None:
					logger.error(
						f'[inventory_error] event_dir={ev_dir!s} event_id={event_id} err={type(e).__name__}: {e}'
					)
					logger.error(traceback.format_exc())
				if not continue_on_event_error:
					raise
				continue

			try:
				load_res = load_u_stream_for_event(
					ev_dir,
					stations=station_list,
					preprocess_cfg=cfg.preprocess,
					inventory=inv,
				)
			except Exception as e:
				_append_skip(
					skips,
					event_dir=ev_dir,
					event_id=event_id,
					reason=f'waveform_load_error:{type(e).__name__}',
				)
				if logger is not None:
					logger.error(
						f'[waveform_load_error] event_dir={ev_dir!s} event_id={event_id} err={type(e).__name__}: {e}'
					)
					logger.error(traceback.format_exc())
				if not continue_on_event_error:
					raise
				continue

			if abs(load_res.fs_hz - float(cfg.preprocess.fs_target_hz)) > 1e-6:
				raise ValueError(
					f'fs_hz mismatch: loader={load_res.fs_hz} cfg={cfg.preprocess.fs_target_hz}'
				)

			for s in load_res.skips:
				s2 = dict(s)
				s2['event_id'] = int(event_id)
				s2.setdefault('station', '')
				s2.setdefault('reason', '')
				skips.append(s2)

			st = load_res.stream
			t0_jst = load_res.t0
			t0_iso = _format_jst_iso(t0_jst)

			try:
				probs_by_station = build_probs_by_station(
					cfg.picker.picker_name,
					st,
					fs=load_res.fs_hz,
					component=cfg.picker.component,
					phase=cfg.picker.phase,
					stalta_spec=stalta_spec,
				)
			except Exception as e:
				_append_skip(
					skips,
					event_dir=ev_dir,
					event_id=event_id,
					reason=f'prob_build_error:{type(e).__name__}',
				)
				if logger is not None:
					logger.error(
						f'[prob_build_error] event_dir={ev_dir!s} event_id={event_id} err={type(e).__name__}: {e}'
					)
					logger.error(traceback.format_exc())
				if not continue_on_event_error:
					raise
				continue

			station_set = set(station_list)
			pick_df_p = pick_df_p.loc[pick_df_p.index.astype(str).isin(station_set)]
			if pick_df_p.empty:
				_append_skip(
					skips,
					event_dir=ev_dir,
					event_id=event_id,
					reason='no_picks_after_allowlist',
				)
				continue

			for sta, r in pick_df_p.iterrows():
				sta_s = str(sta)
				st_meta = inv.station_meta.get(sta_s)
				if st_meta is None:
					_append_skip(
						skips,
						event_dir=ev_dir,
						event_id=event_id,
						station=sta_s,
						reason='station_not_in_inventory',
					)
					continue
				if 'lat' not in st_meta or 'lon' not in st_meta:
					_append_skip(
						skips,
						event_dir=ev_dir,
						event_id=event_id,
						station=sta_s,
						reason='station_latlon_missing',
					)
					continue

				sta_lat = float(st_meta['lat'])
				sta_lon = float(st_meta['lon'])
				src_lat = float(epi_row['latitude_deg'])
				src_lon = float(epi_row['longitude_deg'])
				epi_dist_km = float(
					haversine_distance_km(
						lat0_deg=src_lat,
						lon0_deg=src_lon,
						lat_deg=np.asarray([sta_lat], dtype=float),
						lon_deg=np.asarray([sta_lon], dtype=float),
					)[0]
				)
				hypo_dist_km = float(
					sqrt(epi_dist_km * epi_dist_km + depth_km * depth_km)
				)

				prob_by_phase = probs_by_station.get(sta_s)
				if prob_by_phase is None:
					_append_skip(
						skips,
						event_dir=ev_dir,
						event_id=event_id,
						station=sta_s,
						reason='no_prob',
					)
					continue

				score = prob_by_phase.get(cfg.picker.phase)
				if score is None:
					_append_skip(
						skips,
						event_dir=ev_dir,
						event_id=event_id,
						station=sta_s,
						reason='no_prob_phase',
					)
					continue

				pick_time = r.get('p_time')
				if pick_time is None or pd.isna(pick_time):
					_append_skip(
						skips,
						event_dir=ev_dir,
						event_id=event_id,
						station=sta_s,
						reason='p_time_nan',
					)
					continue

				t_ref = pd.Timestamp(pick_time).to_pydatetime()
				t_ref_jst = _as_jst(t_ref)
				ref_idx = int(
					round(
						(t_ref_jst - _as_jst(t0_jst)).total_seconds() * load_res.fs_hz
					)
				)

				score_arr = np.asarray(score, dtype=float)
				extract: dict[str, Any]
				if ref_idx < 0 or ref_idx >= int(score_arr.size):
					extract = {
						'found_peak': False,
						'est_pick_idx': float('nan'),
						'score_at_pick': float('nan'),
						'n_peaks': 0,
						'search_i0': None,
						'search_i1': None,
						'fail_reason': 'ref_out_of_range',
					}
				else:
					extract = extract_pick_near_ref(
						score_arr,
						float(ref_idx),
						fs_hz=float(load_res.fs_hz),
						search_pre_sec=float(cfg.pick_extract.search_pre_sec),
						search_post_sec=float(cfg.pick_extract.search_post_sec),
						thr=float(cfg.pick_extract.thr),
						min_sep_sec=float(cfg.pick_extract.min_sep_sec),
						clip_search_window=bool(cfg.pick_extract.clip_search_window),
						search_i1_inclusive=bool(cfg.pick_extract.search_i1_inclusive),
					)

				fail_reason = str(extract.get('fail_reason', ''))
				found_peak = bool(extract.get('found_peak', False))
				est_pick_idx = (
					None
					if not found_peak
					else float(extract.get('est_pick_idx', float('nan')))
				)

				eval_row = eval_dt_row(
					t0_jst=t0_jst,
					t_ref=t_ref,
					fs_hz=float(load_res.fs_hz),
					est_pick_idx=est_pick_idx,
					found_peak=found_peak,
					tol_sec=tol_sec,
					keep_missing_rows=bool(cfg.eval.keep_missing_rows),
					score_at_pick=extract.get('score_at_pick', None),
					n_peaks=extract.get('n_peaks', None),
					search_i0=extract.get('search_i0', None),
					search_i1=extract.get('search_i1', None),
					fail_reason=fail_reason,
				)
				if eval_row is None:
					continue

				row = dict(base_row)
				row.update(
					{
						'event_id': int(event_id),
						'station': sta_s,
						'mag': float(mag1),
						'mag1_type': str(mag1_type),
						'distance_hypo_km': float(hypo_dist_km),
						't0_iso': str(t0_iso),
					}
				)
				row.update(eval_row)
				rows.append(row)

			if (i_ev % 50) == 0:
				msg = (
					f'[progress] {i_ev}/{len(event_dirs)} events | '
					f'rows={len(rows)} skips={len(skips)}'
				)
				if logger is not None:
					logger.info(msg)
				else:
					print(msg)

		dt_df = pd.DataFrame(rows)
		dt_df = dt_df.reindex(columns=_DT_TABLE_COLUMNS)
		dt_out = _resolve_out_path(out_dir, Path(cfg.output.dt_table_csv))
		dt_out.parent.mkdir(parents=True, exist_ok=True)
		dt_df.to_csv(dt_out, index=False, na_rep='NaN')

		skip_df = pd.DataFrame(skips)
		skip_df = skip_df.reindex(
			columns=['event_dir', 'event_id', 'station', 'reason']
		)
		skip_out = _resolve_out_path(out_dir, Path(cfg.output.skips_csv))
		skip_out.parent.mkdir(parents=True, exist_ok=True)
		skip_df.to_csv(skip_out, index=False)

		msg1 = f'[saved] dt_table: {dt_out} rows={len(dt_df)}'
		msg2 = f'[saved] skips: {skip_out} rows={len(skip_df)}'
		if logger is not None:
			logger.info(msg1)
			logger.info(msg2)
		else:
			print(msg1)
			print(msg2)

		return dt_df, skip_df

	finally:
		if restore_warnings is not None:
			restore_warnings()
