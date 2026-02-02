# %%
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import detrend as sp_detrend

from common.geo import haversine_distance_km, latlon_to_local_xy_km
from common.time_util import iso_to_ns
from jma.picks import (
	build_pick_table_for_event,
	pick_time_to_index,
)
from jma.prepare.event_txt import read_event_txt_meta, read_origin_jst_iso
from jma.prepare.inventory import build_inventory
from jma.station_reader import read_hinet_channel_table
from jma.stationcode_mappingdb import load_mapping_db
from jma.stationcode_presence import load_presence_db
from jma.win32_reader import read_win32, scan_channel_sampling_rate_map_win32
from waveform.filters import bandpass_iir_filtfilt
from waveform.preprocess import resample_window_poly


@dataclass(frozen=True)
class SNRSpec:
	"""Configuration for SNR computation around phase picks.

	This dataclass controls:
	- target resampling rate (`fs_target_hz`)
	- noise/signal windows relative to the pick time (seconds)
	- preprocessing bandpass filter design parameters
	- basic raw-trace QC thresholds (zero/clip fractions)
	"""

	fs_target_hz: int = 100

	noise_from_s: float = -3.0
	noise_to_s: float = -0.5
	signal_from_s: float = 0.0
	signal_to_s: float = 3.0

	eps_energy: float = 1e-12

	# bandpass (ellip iirdesign)
	fstop_lo: float = 0.5
	fpass_lo: float = 1.0
	fpass_hi: float = 20.0
	fstop_hi: float = 25.0
	gpass: float = 1.0
	gstop: float = 40.0

	# QC thresholds
	zero_frac_max: float = 0.98
	clip_frac_max: float = 0.02


_P_PHASES = {'P', 'EP', 'IP'}
_DATE_ONLY_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')

# 30Hz以下を解析対象外（= fs <= 30 を除外）
MIN_INPUT_FS_HZ = 30


def _parse_iso8601_bound_jst(s: str | None, *, is_end: bool) -> pd.Timestamp | None:
	if s is None:
		return None
	ss = str(s).strip()
	if not ss:
		return None

	# date-only shorthand (inclusive convenience)
	if _DATE_ONLY_RE.match(ss):
		ss = ss + ('T23:59:59.999999' if is_end else 'T00:00:00')

	return pd.to_datetime(ss, format='ISO8601', errors='raise')


def _azimuth_deg_event_to_station(
	event_lat: float, event_lon: float, sta_lat: float, sta_lon: float
) -> float:
	x_km, y_km = latlon_to_local_xy_km(
		np.asarray([sta_lat], dtype=float),
		np.asarray([sta_lon], dtype=float),
		lat0_deg=float(event_lat),
		lon0_deg=float(event_lon),
	)
	az = np.degrees(np.arctan2(x_km[0], y_km[0]))
	return float((az + 360.0) % 360.0)


def _qc_trace_raw(
	x: np.ndarray, *, zero_frac_max: float, clip_frac_max: float
) -> tuple[bool, str]:
	if x.ndim != 1:
		return False, f'bad_shape:{x.shape}'
	if not np.isfinite(x).all():
		return False, 'nan_or_inf'
	if float(np.max(np.abs(x))) <= 0.0:
		return False, 'all_zero'

	zero_frac = float(np.mean(x == 0.0))
	if zero_frac > float(zero_frac_max):
		return False, f'too_many_zeros:{zero_frac:.3f}'

	mx = float(np.max(x))
	mn = float(np.min(x))
	if mx == mn:
		return False, 'constant'

	frac_max = float(np.mean(x == mx))
	frac_min = float(np.mean(x == mn))
	if max(frac_max, frac_min) > float(clip_frac_max):
		return False, f'clipped_suspect:{max(frac_max, frac_min):.3f}'

	return True, ''


def _preprocess_1d(x: np.ndarray, *, fs: float, spec: SNRSpec) -> np.ndarray:
	y = sp_detrend(np.asarray(x, dtype=float), type='linear')
	y = bandpass_iir_filtfilt(
		y,
		fs=float(fs),
		fstop_lo=float(spec.fstop_lo),
		fpass_lo=float(spec.fpass_lo),
		fpass_hi=float(spec.fpass_hi),
		fstop_hi=float(spec.fstop_hi),
		gpass=float(spec.gpass),
		gstop=float(spec.gstop),
	)
	return np.asarray(y, dtype=np.float32)


def _snr_db_energy(
	x_proc: np.ndarray, *, pick_idx: float, fs: float, spec: SNRSpec
) -> tuple[float, float, float]:
	def _idx(t_s: float) -> int:
		return int(round(float(pick_idx) + float(t_s) * float(fs)))

	n0 = _idx(spec.noise_from_s)
	n1 = _idx(spec.noise_to_s)
	s0 = _idx(spec.signal_from_s)
	s1 = _idx(spec.signal_to_s)

	if not (0 <= n0 < n1 <= len(x_proc) and 0 <= s0 < s1 <= len(x_proc)):
		return float('nan'), float('nan'), float('nan')

	noise = x_proc[n0:n1]
	signal = x_proc[s0:s1]

	noise_e = float(np.sum(noise * noise))
	signal_e = float(np.sum(signal * signal))

	snr = signal_e / (noise_e + float(spec.eps_energy))
	snr_db = 10.0 * float(np.log10(max(snr, 1e-300)))

	noise_rms = float(np.sqrt(np.mean(noise * noise)))
	signal_rms = float(np.sqrt(np.mean(signal * signal)))
	return snr_db, noise_rms, signal_rms


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
	epi_df2: pd.DataFrame,
	*,
	origin_iso: str,
	txt_lat: float,
	txt_lon: float,
	mag1_types_allowed: set[str],
	dist_tie_km: float = 0.1,
) -> pd.Series | None:
	ns = int(iso_to_ns(origin_iso))
	cands = epi_df2.loc[epi_df2['_origin_ns'] == ns]
	if cands.empty:
		raise ValueError(f'epicenters row not found by origin_time: {origin_iso}')

	allowed = {str(x).strip() for x in mag1_types_allowed if str(x).strip()}
	if not allowed:
		raise ValueError('mag1_types_allowed is empty')

	cands2 = cands.loc[cands['_mag1_type_norm'].isin(sorted(allowed))]
	if cands2.empty:
		return None

	if len(cands2) == 1:
		return cands2.iloc[0]

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
				return sub2.iloc[0]

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

	return cands2.iloc[best_i]


def build_snr_pick_table(
	*,
	event_root: str | Path,
	epicenters_csv: str | Path,
	measurements_csv: str | Path,
	mapping_report_csv: str | Path,
	near0_csv: str | Path,
	monthly_presence_csv: str | Path,
	out_csv: str | Path,
	out_skip_csv: str | Path,
	mag1_types_allowed: set[str],
	spec: SNRSpec = SNRSpec(),
	start_origin_jst: str | None = None,
	end_origin_jst: str | None = None,
	continue_on_error: bool = False,
	epicenters_dist_tie_km: float = 0.1,
) -> None:
	event_root = Path(event_root)
	out_csv = Path(out_csv)
	out_skip_csv = Path(out_skip_csv)

	start_ts = _parse_iso8601_bound_jst(start_origin_jst, is_end=False)
	end_ts = _parse_iso8601_bound_jst(end_origin_jst, is_end=True)

	epi_df = pd.read_csv(Path(epicenters_csv), low_memory=False)
	meas_df = pd.read_csv(Path(measurements_csv), low_memory=False)
	epi_df2 = _prepare_epicenters(epi_df)

	mdb = load_mapping_db(mapping_report_csv, near0_csv)
	pdb = load_presence_db(monthly_presence_csv)

	event_dirs = sorted([p for p in event_root.iterdir() if p.is_dir()])
	if not event_dirs:
		raise RuntimeError(f'no event dirs found under: {event_root}')

	rows: list[dict[str, object]] = []
	skips: list[dict[str, object]] = []

	if continue_on_error:
		warnings.warn(
			'continue_on_error=True: per-event errors will be skipped (see out_skip_csv).',
			RuntimeWarning,
			stacklevel=2,
		)

	# cache per source_id
	ch_table_cache: dict[str, pd.DataFrame] = {}

	for i_ev, ev_dir in enumerate(event_dirs, start=1):

		def _run_one_event() -> None:
			stem = ev_dir.name
			txt_path = ev_dir / f'{stem}.txt'

			origin_iso = read_origin_jst_iso(txt_path)
			origin_ts = pd.to_datetime(origin_iso, format='ISO8601', errors='raise')

			if start_ts is not None and origin_ts < start_ts:
				skips.append(
					{
						'event_dir': str(ev_dir),
						'origin_jst': str(origin_iso),
						'reason': 'out_of_period_before',
					}
				)
				return
			if end_ts is not None and origin_ts > end_ts:
				skips.append(
					{
						'event_dir': str(ev_dir),
						'origin_jst': str(origin_iso),
						'reason': 'out_of_period_after',
					}
				)
				return

			txt_meta = read_event_txt_meta(txt_path)

			epi_row = _resolve_epicenters_row(
				epi_df2,
				origin_iso=origin_iso,
				txt_lat=float(txt_meta.lat),
				txt_lon=float(txt_meta.lon),
				mag1_types_allowed=set(mag1_types_allowed),
				dist_tie_km=float(epicenters_dist_tie_km),
			)
			if epi_row is None:
				skips.append(
					{
						'event_dir': str(ev_dir),
						'origin_jst': str(origin_iso),
						'reason': f'mag1_type_not_allowed:{sorted(mag1_types_allowed)}',
					}
				)
				return

			eid = int(epi_row['event_id'])

			mag1 = epi_row.get('mag1', None)
			depth_km = epi_row.get('depth_km', None)
			if pd.isna(mag1) or pd.isna(depth_km):
				skips.append(
					{
						'event_dir': str(ev_dir),
						'event_id': eid,
						'origin_jst': str(origin_iso),
						'reason': 'missing_mag_or_depth',
					}
				)
				return

			ev_lat = float(epi_row['latitude_deg'])
			ev_lon = float(epi_row['longitude_deg'])
			depth_km = float(depth_km)
			mag1 = float(mag1)
			mag1_type = str(epi_row['mag1_type']).strip()

			pick_df, _ = build_pick_table_for_event(
				meas_df,
				event_id=eid,
				event_month=txt_meta.event_month,
				mdb=mdb,
				pdb=pdb,
				p_phases=set(_P_PHASES),
				s_phases=set(),
			)
			pick_df = pick_df[pd.notna(pick_df['p_time'])].copy()
			if pick_df.empty:
				skips.append(
					{
						'event_dir': str(ev_dir),
						'event_id': eid,
						'origin_jst': str(origin_iso),
						'reason': 'no_p_picks',
					}
				)
				return

			inv = build_inventory(ev_dir)
			source_by_id = {s.source_id: s for s in inv.sources}

			# まず、このイベントで必要なUチャンネルを source_id ごとにまとめる（ファイル走査回数を減らす）
			required_u_by_source: dict[str, set[int]] = {}
			u_meta_by_station: dict[str, dict[str, object]] = {}

			for sta, r in pick_df.iterrows():
				st_meta = inv.station_meta.get(sta)
				if st_meta is None:
					u_meta_by_station[str(sta)] = {
						'_skip_reason': 'station_not_in_inventory'
					}
					continue

				u = st_meta.get('U')
				if not u:
					u_meta_by_station[str(sta)] = {'_skip_reason': 'no_U_component'}
					continue

				source_id = str(u['source_id'])
				ch_int = int(u['ch_int'])

				u_meta_by_station[str(sta)] = {
					'_skip_reason': '',
					'u': u,
					'st_meta': st_meta,
					'row': r,
					'source_id': source_id,
					'ch_int': ch_int,
				}
				required_u_by_source.setdefault(source_id, set()).add(ch_int)

			# source_idごとに「必要チャンネルだけ」fsをスキャン（混在はdrop）
			fs_map_by_source: dict[str, dict[int, int]] = {}
			for source_id, ch_set in required_u_by_source.items():
				if source_id not in source_by_id:
					continue
				data_path = Path(source_by_id[source_id].data_path)

				fs_map = scan_channel_sampling_rate_map_win32(
					data_path,
					channel_filter=set(ch_set),
					on_mixed='drop',
				)
				fs_map_by_source[source_id] = {
					int(k): int(v) for k, v in fs_map.items()
				}

			# stationごとに処理
			for sta, _ in pick_df.iterrows():
				key = str(sta)
				meta0 = u_meta_by_station.get(key)
				if meta0 is None:
					skips.append(
						{
							'event_dir': str(ev_dir),
							'event_id': eid,
							'station': str(sta),
							'reason': 'station_meta_missing_internal',
						}
					)
					continue

				skip_reason = str(meta0.get('_skip_reason', ''))
				if skip_reason:
					skips.append(
						{
							'event_dir': str(ev_dir),
							'event_id': eid,
							'station': str(sta),
							'reason': skip_reason,
						}
					)
					continue

				u = meta0['u']
				st_meta = meta0['st_meta']
				r = meta0['row']
				source_id = str(meta0['source_id'])
				ch_int = int(meta0['ch_int'])

				if source_id not in source_by_id:
					skips.append(
						{
							'event_dir': str(ev_dir),
							'event_id': eid,
							'station': str(sta),
							'reason': f'source_not_found:{source_id}',
						}
					)
					continue

				src = source_by_id[source_id]
				meta = inv.sources_meta[source_id]

				start_time = datetime.fromisoformat(str(meta['start_time']))
				span_seconds = int(meta['span_seconds'])
				n_t = int(span_seconds) * int(spec.fs_target_hz)

				pick_idx = pick_time_to_index(
					r['p_time'],
					fs_hz=float(spec.fs_target_hz),
					t_start=start_time,
					n_t=int(n_t),
				)
				if not np.isfinite(pick_idx):
					skips.append(
						{
							'event_dir': str(ev_dir),
							'event_id': eid,
							'station': str(sta),
							'reason': 'pick_out_of_range',
						}
					)
					continue

				data_path = Path(src.data_path)
				ch_path = Path(src.ch_path)

				if source_id not in ch_table_cache:
					ch_table_cache[source_id] = read_hinet_channel_table(ch_path)
				ch_table = ch_table_cache[source_id]

				fs_map = fs_map_by_source.get(source_id, {})
				fs_in = fs_map.get(ch_int, None)

				# 混在(drop) or 見つからない -> 除外
				if fs_in is None:
					skips.append(
						{
							'event_dir': str(ev_dir),
							'event_id': eid,
							'station': str(sta),
							'reason': 'mixed_sampling_rate_or_not_found',
						}
					)
					continue

				# 30Hz以下は除外（= fs <= 30）
				if int(fs_in) <= int(MIN_INPUT_FS_HZ):
					skips.append(
						{
							'event_dir': str(ev_dir),
							'event_id': eid,
							'station': str(sta),
							'reason': f'low_sampling_rate:{int(fs_in)}',
						}
					)
					continue

				ch_hex = f'{ch_int:04X}'

				raw2d = read_win32(
					data_path,
					ch_table,
					base_sampling_rate_HZ=int(fs_in),
					duration_SECOND=int(span_seconds),
					channels_hex=[ch_hex],
				)
				raw = np.asarray(raw2d[0], dtype=np.float32)

				ok, why = _qc_trace_raw(
					raw,
					zero_frac_max=float(spec.zero_frac_max),
					clip_frac_max=float(spec.clip_frac_max),
				)
				if not ok:
					skips.append(
						{
							'event_dir': str(ev_dir),
							'event_id': eid,
							'station': str(sta),
							'reason': why,
						}
					)
					continue

				out_len = int(span_seconds) * int(spec.fs_target_hz)
				raw_rs = resample_window_poly(
					raw[None, :],
					fs_in=float(fs_in),
					fs_out=float(spec.fs_target_hz),
					out_len=int(out_len),
				)[0]

				x_proc = _preprocess_1d(raw_rs, fs=float(spec.fs_target_hz), spec=spec)

				snr_db, noise_rms, signal_rms = _snr_db_energy(
					x_proc,
					pick_idx=float(pick_idx),
					fs=float(spec.fs_target_hz),
					spec=spec,
				)
				if not np.isfinite(snr_db):
					skips.append(
						{
							'event_dir': str(ev_dir),
							'event_id': eid,
							'station': str(sta),
							'reason': 'snr_window_out_of_range',
						}
					)
					continue

				sta_lat = float(st_meta['lat'])
				sta_lon = float(st_meta['lon'])

				epi_dist_km = float(
					haversine_distance_km(
						lat0_deg=float(ev_lat),
						lon0_deg=float(ev_lon),
						lat_deg=np.asarray([sta_lat], dtype=float),
						lon_deg=np.asarray([sta_lon], dtype=float),
					)[0]
				)
				hypo_dist_km = float(
					np.sqrt(epi_dist_km * epi_dist_km + depth_km * depth_km)
				)
				az = _azimuth_deg_event_to_station(ev_lat, ev_lon, sta_lat, sta_lon)

				rows.append(
					{
						'event_dir': str(ev_dir),
						'event_id': int(eid),
						'origin_jst': str(origin_iso),
						'origin_time_epicenters': str(epi_row['origin_time']),
						'event_lat': float(ev_lat),
						'event_lon': float(ev_lon),
						'depth_km': float(depth_km),
						'mag1': float(mag1),
						'mag1_type': str(mag1_type),
						'station': str(sta),
						'station_lat': float(sta_lat),
						'station_lon': float(sta_lon),
						'distance_hypo_km': float(hypo_dist_km),
						'azimuth_deg': float(az),
						'pick_time': pd.Timestamp(r['p_time']).strftime(
							'%Y-%m-%dT%H:%M:%S.%f'
						)[:-3],
						'snr_db': float(snr_db),
						'noise_rms': float(noise_rms),
						'signal_rms': float(signal_rms),
						'waveform_kind': str(u['kind']),
						'network_code': str(u['network_code']),
						'source_id': str(source_id),
						'ch_int': int(ch_int),
						'fs_in_hz': int(fs_in),
						'fs_target_hz': int(spec.fs_target_hz),
						'bandpass_fpass_lo_hz': float(spec.fpass_lo),
						'bandpass_fpass_hi_hz': float(spec.fpass_hi),
						'noise_win_s': f'{spec.noise_from_s},{spec.noise_to_s}',
						'signal_win_s': f'{spec.signal_from_s},{spec.signal_to_s}',
					}
				)

		if continue_on_error:
			try:
				_run_one_event()
			except (FileNotFoundError, ValueError, KeyError) as e:
				warnings.warn(
					f'skip event_dir={ev_dir} error={type(e).__name__}: {e}',
					RuntimeWarning,
				)
				skips.append(
					{
						'event_dir': str(ev_dir),
						'reason': f'event_error:{type(e).__name__}:{e}',
					}
				)
		else:
			_run_one_event()

		if (i_ev % 50) == 0:
			print(
				f'[progress] {i_ev}/{len(event_dirs)} events | rows={len(rows)} skips={len(skips)}'
			)

	df = pd.DataFrame(rows)
	df.to_csv(out_csv, index=False)

	df_skip = pd.DataFrame(skips)
	df_skip.to_csv(out_skip_csv, index=False)

	print(f'[done] wrote: {out_csv}')
	print(f'[done] wrote: {out_skip_csv}')
	print(f'[done] rows={len(df)} skips={len(df_skip)}')


EVENT_ROOT = Path('/workspace/data/waveform/jma/event')

EPICENTERS_CSV = Path('/workspace/data/arrivetime/JMA/arrivetime_epicenters_2023.0.csv')
MEASUREMENTS_CSV = Path(
	'/workspace/data/arrivetime/JMA/arrivetime_measurements_2023.0.csv'
)

MAPPING_REPORT_CSV = Path(
	'/workspace/proc/prepare_data/jma/stationcode_match/v1/match_out_final/mapping_report.csv'
)
NEAR0_CSV = Path('/workspace/data/arrivetime/JMA/match_out_final/near0_suggestions.csv')
MONTHLY_PRESENCE_CSV = Path(
	'/workspace/proc/prepare_data/jma/stationcode_match/v1/snapshots/monthly/monthly_presence.csv'
)

OUT_CSV = Path('/workspace/data/waveform/jma/snr_pick_table.csv')
OUT_SKIP_CSV = Path('/workspace/data/waveform/jma/snr_pick_table_skips.csv')

MAG1_TYPES_ALLOWED = {'v', 'V'}

# 期間（JST, inclusive）
START_ORIGIN_JST = '2023-01-01'
END_ORIGIN_JST = '2023-01-31'

SPEC = SNRSpec(
	fs_target_hz=100,
	noise_from_s=-3.0,
	noise_to_s=-0.5,
	signal_from_s=0.0,
	signal_to_s=3.0,
	fstop_lo=0.5,
	fpass_lo=1.0,
	fpass_hi=20.0,
	fstop_hi=25.0,
	eps_energy=1e-12,
	zero_frac_max=0.98,
	clip_frac_max=0.02,
)

CONTINUE_ON_ERROR = True

# origin_time重複のとき、lat/lon距離がこの値以内で僅差なら「曖昧」として落とす（誤対応防止）
EPICENTERS_DIST_TIE_KM = 0.1

build_snr_pick_table(
	event_root=EVENT_ROOT,
	epicenters_csv=EPICENTERS_CSV,
	measurements_csv=MEASUREMENTS_CSV,
	mapping_report_csv=MAPPING_REPORT_CSV,
	near0_csv=NEAR0_CSV,
	monthly_presence_csv=MONTHLY_PRESENCE_CSV,
	out_csv=OUT_CSV,
	out_skip_csv=OUT_SKIP_CSV,
	mag1_types_allowed=MAG1_TYPES_ALLOWED,
	spec=SPEC,
	start_origin_jst=START_ORIGIN_JST,
	end_origin_jst=END_ORIGIN_JST,
	continue_on_error=CONTINUE_ON_ERROR,
	epicenters_dist_tie_km=EPICENTERS_DIST_TIE_KM,
)
