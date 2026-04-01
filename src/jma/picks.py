# src/jma/picks.py
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from common.time_util import iso_to_ns
from jma.stationcode_common import normalize_code
from jma.stationcode_mappingdb import MappingDB
from jma.stationcode_presence import PresenceDB
from jma.stationcode_resolve import decide_mea_to_ch_for_month

P_PHASES_DEFAULT = {'P', 'EP', 'IP'}
S_PHASES_DEFAULT = {'S', 'ES', 'IS'}
_PHASE_SLOT_SPECS = (
	('phase1', 'phase_name_1', 'phase1_time'),
	('phase2', 'phase_name_2', 'phase2_time'),
)


def _raw_text_or_empty(value: object) -> str:
	if value is None or pd.isna(value):
		return ''
	return str(value)


def _is_high_confidence_phase_name(phase_name_raw: str) -> bool:
	return phase_name_raw.strip().isupper()


def _pick_trace_candidates_for_row(
	row: pd.Series,
	*,
	t1: pd.Timestamp | pd.NaT,
	t2: pd.Timestamp | pd.NaT,
	p_phases: set[str],
	s_phases: set[str],
) -> dict[str, dict[str, object]]:
	candidates: dict[str, dict[str, object]] = {}
	time_by_slot = {'phase1': t1, 'phase2': t2}

	for phase_slot, phase_name_col, _time_col in _PHASE_SLOT_SPECS:
		phase_name_raw = _raw_text_or_empty(row.get(phase_name_col))
		phase_name_norm = phase_name_raw.upper()
		picked_time = time_by_slot[phase_slot]
		if pd.isna(picked_time):
			continue

		phase_type = ''
		if phase_name_norm in p_phases:
			phase_type = 'P'
		elif phase_name_norm in s_phases:
			phase_type = 'S'

		if phase_type == '':
			continue

		prev = candidates.get(phase_type)
		if prev is None or picked_time < prev['picked_time']:
			candidates[phase_type] = {
				'phase_slot': phase_slot,
				'phase_name_raw': phase_name_raw,
				'phase_type': phase_type,
				'is_high_confidence': _is_high_confidence_phase_name(phase_name_raw),
				'picked_time': picked_time,
			}

	return candidates


def build_epicenters_origin_index(epi_df: pd.DataFrame) -> dict[int, int]:
	"""epicenters: origin_time(timestamp ns) -> event_id の辞書を作る。重複はエラー。"""
	req = {'event_id', 'origin_time'}
	if not req.issubset(epi_df.columns):
		raise ValueError(
			f'epicenters csv missing columns: {sorted(req - set(epi_df.columns))}'
		)

	origin_str = epi_df['origin_time'].astype(str)
	dt64 = pd.to_datetime(origin_str, format='ISO8601', errors='raise').to_numpy(
		dtype='datetime64[ns]'
	)
	ns = dt64.astype('int64')
	eid = epi_df['event_id'].astype(int).to_numpy()

	dup_mask = pd.Series(ns).duplicated(keep=False).to_numpy()
	if bool(np.any(dup_mask)):
		dup_ns = sorted(set(int(x) for x in ns[dup_mask].tolist()))
		raise ValueError(
			f'epicenters has duplicated origin_time(ns). Example ns: {dup_ns[:20]}'
		)

	return {int(k): int(v) for k, v in zip(ns.tolist(), eid.tolist())}


def find_event_id_by_origin_exact(
	epi_origin_ns_to_event_id: dict[int, int],
	origin_iso: str,
) -> int:
	"""origin_iso を timestamp(ns) にして完全一致検索。無ければエラー（フォールバック無し）。"""
	ns = iso_to_ns(origin_iso)
	if ns not in epi_origin_ns_to_event_id:
		raise ValueError(f'event_id not found by exact origin_time: {origin_iso}')
	return int(epi_origin_ns_to_event_id[ns])


def build_pick_table_for_event(
	meas_df: pd.DataFrame,
	*,
	event_id: int,
	event_month: str,
	mdb: MappingDB,
	pdb: PresenceDB,
	p_phases: set[str] | None = None,
	s_phases: set[str] | None = None,
	pick_trace_rows: list[dict[str, object]] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
	"""arrivetime_measurements からイベント1つ分の pick table を作る。

	- station_code(measurement) -> ch_station(presence/.ch側) は現行ルール（MappingDB+PresenceDB）で解決
	- 同一 ch_station に複数行が来たら、p_time/s_time は最も早いものを採用
	- pick_trace_rows を渡すと、採用された phase slot の provenance を rows に格納する
	- 返り値
	- pick_df: index=ch_station, columns=[p_time, s_time, preferred_network_code]
	- log_rows: mapping判定ログ（学習前の整合チェック用）
	"""
	p_ph = P_PHASES_DEFAULT if p_phases is None else set(p_phases)
	s_ph = S_PHASES_DEFAULT if s_phases is None else set(s_phases)
	if pick_trace_rows is not None:
		pick_trace_rows.clear()

	m = meas_df[meas_df['event_id'] == event_id].copy()
	out = pd.DataFrame(columns=['p_time', 's_time', 'preferred_network_code'])
	out.index.name = 'ch_station'
	if m.empty:
		return out, []

	t1 = pd.to_datetime(m['phase1_time'], format='ISO8601', errors='raise')
	t2 = pd.to_datetime(m['phase2_time'], format='ISO8601', errors='raise')

	acc_p: dict[str, pd.Timestamp | pd.NaT] = {}
	acc_s: dict[str, pd.Timestamp | pd.NaT] = {}
	acc_nc: dict[str, set[str]] = {}
	trace_p: dict[str, dict[str, object]] = {}
	trace_s: dict[str, dict[str, object]] = {}
	log_rows: list[dict[str, object]] = []

	for i, row in m.iterrows():
		sta_raw = row.get('station_code')
		mea_norm = normalize_code(sta_raw)

		dec = decide_mea_to_ch_for_month(
			mea_norm, event_month=event_month, mdb=mdb, pdb=pdb
		)

		log_rows.append(
			{
				'event_id': int(event_id),
				'event_month': event_month,
				'mea_station_code': '' if sta_raw is None else str(sta_raw),
				'mea_norm': mea_norm,
				'map_status': dec.status,
				'ch_station': '' if dec.ch_key is None else dec.ch_key,
				'network_code': '' if dec.network_code is None else dec.network_code,
				'map_rule': '' if dec.rule is None else dec.rule,
				'candidates_in_month': '|'.join(
					[f'{c}:{r}' for c, r in dec.candidates_in_month]
				),
			}
		)

		if dec.ch_key is None or not dec.network_code:
			continue

		row_candidates = _pick_trace_candidates_for_row(
			row,
			t1=t1.loc[i],
			t2=t2.loc[i],
			p_phases=p_ph,
			s_phases=s_ph,
		)
		pt = row_candidates['P']['picked_time'] if 'P' in row_candidates else pd.NaT
		st = row_candidates['S']['picked_time'] if 'S' in row_candidates else pd.NaT

		curp = acc_p.get(dec.ch_key, pd.NaT)
		curs = acc_s.get(dec.ch_key, pd.NaT)

		if pd.notna(pt) and (pd.isna(curp) or pt < curp):
			if pick_trace_rows is not None:
				trace_p[dec.ch_key] = {
					'event_id': int(event_id),
					'event_month': event_month,
					'mea_station_code_raw': _raw_text_or_empty(sta_raw),
					'mea_station_code_norm': mea_norm,
					'phase_slot': row_candidates['P']['phase_slot'],
					'phase_name_raw': row_candidates['P']['phase_name_raw'],
					'phase_type': 'P',
					'is_high_confidence': row_candidates['P']['is_high_confidence'],
					'picked_time': row_candidates['P']['picked_time'].isoformat(),
					'ch_station': dec.ch_key,
					'network_code': dec.network_code,
					'map_status': dec.status,
					'map_rule': '' if dec.rule is None else dec.rule,
				}
			curp = pt if pd.isna(curp) else min(curp, pt)
		if pd.notna(st) and (pd.isna(curs) or st < curs):
			if pick_trace_rows is not None:
				trace_s[dec.ch_key] = {
					'event_id': int(event_id),
					'event_month': event_month,
					'mea_station_code_raw': _raw_text_or_empty(sta_raw),
					'mea_station_code_norm': mea_norm,
					'phase_slot': row_candidates['S']['phase_slot'],
					'phase_name_raw': row_candidates['S']['phase_name_raw'],
					'phase_type': 'S',
					'is_high_confidence': row_candidates['S']['is_high_confidence'],
					'picked_time': row_candidates['S']['picked_time'].isoformat(),
					'ch_station': dec.ch_key,
					'network_code': dec.network_code,
					'map_status': dec.status,
					'map_rule': '' if dec.rule is None else dec.rule,
				}
			curs = st if pd.isna(curs) else min(curs, st)

		acc_p[dec.ch_key] = curp
		acc_s[dec.ch_key] = curs
		acc_nc.setdefault(dec.ch_key, set()).add(dec.network_code)

	keys = sorted(acc_p.keys())
	out = pd.DataFrame(
		{
			'p_time': [acc_p.get(k, pd.NaT) for k in keys],
			's_time': [acc_s.get(k, pd.NaT) for k in keys],
			'preferred_network_code': [
				';'.join(sorted(acc_nc.get(k, set()))) for k in keys
			],
		},
		index=pd.Index(keys, name='ch_station'),
	)
	if pick_trace_rows is not None:
		for ch_key in keys:
			if ch_key in trace_p:
				pick_trace_rows.append(trace_p[ch_key])
			if ch_key in trace_s:
				pick_trace_rows.append(trace_s[ch_key])
	return out, log_rows


def pick_time_to_index(
	pick_time: pd.Timestamp | datetime | None,
	*,
	fs_hz: float,
	t_start: datetime,
	n_t: int,
) -> float:
	"""pick時刻をサンプルindex（float）に変換する。

	範囲外は NaN を返す。
	"""
	if pick_time is None or pd.isna(pick_time):
		return float('nan')

	dt_s = (pd.Timestamp(pick_time) - pd.Timestamp(t_start)).total_seconds()
	idx = float(dt_s) * float(fs_hz)
	if 0.0 <= idx < float(n_t):
		return idx
	return float('nan')
