# src/jma/picks.py
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from jma.stationcode_common import normalize_code
from jma.stationcode_mappingdb import MappingDB
from jma.stationcode_presence import PresenceDB
from jma.stationcode_resolve import decide_mea_to_ch_for_month

_ORIGIN_RE = re.compile(
	r'^\s*ORIGIN_JST\s*:\s*(\d{4})/(\d{2})/(\d{2})\s+(\d{2}):(\d{2}):(\d{2})\.(\d+)\s*$'
)

P_PHASES_DEFAULT = {'P', 'EP', 'IP'}
S_PHASES_DEFAULT = {'S', 'ES', 'IS'}


def _iso_to_ns(origin_iso: str) -> int:
	t = pd.to_datetime(origin_iso, format='ISO8601', errors='raise')
	dt64 = np.datetime64(t.to_datetime64())
	return int(dt64.astype('datetime64[ns]').astype('int64'))


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
	ns = _iso_to_ns(origin_iso)
	if ns not in epi_origin_ns_to_event_id:
		raise ValueError(f'event_id not found by exact origin_time: {origin_iso}')
	return int(epi_origin_ns_to_event_id[ns])


def read_origin_iso_from_txt(txt_path: str | Path) -> str:
	"""HinetPyが出力する *.txt の ORIGIN_JST から ISO8601文字列を作る。

	返り値例: "2023-01-18T02:21:35.35"（小数2桁）
	"""
	p = Path(txt_path)
	if not p.is_file():
		raise FileNotFoundError(p)

	for line in p.read_text(encoding='cp932', errors='strict').splitlines():
		m = _ORIGIN_RE.match(line)
		if m is None:
			continue
		y, mo, d, hh, mm, ss, frac = m.groups()
		frac2 = (frac + '00')[:2]
		return f'{y}-{mo}-{d}T{hh}:{mm}:{ss}.{frac2}'

	raise ValueError(f'ORIGIN_JST not found in {p}')


def build_pick_table_for_event(
	meas_df: pd.DataFrame,
	*,
	event_id: int,
	event_month: str,
	mdb: MappingDB,
	pdb: PresenceDB,
	p_phases: set[str] | None = None,
	s_phases: set[str] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
	"""arrivetime_measurements からイベント1つ分の pick table を作る。

	- station_code(measurement) -> ch_station(presence/.ch側) は現行ルール（MappingDB+PresenceDB）で解決
	- 同一 ch_station に複数行が来たら、p_time/s_time は最も早いものを採用
	- 返り値
	  - pick_df: index=ch_station, columns=[p_time, s_time, preferred_network_code]
	  - log_rows: mapping判定ログ（学習前の整合チェック用）
	"""
	p_ph = P_PHASES_DEFAULT if p_phases is None else set(p_phases)
	s_ph = S_PHASES_DEFAULT if s_phases is None else set(s_phases)

	m = meas_df[meas_df['event_id'] == event_id].copy()
	out = pd.DataFrame(columns=['p_time', 's_time', 'preferred_network_code'])
	out.index.name = 'ch_station'
	if m.empty:
		return out, []

	ph1 = m['phase_name_1'].astype(str).str.upper()
	ph2 = m['phase_name_2'].astype(str).str.upper()

	t1 = pd.to_datetime(m['phase1_time'], format='ISO8601', errors='raise')
	t2 = pd.to_datetime(m['phase2_time'], format='ISO8601', errors='raise')

	p1 = t1.where(ph1.isin(p_ph))
	p2 = t2.where(ph2.isin(p_ph))
	s1 = t1.where(ph1.isin(s_ph))
	s2 = t2.where(ph2.isin(s_ph))

	p_time = pd.concat([p1, p2], axis=1).min(axis=1)
	s_time = pd.concat([s1, s2], axis=1).min(axis=1)

	acc_p: dict[str, pd.Timestamp | pd.NaT] = {}
	acc_s: dict[str, pd.Timestamp | pd.NaT] = {}
	acc_nc: dict[str, set[str]] = {}
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

		pt = p_time.loc[i]
		st = s_time.loc[i]

		curp = acc_p.get(dec.ch_key, pd.NaT)
		curs = acc_s.get(dec.ch_key, pd.NaT)

		if pd.notna(pt):
			curp = pt if pd.isna(curp) else min(curp, pt)
		if pd.notna(st):
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
