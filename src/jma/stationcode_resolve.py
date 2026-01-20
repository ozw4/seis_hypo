# src/jma/stationcode_resolve.py
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from jma.stationcode_common import normalize_code, pick_preferred_network_code
from jma.stationcode_mappingdb import MappingDB
from jma.stationcode_presence import PresenceDB


def _parse_pipe_list(s: object) -> list[str]:
	if s is None or pd.isna(s):
		return []
	ss = str(s).strip()
	if not ss:
		return []
	return [normalize_code(x) for x in ss.split('|') if x.strip()]


def _rule_prefix(rule: str) -> str:
	m = re.match(r'^([^:]+):', rule)
	return m.group(1) if m else rule


def _present_rows(pdb: PresenceDB, *, ch_key: str, event_month: str) -> pd.DataFrame:
	if event_month not in pdb.month_cols:
		raise ValueError(f'event_month={event_month} not in presence columns')
	rows = pdb.pres[pdb.pres['ch_key'] == ch_key]
	if rows.empty:
		return rows
	return rows[rows[event_month] == 1]


def _pick_network_for_ch(pdb: PresenceDB, *, ch_key: str, event_month: str) -> str:
	present = _present_rows(pdb, ch_key=ch_key, event_month=event_month)
	if present.empty:
		return ''
	return pick_preferred_network_code(present['network_code'].tolist())


@dataclass(frozen=True)
class MappingDecision:
	status: str
	mea_norm: str
	ch_key: str | None
	rule: str | None
	network_code: str | None
	candidates_in_month: list[tuple[str, str]]


def decide_mea_to_ch_for_month(
	mea_norm: str,
	*,
	event_month: str,
	mdb: MappingDB,
	pdb: PresenceDB,
) -> MappingDecision:
	mea_norm2 = normalize_code(mea_norm)

	if mea_norm2 in mdb.report.index:
		row = mdb.report.loc[mea_norm2]
		match_status = str(row.get('match_status', '')).strip().lower()
		if match_status != 'matched':
			return MappingDecision(
				status=f'report_{match_status}',
				mea_norm=mea_norm2,
				ch_key=None,
				rule=None,
				network_code=None,
				candidates_in_month=[],
			)

		hits = _parse_pipe_list(row.get('all_hit_stations_norm'))
		rules = _parse_pipe_list(row.get('all_hit_rules'))
		if len(hits) != len(rules):
			raise ValueError(
				f'mapping_report mismatch mea_norm={mea_norm2}: hits={len(hits)} rules={len(rules)}'
			)

		pairs = [(h, r) for h, r in zip(hits, rules, strict=True) if h in pdb.ch_set]
		pairs_in_month = [
			(h, r)
			for h, r in pairs
			if not _present_rows(pdb, ch_key=h, event_month=event_month).empty
		]
		if not pairs_in_month:
			return MappingDecision(
				status='no_candidate_in_month',
				mea_norm=mea_norm2,
				ch_key=None,
				rule=None,
				network_code=None,
				candidates_in_month=[],
			)

		pref0 = [(h, r) for h, r in pairs_in_month if _rule_prefix(r) == '0']
		strong = [
			(h, r) for h, r in pairs_in_month if _rule_prefix(r) in {'J', 'A', 'C', 'D'}
		]
		weak = [(h, r) for h, r in pairs_in_month if _rule_prefix(r) == 'B']

		bucket = pref0 if pref0 else (strong if strong else weak)
		if len(bucket) != 1:
			return MappingDecision(
				status='ambiguous_in_month',
				mea_norm=mea_norm2,
				ch_key=None,
				rule=None,
				network_code=None,
				candidates_in_month=bucket,
			)

		ch_key, rule = bucket[0]
		nc = _pick_network_for_ch(pdb, ch_key=ch_key, event_month=event_month)
		if not nc:
			return MappingDecision(
				status='no_network_in_month',
				mea_norm=mea_norm2,
				ch_key=None,
				rule=None,
				network_code=None,
				candidates_in_month=bucket,
			)

		return MappingDecision(
			status='mapped',
			mea_norm=mea_norm2,
			ch_key=ch_key,
			rule=rule,
			network_code=nc,
			candidates_in_month=bucket,
		)

	# near0 rescue
	if mdb.near0.empty:
		return MappingDecision(
			status='unmatched_no_near0',
			mea_norm=mea_norm2,
			ch_key=None,
			rule=None,
			network_code=None,
			candidates_in_month=[],
		)

	n0 = mdb.near0[mdb.near0['mea_norm'] == mea_norm2]
	if n0.empty:
		return MappingDecision(
			status='unmatched_no_near0',
			mea_norm=mea_norm2,
			ch_key=None,
			rule=None,
			network_code=None,
			candidates_in_month=[],
		)

	cands: list[tuple[str, str]] = []
	for _, rr in n0.iterrows():
		ch_key = normalize_code(rr.get('suggest_ch_station_norm'))
		if ch_key not in pdb.ch_set:
			continue
		dkm = float(rr.get('nearest_distance_km'))
		if dkm > 0.03:
			continue

		first = str(rr.get('overlap_first', '')).strip()
		last = str(rr.get('overlap_last', '')).strip()
		if not first or not last:
			continue
		if not (first <= event_month <= last):
			continue
		if _present_rows(pdb, ch_key=ch_key, event_month=event_month).empty:
			continue

		cands.append((ch_key, 'near0'))

	uniq = sorted(set(cands))
	if len(uniq) != 1:
		return MappingDecision(
			status='near0_ambiguous_or_none',
			mea_norm=mea_norm2,
			ch_key=None,
			rule=None,
			network_code=None,
			candidates_in_month=uniq,
		)

	ch_key, rule = uniq[0]
	nc = _pick_network_for_ch(pdb, ch_key=ch_key, event_month=event_month)
	if not nc:
		return MappingDecision(
			status='no_network_in_month',
			mea_norm=mea_norm2,
			ch_key=None,
			rule=None,
			network_code=None,
			candidates_in_month=uniq,
		)

	return MappingDecision(
		status='mapped_near0',
		mea_norm=mea_norm2,
		ch_key=ch_key,
		rule=rule,
		network_code=nc,
		candidates_in_month=uniq,
	)
