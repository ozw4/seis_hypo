# %%
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common.core import validate_columns
from jma.stationcode_common import month_columns

# =========================
# CONFIG (edit here)
# =========================
MEA_CSV = Path('/workspace/proc/prepare_data/jma/station_measurement.csv')
STA_CSV = Path('/workspace/data/station/jma/station.csv')
CH_CSV = Path(
	'/workspace/proc/prepare_data/jma/stationcode_match/v1/snapshots/monthly/monthly_presence.csv'
)

# Exclude lists
I95_TXT = Path('/workspace/data/station/jma/i95.txt')  # I95(Jxxxx...) table text
EARTHSCOPE_CODES_EXCLUDE = {
	'W.ADK',
	'W.COR',
	'W.CTAO',
	'W.DAV',
	'W.GNI',
	'W.HKT',
	'W.HRV',
	'W.KBS',
	'W.KEV',
	'W.KIEV',
	'W.KIP',
	'W.KMNB',
	'W.KONO',
	'W.LSZ',
	'W.MAJO',
	'W.NACB',
	'W.OTAV',
	'W.PET',
	'W.SDV',
	'W.SFJD',
	'W.SJG',
	'W.SSLB',
	'W.TATO',
	'W.TLY',
	'W.TWGB',
	'W.ULN',
	'W.YAK',
	'W.YHNB',
	'W.YSS',
	'W.YULB',
}

# CH downloadable coverage start
CH_START_DATE = pd.Timestamp('2004-04-01')

# Outputs
OUT_DIR = Path('./match_out_final')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Nearest neighbor diagnostics for unmatched
NEAREST_CH_CHUNK = 4096

# Optional near-zero suggestions (do NOT auto-accept; just export suggestions)
ENABLE_NEAR0_SUGGESTIONS = True
NEAR0_KM = 0.03  # 30m

# Export options
EXPORT_XLSX = True
EXPORT_UTF8_BOM_CSV = True
EXPORT_CP932_CSV = False  # enable only if needed


# =========================
# Column names (edit if differs)
# =========================
MEA_STATION_COL = 'station_code'

STA_STATION_COL = 'station_code'
STA_LAT_COL = 'Latitude_deg'
STA_LON_COL = 'Longitude_deg'
STA_FROM_COL = 'From'
STA_TO_COL = 'To'
STA_COMMENT_COL = 'Comment'  # optional
STA_SEIS_COL = 'Seismographs'  # optional

CH_STATION_COL = 'station'
CH_NETWORK_COL = 'network_code'
CH_LAT_COL = 'lat'
CH_LON_COL = 'lon'
CH_NAME_COL = 'station_name'  # optional


# =========================
# DONET mapping (provided table: JMAコード -> 観測点名(英語)=KMA01等)
# Used as: M.01DO -> M.KMA01, M209DO -> M.MRC09, ...
# =========================
DONET_TO_JMA: dict[str, str] = {
	'M.01DO': 'KMA01',
	'M.02DO': 'KMA02',
	'M.03DO': 'KMA03',
	'M.04DO': 'KMA04',
	'M.05DO': 'KMB05',
	'M.06DO': 'KMB06',
	'M.07DO': 'KMB07',
	'M.08DO': 'KMB08',
	'M.09DO': 'KMC09',
	'M.10DO': 'KMC10',
	'M.11DO': 'KMC11',
	'M.12DO': 'KMC12',
	'M.13DO': 'KMD13',
	'M.14DO': 'KMD14',
	'M.15DO': 'KMD15',
	'M.16DO': 'KMD16',
	'M.17DO': 'KME17',
	'M.18DO': 'KME18',
	'M.19DO': 'KME19',
	'M.21DO': 'KMC21',
	'M201DO': 'MRA01',
	'M202DO': 'MRA02',
	'M203DO': 'MRA03',
	'M204DO': 'MRA04',
	'M206DO': 'MRB06',
	'M207DO': 'MRB07',
	'M208DO': 'MRB08',
	'M209DO': 'MRC09',
	'M210DO': 'MRC10',
	'M211DO': 'MRC11',
	'M212DO': 'MRC12',
	'M213DO': 'MRD13',
	'M214DO': 'MRD14',
	'M215DO': 'MRD15',
	'M216DO': 'MRD16',
	'M217DO': 'MRD17',
	'M218DO': 'MRE18',
	'M219DO': 'MRE19',
	'M220DO': 'MRE20',
	'M221DO': 'MRE21',
	'M222DO': 'MRF22',
	'M223DO': 'MRF23',
	'M224DO': 'MRF24',
	'M225DO': 'MRF25',
	'M226DO': 'MRG26',
	'M227DO': 'MRG27',
	'M228DO': 'MRG28',
	'M229DO': 'MRG29',
}


# =========================
# Regex / matching rules
# =========================
_MONTH_COL_RE = re.compile(r'^\d{4}-\d{2}$')
_RULE_A_RE = re.compile(r'^N\.(\d)(\d{2})S$')
_RULE_B_RE = re.compile(r'^([A-Z]{1,4})(\d)([A-Z0-9]{2,})$')
_DONET_RE = re.compile(r'^M(\.\d{2}DO|\d{3}DO)$')


# =========================
# Utilities
# =========================


def normalize_code(x: object) -> str:
	if pd.isna(x):
		return ''
	s = unicodedata.normalize('NFKC', str(x))
	s = re.sub(r'\s+', '', s.strip())
	return s.upper()


def parse_date_series(s: pd.Series) -> pd.Series:
	# station.csv uses YYYY/MM/DD (some blank)
	ss = s.astype(str).str.strip().replace({'': pd.NA, 'nan': pd.NA})
	return pd.to_datetime(ss, errors='coerce')


def haversine_km(
	lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
	r = 6371.0
	lat1r = np.radians(lat1)
	lon1r = np.radians(lon1)
	lat2r = np.radians(lat2)
	lon2r = np.radians(lon2)
	dlat = lat2r - lat1r
	dlon = lon2r - lon1r
	a = (
		np.sin(dlat / 2.0) ** 2
		+ np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
	)
	c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
	return r * c


def months_between_inclusive(start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> set[str]:
	if pd.isna(start_dt) or pd.isna(end_dt) or end_dt < start_dt:
		return set()
	start = pd.Timestamp(year=start_dt.year, month=start_dt.month, day=1)
	end = pd.Timestamp(year=end_dt.year, month=end_dt.month, day=1)
	cur = start
	out: set[str] = set()
	while cur <= end:
		out.add(f'{cur.year:04d}-{cur.month:02d}')
		cur = cur + pd.offsets.MonthBegin(1)
	return out


def pick_preferred_network_code(codes: Sequence[str]) -> str:
	codes_norm = [normalize_code(c) for c in codes if normalize_code(c)]
	if not codes_norm:
		return ''
	digits = [c for c in codes_norm if c.isdigit()]
	if digits:
		return sorted((int(c), c) for c in digits)[0][1]
	return sorted(codes_norm)[0]


def presence_months_from_row(row: pd.Series, months: Sequence[str]) -> set[str]:
	present: set[str] = set()
	for m in months:
		v = row.get(m, np.nan)
		if pd.isna(v):
			continue
		if isinstance(v, (int, np.integer, float, np.floating)):
			if float(v) == 1.0:
				present.add(m)
		elif str(v).strip() == '1':
			present.add(m)
	return present


def load_i95_exclude_codes(path: Path) -> set[str]:
	if not path.exists():
		return set()
	txt = path.read_text(encoding='utf-8', errors='strict')
	codes: set[str] = set()
	for line in txt.splitlines():
		line = line.strip()
		if not line:
			continue
		first = line.split('\t')[0].strip()
		c = normalize_code(first)
		# robust: J + alnum length >= 4 (covers JNAHA etc.)
		if re.fullmatch(r'J[A-Z0-9]{3,}', c):
			codes.add(c)
	return codes


# =========================
# Candidate rules (priority updated)
# =========================

# Priority constants
PRIO = {
	'identity': 0,
	'donet': 10,
	'rule_a': 11,
	'rule_c': 20,
	'rule_d': 21,
	'b_drop_digit': 30,
	'b_move_digit_tail': 31,
	'b_move_digit_tail_us': 32,
	'b_add_tail_1': 33,
}


@dataclass(frozen=True)
class Candidate:
	code: str
	rule: str
	priority: int


def cand_identity(code: str) -> Candidate:
	return Candidate(code=code, rule='0:identity', priority=PRIO['identity'])


def cand_donet(code: str) -> Candidate | None:
	if not _DONET_RE.match(code):
		return None
	jma = DONET_TO_JMA.get(code, '')
	if not jma:
		return None
	return Candidate(
		code=f'M.{jma}', rule='J:DONET(table)->M.<JMA>', priority=PRIO['donet']
	)


def cand_rule_a(code: str) -> Candidate | None:
	m = _RULE_A_RE.match(code)
	if not m:
		return None
	d = m.group(1)
	nn = m.group(2)
	return Candidate(
		code=f'N.S{d}N{nn}', rule='A:N.DDDS->N.SdNnn', priority=PRIO['rule_a']
	)


def cand_rule_c(code: str) -> Candidate | None:
	if not code.startswith('E.'):
		return None
	return Candidate(code='TT.' + code[2:], rule='C:E->TT', priority=PRIO['rule_c'])


def cand_rule_d(code: str) -> Candidate | None:
	if not code.startswith('LG.'):
		return None
	return Candidate(code='AO.' + code[3:], rule='D:LG->AO', priority=PRIO['rule_d'])


def cand_rule_b_variants(code: str) -> list[Candidate]:
	"""Prefix + digit + rest variants:
	- drop digit:  DP2AMJ -> DP.AMJ
	- move digit:  DP2MZT -> DP.MZT2 / DP.MZT_2
	"""
	out: list[Candidate] = []
	m = _RULE_B_RE.match(code)
	if not m:
		return out
	prefix, digit, rest = m.group(1), m.group(2), m.group(3)

	out.append(
		Candidate(
			code=f'{prefix}.{rest}',
			rule='B:drop_digit(prefix+digit+rest->prefix.rest)',
			priority=PRIO['b_drop_digit'],
		)
	)
	out.append(
		Candidate(
			code=f'{prefix}.{rest}{digit}',
			rule='B:move_digit(prefix+digit+rest->prefix.restdigit)',
			priority=PRIO['b_move_digit_tail'],
		)
	)
	out.append(
		Candidate(
			code=f'{prefix}.{rest}_{digit}',
			rule='B:move_digit(prefix+digit+rest->prefix.rest_digit)',
			priority=PRIO['b_move_digit_tail_us'],
		)
	)
	return out


def cand_add_tail_1(code: str) -> Candidate | None:
	if not code or code[-1].isdigit():
		return None
	return Candidate(
		code=f'{code}1', rule='B:add_tail_1(code->code1)', priority=PRIO['b_add_tail_1']
	)


def build_candidates(mea_code_norm: str) -> list[Candidate]:
	cands: list[Candidate] = [cand_identity(mea_code_norm)]

	cj = cand_donet(mea_code_norm)
	if cj is not None:
		cands.append(cj)

	ca = cand_rule_a(mea_code_norm)
	if ca is not None:
		cands.append(ca)

	cc = cand_rule_c(mea_code_norm)
	if cc is not None:
		cands.append(cc)

	cd = cand_rule_d(mea_code_norm)
	if cd is not None:
		cands.append(cd)

	cands.extend(cand_rule_b_variants(mea_code_norm))

	cb = cand_add_tail_1(mea_code_norm)
	if cb is not None:
		cands.append(cb)

	cands.sort(key=lambda x: x.priority)
	return cands


# =========================
# Load and prepare references
# =========================
def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	mea = pd.read_csv(MEA_CSV)
	sta = pd.read_csv(STA_CSV)
	ch = pd.read_csv(CH_CSV)

	validate_columns(mea, [MEA_STATION_COL], 'station_measurement.csv')
	validate_columns(
		sta,
		[STA_STATION_COL, STA_LAT_COL, STA_LON_COL, STA_FROM_COL, STA_TO_COL],
		'station.csv',
	)
	validate_columns(
		ch,
		[CH_STATION_COL, CH_NETWORK_COL, CH_LAT_COL, CH_LON_COL],
		'monthly_presence_update.csv',
	)
	return mea, sta, ch


def prepare_sta_rep(sta: pd.DataFrame) -> pd.DataFrame:
	sta = sta.copy()
	sta['station_norm'] = sta[STA_STATION_COL].map(normalize_code)
	sta['_lat'] = pd.to_numeric(sta[STA_LAT_COL], errors='coerce')
	sta['_lon'] = pd.to_numeric(sta[STA_LON_COL], errors='coerce')
	sta['_from'] = parse_date_series(sta[STA_FROM_COL])
	sta['_to'] = parse_date_series(sta[STA_TO_COL])

	agg_dict = {
		'sta_lat': ('_lat', 'mean'),
		'sta_lon': ('_lon', 'mean'),
		'sta_from': ('_from', 'min'),
		'sta_to': ('_to', 'max'),
	}
	sta_rep = sta.groupby('station_norm', as_index=False).agg(**agg_dict)
	return sta_rep


def sta_codes_to_exclude_by_to(sta_rep: pd.DataFrame, cutoff: pd.Timestamp) -> set[str]:
	out: set[str] = set()
	for _, r in sta_rep.iterrows():
		code = str(r['station_norm'])
		to_dt = r['sta_to']
		if pd.isna(to_dt):
			continue
		if pd.Timestamp(to_dt) < cutoff:
			out.add(code)
	return out


def prepare_ch(
	ch: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], set[str], pd.DataFrame, dict[str, set[str]]]:
	ch = ch.copy()
	ch['station_norm'] = ch[CH_STATION_COL].map(normalize_code)
	ch['network_norm'] = ch[CH_NETWORK_COL].map(normalize_code)

	mcols = month_columns(ch)
	station_set = set(ch['station_norm'].unique().tolist())

	ch_station_rep = ch.groupby('station_norm', as_index=False).agg(
		ch_lat=(CH_LAT_COL, 'mean'), ch_lon=(CH_LON_COL, 'mean')
	)

	station_presence: dict[str, set[str]] = {}
	for st, sub in ch.groupby('station_norm'):
		u: set[str] = set()
		for _, row in sub.iterrows():
			u |= presence_months_from_row(row, mcols)
		station_presence[st] = u

	return ch, mcols, station_set, ch_station_rep, station_presence


def build_station_month_preferred_network(
	ch: pd.DataFrame, month_cols: list[str]
) -> dict[tuple[str, str], str]:
	"""For each (station_norm, month), among networks with presence==1, choose preferred network by:
	digits-only -> smallest integer, else lexicographically smallest
	"""
	out: dict[tuple[str, str], str] = {}
	for st, sub in ch.groupby('station_norm'):
		for m in month_cols:
			v = sub[m].to_numpy()
			idx = np.where(v == 1)[0]
			if idx.size == 0:
				continue
			cand = [sub.iloc[int(i)]['network_norm'] for i in idx.tolist()]
			chosen = pick_preferred_network_code(cand)
			if chosen:
				out[(st, m)] = chosen
	return out


# =========================
# Exclusions and matching
# =========================
def apply_exclusions(
	mea_codes_norm: Sequence[str],
	i95_exclude: set[str],
	earthscope_exclude: set[str],
	sta_to_exclude: set[str],
) -> tuple[list[str], pd.DataFrame]:
	kept: list[str] = []
	rows: list[dict[str, str]] = []
	for c in mea_codes_norm:
		if c in i95_exclude:
			rows.append({'mea_norm': c, 'exclude_reason': 'exclude:I95'})
			continue
		if c in earthscope_exclude:
			rows.append({'mea_norm': c, 'exclude_reason': 'exclude:EarthScope'})
			continue
		if c in sta_to_exclude:
			rows.append({'mea_norm': c, 'exclude_reason': 'exclude:sta_To_lt_20040401'})
			continue
		kept.append(c)
	return kept, pd.DataFrame(rows)


def match_mea_codes(
	mea_codes_norm: Sequence[str],
	ch_station_set: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Returns:
	- mapping_report: one row per mea_norm, with best match + all hits concatenated
	- mapping_candidates: long format (one row per candidate hit)

	"""
	report_rows: list[dict[str, object]] = []
	cand_rows: list[dict[str, object]] = []

	for mc in mea_codes_norm:
		cands = build_candidates(mc)

		hits: list[Candidate] = []
		for cand in cands:
			if cand.code not in ch_station_set:
				continue

			# conditional C/D: don't convert if original exists
			if cand.rule == 'C:E->TT' and mc in ch_station_set:
				continue
			if cand.rule == 'D:LG->AO' and mc in ch_station_set:
				continue

			hits.append(cand)

		hits.sort(key=lambda x: x.priority)

		if hits:
			best = hits[0]
			report_rows.append(
				{
					'mea_norm': mc,
					'match_status': 'matched',
					'best_ch_station_norm': best.code,
					'best_rule': best.rule,
					'n_hits': len(hits),
					'all_hit_stations_norm': '|'.join([h.code for h in hits]),
					'all_hit_rules': '|'.join([h.rule for h in hits]),
				}
			)
			for h in hits:
				cand_rows.append(
					{
						'mea_norm': mc,
						'ch_station_norm': h.code,
						'rule': h.rule,
						'priority': h.priority,
					}
				)
		else:
			report_rows.append(
				{
					'mea_norm': mc,
					'match_status': 'unmatched',
					'best_ch_station_norm': '',
					'best_rule': '',
					'n_hits': 0,
					'all_hit_stations_norm': '',
					'all_hit_rules': '',
				}
			)

	report = (
		pd.DataFrame(report_rows)
		.sort_values(['match_status', 'mea_norm'])
		.reset_index(drop=True)
	)
	cands_df = (
		pd.DataFrame(cand_rows)
		.sort_values(['mea_norm', 'priority', 'ch_station_norm'])
		.reset_index(drop=True)
	)
	return report, cands_df


def attach_sta_info(report: pd.DataFrame, sta_rep: pd.DataFrame) -> pd.DataFrame:
	m = report.merge(sta_rep, how='left', left_on='mea_norm', right_on='station_norm')
	m.drop(columns=['station_norm'], inplace=True)
	return m


def attach_best_ch_info(
	report: pd.DataFrame, ch_station_rep: pd.DataFrame
) -> pd.DataFrame:
	ch_station_rep2 = ch_station_rep.rename(
		columns={'station_norm': 'best_ch_station_norm'}
	)
	m = report.merge(ch_station_rep2, how='left', on='best_ch_station_norm')
	m.rename(columns={'ch_lat': 'best_ch_lat', 'ch_lon': 'best_ch_lon'}, inplace=True)
	return m


def attach_best_distance_km(report: pd.DataFrame) -> pd.DataFrame:
	m = report.copy()
	slat = pd.to_numeric(m['sta_lat'], errors='coerce').to_numpy()
	slon = pd.to_numeric(m['sta_lon'], errors='coerce').to_numpy()
	clat = pd.to_numeric(m['best_ch_lat'], errors='coerce').to_numpy()
	clon = pd.to_numeric(m['best_ch_lon'], errors='coerce').to_numpy()

	dist = np.full(len(m), np.nan, dtype=float)
	ok = (~np.isnan(slat)) & (~np.isnan(slon)) & (~np.isnan(clat)) & (~np.isnan(clon))
	if ok.any():
		dist[ok] = haversine_km(slat[ok], slon[ok], clat[ok], clon[ok])
	m['best_distance_km'] = dist
	return m


def nearest_ch_for_unmatched(
	report: pd.DataFrame, ch_station_rep: pd.DataFrame
) -> pd.DataFrame:
	u = report.loc[report['match_status'] == 'unmatched'].copy()
	if u.empty:
		return u

	ch_lat = pd.to_numeric(ch_station_rep['ch_lat'], errors='coerce').to_numpy()
	ch_lon = pd.to_numeric(ch_station_rep['ch_lon'], errors='coerce').to_numpy()
	ch_st = ch_station_rep['station_norm'].astype(str).to_numpy()

	sta_lat = pd.to_numeric(u['sta_lat'], errors='coerce').to_numpy()
	sta_lon = pd.to_numeric(u['sta_lon'], errors='coerce').to_numpy()

	nearest_station: list[str] = []
	nearest_dist: list[float] = []

	for i in range(len(u)):
		lat0 = sta_lat[i]
		lon0 = sta_lon[i]
		if np.isnan(lat0) or np.isnan(lon0):
			nearest_station.append('')
			nearest_dist.append(np.nan)
			continue

		best_d = float('inf')
		best_j = -1
		for j0 in range(0, len(ch_station_rep), NEAREST_CH_CHUNK):
			j1 = min(len(ch_station_rep), j0 + NEAREST_CH_CHUNK)
			d = haversine_km(
				np.full(j1 - j0, lat0, dtype=float),
				np.full(j1 - j0, lon0, dtype=float),
				ch_lat[j0:j1].astype(float),
				ch_lon[j0:j1].astype(float),
			)
			k = int(np.argmin(d))
			dk = float(d[k])
			if dk < best_d:
				best_d = dk
				best_j = j0 + k

		nearest_station.append(str(ch_st[best_j]))
		nearest_dist.append(float(best_d))

	u['nearest_ch_station_norm'] = nearest_station
	u['nearest_distance_km'] = nearest_dist
	return u.sort_values(['nearest_distance_km', 'mea_norm']).reset_index(drop=True)


def near0_suggestions(
	unmatched_nearest: pd.DataFrame,
	sta_rep: pd.DataFrame,
	station_presence: dict[str, set[str]],
) -> pd.DataFrame:
	if not ENABLE_NEAR0_SUGGESTIONS or unmatched_nearest.empty:
		return pd.DataFrame()

	sta_from = dict(zip(sta_rep['station_norm'].tolist(), sta_rep['sta_from'].tolist()))
	sta_to = dict(zip(sta_rep['station_norm'].tolist(), sta_rep['sta_to'].tolist()))

	rows: list[dict[str, object]] = []
	for _, r in unmatched_nearest.iterrows():
		mc = str(r['mea_norm'])
		st = str(r.get('nearest_ch_station_norm', ''))
		d = r.get('nearest_distance_km', np.nan)
		if not st or pd.isna(d) or float(d) > NEAR0_KM:
			continue

		sfrom = sta_from.get(mc, pd.NaT)
		sto = sta_to.get(mc, pd.NaT)
		if pd.isna(sfrom):
			continue

		end_dt = sto if not pd.isna(sto) else pd.Timestamp('2025-12-31')
		sta_months = months_between_inclusive(pd.Timestamp(sfrom), pd.Timestamp(end_dt))
		pres = station_presence.get(st, set())
		overlap = sorted(list(sta_months & pres)) if sta_months and pres else []
		if not overlap:
			continue

		rows.append(
			{
				'mea_norm': mc,
				'suggest_ch_station_norm': st,
				'nearest_distance_km': float(d),
				'overlap_months_n': len(overlap),
				'overlap_first': overlap[0],
				'overlap_last': overlap[-1],
				'suggest_strength': 'near0_with_period_overlap',
			}
		)

	return (
		pd.DataFrame(rows)
		.sort_values(['nearest_distance_km', 'mea_norm'])
		.reset_index(drop=True)
	)


def classify_multi_candidates_by_presence(
	mapping_candidates: pd.DataFrame,
	station_presence: dict[str, set[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""For mea_norm with >=2 candidates, compare station-level presence month sets (union across networks).
	- disjoint: all pairs have empty intersection
	- overlap: any pair intersects
	"""
	if mapping_candidates.empty:
		return pd.DataFrame(), pd.DataFrame()

	g = mapping_candidates.groupby('mea_norm')['ch_station_norm'].apply(list).to_dict()

	dis_rows: list[dict[str, object]] = []
	ov_rows: list[dict[str, object]] = []

	for mc, stations in g.items():
		if len(stations) < 2:
			continue

		pres_list = [(st, station_presence.get(st, set())) for st in stations]
		any_overlap = False
		for i in range(len(pres_list)):
			for j in range(i + 1, len(pres_list)):
				if pres_list[i][1] & pres_list[j][1]:
					any_overlap = True
					break
			if any_overlap:
				break

		first_last = []
		for st, pres in pres_list:
			if pres:
				ms = sorted(pres)
				first_last.append((st, ms[0], ms[-1], len(ms)))
			else:
				first_last.append((st, '', '', 0))

		row = {
			'mea_norm': mc,
			'candidates_norm': '|'.join([x[0] for x in first_last]),
			'presence_first': '|'.join([x[1] for x in first_last]),
			'presence_last': '|'.join([x[2] for x in first_last]),
			'n_present_months': '|'.join([str(x[3]) for x in first_last]),
		}

		if any_overlap:
			ov_rows.append(row)
		else:
			dis_rows.append(row)

	dis = pd.DataFrame(dis_rows).sort_values(['mea_norm']).reset_index(drop=True)
	ov = pd.DataFrame(ov_rows).sort_values(['mea_norm']).reset_index(drop=True)
	return dis, ov


def export_outputs(
	report: pd.DataFrame,
	mapping_candidates: pd.DataFrame,
	excluded: pd.DataFrame,
	unmatched_nearest: pd.DataFrame,
	near0_df: pd.DataFrame,
	disjoint: pd.DataFrame,
	overlap: pd.DataFrame,
	st_month_prefnet: dict[tuple[str, str], str],
) -> None:
	report.to_csv(OUT_DIR / 'mapping_report.csv', index=False)

	if not mapping_candidates.empty:
		mapping_candidates.to_csv(OUT_DIR / 'mapping_candidates.csv', index=False)

	excluded.to_csv(OUT_DIR / 'excluded.csv', index=False)

	unmatched_nearest.to_csv(OUT_DIR / 'unmatched_nearest.csv', index=False)

	if not near0_df.empty:
		near0_df.to_csv(OUT_DIR / 'near0_suggestions.csv', index=False)

	if EXPORT_UTF8_BOM_CSV:
		report.to_csv(
			OUT_DIR / 'mapping_report_utf8bom.csv', index=False, encoding='utf-8-sig'
		)
		if not mapping_candidates.empty:
			mapping_candidates.to_csv(
				OUT_DIR / 'mapping_candidates_utf8bom.csv',
				index=False,
				encoding='utf-8-sig',
			)
		excluded.to_csv(
			OUT_DIR / 'excluded_utf8bom.csv', index=False, encoding='utf-8-sig'
		)
		unmatched_nearest.to_csv(
			OUT_DIR / 'unmatched_nearest_utf8bom.csv', index=False, encoding='utf-8-sig'
		)
		if not near0_df.empty:
			near0_df.to_csv(
				OUT_DIR / 'near0_suggestions_utf8bom.csv',
				index=False,
				encoding='utf-8-sig',
			)

	if EXPORT_CP932_CSV:
		report.to_csv(
			OUT_DIR / 'mapping_report_cp932.csv', index=False, encoding='cp932'
		)
		if not mapping_candidates.empty:
			mapping_candidates.to_csv(
				OUT_DIR / 'mapping_candidates_cp932.csv', index=False, encoding='cp932'
			)
		excluded.to_csv(OUT_DIR / 'excluded_cp932.csv', index=False, encoding='cp932')
		unmatched_nearest.to_csv(
			OUT_DIR / 'unmatched_nearest_cp932.csv', index=False, encoding='cp932'
		)
		if not near0_df.empty:
			near0_df.to_csv(
				OUT_DIR / 'near0_suggestions_cp932.csv', index=False, encoding='cp932'
			)

	if EXPORT_XLSX:
		pref_rows = [
			{'station_norm': st, 'month': m, 'preferred_network_norm': net}
			for (st, m), net in st_month_prefnet.items()
		]
		df_pref = (
			pd.DataFrame(pref_rows)
			.sort_values(['station_norm', 'month'])
			.reset_index(drop=True)
		)

		with pd.ExcelWriter(OUT_DIR / 'mapping_outputs.xlsx', engine='openpyxl') as w:
			report.to_excel(w, index=False, sheet_name='mapping_report')
			mapping_candidates.to_excel(w, index=False, sheet_name='mapping_candidates')
			excluded.to_excel(w, index=False, sheet_name='excluded')
			unmatched_nearest.to_excel(w, index=False, sheet_name='unmatched_nearest')
			if not near0_df.empty:
				near0_df.to_excel(w, index=False, sheet_name='near0_suggestions')
			disjoint.to_excel(w, index=False, sheet_name='multi_disjoint')
			overlap.to_excel(w, index=False, sheet_name='multi_overlap')
			df_pref.to_excel(w, index=False, sheet_name='pref_network_by_month')


def main() -> None:
	mea, sta, ch = load_inputs()

	mea_codes_norm = sorted(
		{
			normalize_code(x)
			for x in mea[MEA_STATION_COL].dropna().astype(str).tolist()
			if normalize_code(x)
		}
	)

	sta_rep = prepare_sta_rep(sta)
	sta_to_exclude = sta_codes_to_exclude_by_to(sta_rep, CH_START_DATE)

	i95_exclude = load_i95_exclude_codes(I95_TXT) if I95_TXT is not None else set()
	earthscope_exclude = {normalize_code(x) for x in EARTHSCOPE_CODES_EXCLUDE}

	kept, excluded_df = apply_exclusions(
		mea_codes_norm, i95_exclude, earthscope_exclude, sta_to_exclude
	)

	ch_all, month_cols, ch_station_set, ch_station_rep, station_presence = prepare_ch(
		ch
	)
	st_month_prefnet = build_station_month_preferred_network(ch_all, month_cols)

	report_base, mapping_candidates = match_mea_codes(kept, ch_station_set)
	report = attach_sta_info(report_base, sta_rep)
	report = attach_best_ch_info(report, ch_station_rep)
	report = attach_best_distance_km(report)

	unmatched_nearest = nearest_ch_for_unmatched(report, ch_station_rep)
	near0_df = near0_suggestions(unmatched_nearest, sta_rep, station_presence)

	disjoint, overlap = classify_multi_candidates_by_presence(
		mapping_candidates, station_presence
	)

	export_outputs(
		report=report,
		mapping_candidates=mapping_candidates,
		excluded=excluded_df,
		unmatched_nearest=unmatched_nearest,
		near0_df=near0_df,
		disjoint=disjoint,
		overlap=overlap,
		st_month_prefnet=st_month_prefnet,
	)

	n_total = len(report)
	n_matched = int((report['match_status'] == 'matched').sum())
	n_unmatched = int((report['match_status'] == 'unmatched').sum())
	print(f'[OK] total={n_total} matched={n_matched} unmatched={n_unmatched}')
	print(f'[OUT] {OUT_DIR}')


if __name__ == '__main__':
	main()
