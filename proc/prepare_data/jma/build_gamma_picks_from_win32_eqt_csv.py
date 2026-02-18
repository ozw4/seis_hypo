# %%
"""Build one GaMMA picks CSV from one or more WIN32 EqT pick CSV files."""

# file: proc/prepare_data/jma/build_gamma_picks_from_win32_eqt_csv.py
#
# Purpose:
# - Convert one or more WIN32 continuous EqT pick CSVs into one GaMMA picks CSV.
#
# Input CSV required columns:
#   station_code, Phase, pick_time, w_conf, network_code
#
# Output GaMMA core columns:
#   station_id  : station key (default '{network_code}.{station_code}')
#   phase_time  : UTC ISO8601 with trailing 'Z'
#   phase_type  : 'P' or 'S'
#   phase_score : EqT confidence (copied from w_conf)

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import pandas as pd

from common.core import validate_columns

# =========================
# Parameters (edit here)
# =========================
IN_PICK_CSVS = [
	Path(
		'/workspace/proc/run_continuous/pick/jma0301/proc/run_continuous/pick/win32/out/eqt_picks_win32_0101.csv'
	),
	Path(
		'/workspace/proc/run_continuous/pick/jma0301/proc/run_continuous/pick/win32/out/eqt_picks_win32_0203.csv'
	),
	Path(
		'/workspace/proc/run_continuous/pick/jma0301/proc/run_continuous/pick/win32/out/eqt_picks_win32_0207.csv'
	),
	Path(
		'/workspace/proc/run_continuous/pick/jma0301/proc/run_continuous/pick/win32/out/eqt_picks_win32_0301.csv'
	),
]

OUT_GAMMA_PICKS_CSV = Path(
	'/workspace/proc/run_continuous/association/jma/out/gamma_picks.csv'
)

# station_id mode:
# - 'network_station': '{network_code}.{station_code}' (default)
# - 'station_only'   : '{station_code}' (collision across networks is an error)
STATION_ID_MODE = 'network_station'

OUT_COLUMNS = ['station_id', 'phase_time', 'phase_type', 'phase_score']

INCLUDE_TRACE_COLUMNS = True
TRACE_COLUMNS = [
	'network_code',
	'station_code',
	'pick_time',
	'source_csv',
]

PHASE_MAP = {'P': 'P', 'S': 'S'}
PHASE_TIME_FMT = '%Y-%m-%dT%H:%M:%S.%fZ'
_JST = dt.timezone(dt.timedelta(hours=9))
_TZ_SUFFIX_RE = re.compile(r'(?:Z|[+-]\d{2}:\d{2})$')


def _normalize_string_column(df: pd.DataFrame, col: str, label: str) -> None:
	df[col] = df[col].astype('string').str.strip()
	if df[col].isna().any() or (df[col] == '').any():
		raise ValueError(f'{label} contains empty values in column={col}')


def _load_pick_csv(csv_path: Path) -> pd.DataFrame:
	df = pd.read_csv(
		csv_path,
		dtype={
			'station_code': 'string',
			'Phase': 'string',
			'pick_time': 'string',
			'network_code': 'string',
		},
	)
	validate_columns(
		df,
		['station_code', 'Phase', 'pick_time', 'w_conf', 'network_code'],
		f'WIN32 EqT pick CSV: {csv_path}',
	)

	df = df.copy()
	_normalize_string_column(df, 'station_code', f'WIN32 EqT pick CSV: {csv_path}')
	_normalize_string_column(df, 'Phase', f'WIN32 EqT pick CSV: {csv_path}')
	_normalize_string_column(df, 'pick_time', f'WIN32 EqT pick CSV: {csv_path}')
	_normalize_string_column(df, 'network_code', f'WIN32 EqT pick CSV: {csv_path}')

	df['Phase'] = df['Phase'].str.upper()
	df['w_conf'] = pd.to_numeric(df['w_conf'], errors='raise').astype('float64')
	if df['w_conf'].isna().any():
		raise ValueError(f'WIN32 EqT pick CSV has NaN w_conf: {csv_path}')

	df['source_csv'] = str(csv_path)
	return df


def _pick_time_jst_to_utc_iso8601_z(pick_time: pd.Series) -> pd.Series:
	if pick_time.empty:
		return pd.Series(index=pick_time.index, dtype='string')

	has_tz = pick_time.str.contains(_TZ_SUFFIX_RE, regex=True)
	if bool(has_tz.all()):
		ts_utc = pd.to_datetime(pick_time, utc=True, errors='raise')
	elif bool((~has_tz).all()):
		# If timezone offset is absent, treat values as JST by contract.
		ts_naive = pd.to_datetime(pick_time, errors='raise')
		ts_utc = ts_naive.dt.tz_localize(_JST).dt.tz_convert(dt.timezone.utc)
	else:
		examples = pick_time.loc[has_tz != has_tz.iloc[0]].head(5).tolist()
		raise ValueError(
			'pick_time timezone format is mixed (offset-aware and offset-naive). '
			f'examples={examples}'
		)

	return ts_utc.dt.strftime(PHASE_TIME_FMT)


def _build_station_id(picks: pd.DataFrame) -> pd.Series:
	if STATION_ID_MODE == 'network_station':
		return picks['network_code'] + '.' + picks['station_code']

	if STATION_ID_MODE == 'station_only':
		net_per_station = (
			picks[['station_code', 'network_code']]
			.drop_duplicates()
			.groupby('station_code', as_index=False)['network_code']
			.nunique()
		)
		conflict = net_per_station.loc[net_per_station['network_code'] > 1]
		if not conflict.empty:
			examples = conflict['station_code'].head(20).tolist()
			raise ValueError(
				'station_only mode causes station_id collisions across network_code. '
				f'conflicting station_code examples={examples}'
			)
		return picks['station_code'].copy()

	raise ValueError(
		'STATION_ID_MODE must be '
		f"'network_station' or 'station_only', got {STATION_ID_MODE}"
	)


def build_gamma_picks_from_win32_eqt_csv(pick_csvs: list[Path]) -> pd.DataFrame:
	"""Merge WIN32 EqT picks and convert them to GaMMA picks schema."""
	if not pick_csvs:
		raise ValueError('pick_csvs is empty')

	parts = [_load_pick_csv(Path(p)) for p in pick_csvs]
	picks = pd.concat(parts, axis=0, ignore_index=True)

	if picks.empty:
		raise ValueError('no pick rows found in input CSVs')

	picks = picks.copy()
	picks['phase_type'] = picks['Phase'].map(PHASE_MAP)
	if picks['phase_type'].isna().any():
		bad = (
			picks.loc[picks['phase_type'].isna(), 'Phase']
			.astype(str)
			.drop_duplicates()
			.tolist()
		)
		raise ValueError(f'Unexpected Phase values. Only P/S are allowed, got={bad}')

	picks['phase_time'] = _pick_time_jst_to_utc_iso8601_z(picks['pick_time'])
	picks['phase_score'] = picks['w_conf'].astype('float64')
	picks['station_id'] = _build_station_id(picks)

	core = picks[OUT_COLUMNS].copy()

	if INCLUDE_TRACE_COLUMNS:
		validate_columns(picks, TRACE_COLUMNS, 'trace columns source')
		out = pd.concat([core, picks[TRACE_COLUMNS].copy()], axis=1)
	else:
		out = core

	return out.sort_values(
		['phase_time', 'station_id', 'phase_type'], kind='mergesort'
	).reset_index(drop=True)


def main() -> None:
	"""Run conversion using top-of-file constants."""
	if not IN_PICK_CSVS:
		raise ValueError('IN_PICK_CSVS is empty')

	csv_paths = [Path(p) for p in IN_PICK_CSVS]
	if len(set(csv_paths)) != len(csv_paths):
		raise ValueError('IN_PICK_CSVS contains duplicated paths')

	missing = [p for p in csv_paths if not p.is_file()]
	if missing:
		raise FileNotFoundError(f'input pick CSV not found: {missing}')

	OUT_GAMMA_PICKS_CSV.parent.mkdir(parents=True, exist_ok=True)

	out = build_gamma_picks_from_win32_eqt_csv(csv_paths)
	validate_columns(out, OUT_COLUMNS, 'output GaMMA picks CSV')

	out.to_csv(OUT_GAMMA_PICKS_CSV, index=False)

	print('Wrote:', OUT_GAMMA_PICKS_CSV)
	print('Rows:', int(out.shape[0]))
	print('Time range:', out['phase_time'].iloc[0], '->', out['phase_time'].iloc[-1])
	print('Stations:', int(out['station_id'].nunique()))


if __name__ == '__main__':
	main()

# 実行例:
# export PYTHONPATH="$PWD/src"
# python proc/prepare_data/jma/build_gamma_picks_from_win32_eqt_csv.py
# 入力CSV例:
#   /workspace/proc/run_continuous/pick/win32/out/eqt_picks_win32_0101.csv
#   /workspace/proc/run_continuous/pick/win32/out/eqt_picks_win32_0203.csv
