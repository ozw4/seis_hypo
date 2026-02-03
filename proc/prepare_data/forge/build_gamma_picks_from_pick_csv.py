# %%
# file: proc/prepare_data/forge/build_gamma_picks_from_pick_csv.py
#
# Purpose:
# - Convert continuous pick CSV (EqT/PhaseNet pipeline output) to GaMMA picks CSV
# - Join station_id using forge_das_station_metadata.csv (channel -> station_id)
#
# Input (pick CSV expected columns):
#   segment_id, block_start, channel, phase, pick_time_utc_ms, pick_time_utc_iso, prob
#
# Input (station metadata expected columns):
#   channel, station_id, ... (others are ignored)
#
# Output (GaMMA picks CSV default columns):
#   station_id, phase_time, phase_type, phase_score
# where:
#   phase_time  : ISO8601 UTC string with 'Z' (e.g., 2022-04-17T10:59:56.202389Z)
#   phase_type  : 'P' or 'S'
#   phase_score : float (probability)

from __future__ import annotations

from pathlib import Path

import pandas as pd

from common.core import validate_columns

# =========================
# Parameters (edit here)
# =========================
IN_PICK_CSV = Path(
	'/workspace/proc/pick_continuous/forge/out/das_eqt_picks_woconvert.csv'
)
IN_STATION_META_CSV = Path(
	'/workspace/data/station/forge/forge_das_station_metadata.csv'
)
OUT_GAMMA_PICKS_CSV = Path('./forge_gamma_picks.csv')

# Output schema: keep minimal for GaMMA
OUT_COLUMNS = ['station_id', 'phase_time', 'phase_type', 'phase_score']

# If True, include extra traceability columns after the core columns
INCLUDE_TRACE_COLUMNS = True
TRACE_COLUMNS = ['channel', 'segment_id', 'block_start', 'pick_time_utc_ms']

# Phase mapping (input 'phase' -> output 'phase_type')
PHASE_MAP = {'P': 'P', 'S': 'S'}

# Format for phase_time (ISO8601 UTC with Z). Microseconds kept.
PHASE_TIME_FMT = '%Y-%m-%dT%H:%M:%S.%fZ'


def _load_station_channel_map(station_csv: Path) -> pd.DataFrame:
	st = pd.read_csv(station_csv)
	validate_columns(st, ['channel', 'station_id'], f'station meta CSV: {station_csv}')

	st = st[['channel', 'station_id']].copy()
	st['channel'] = st['channel'].astype('int64')

	if st['channel'].duplicated().any():
		dups = (
			st.loc[st['channel'].duplicated(), 'channel'].astype(int).unique().tolist()
		)
		raise ValueError(
			f'station meta has duplicated channel rows (examples): {dups[:20]}'
		)

	if st['station_id'].isna().any():
		raise ValueError('station meta contains empty station_id')

	return st


def build_gamma_picks(pick_csv: Path, station_csv: Path) -> pd.DataFrame:
	picks = pd.read_csv(pick_csv)
	validate_columns(
		picks,
		[
			'segment_id',
			'block_start',
			'channel',
			'phase',
			'pick_time_utc_ms',
			'prob',
		],
		f'pick CSV: {pick_csv}',
	)

	picks = picks.copy()
	picks['channel'] = picks['channel'].astype('int64')
	picks['pick_time_utc_ms'] = picks['pick_time_utc_ms'].astype('int64')

	ch_map = _load_station_channel_map(station_csv)

	merged = picks.merge(ch_map, on='channel', how='left', validate='many_to_one')

	if merged['station_id'].isna().any():
		miss = (
			merged.loc[merged['station_id'].isna(), 'channel']
			.astype(int)
			.drop_duplicates()
			.sort_values()
			.tolist()
		)
		raise ValueError(
			'Found pick channels that do not exist in station meta. '
			f'Missing channels (examples): {miss[:40]}'
		)

	merged['phase_type'] = merged['phase'].map(PHASE_MAP)
	if merged['phase_type'].isna().any():
		bad = (
			merged.loc[merged['phase_type'].isna(), 'phase']
			.astype(str)
			.drop_duplicates()
			.tolist()
		)
		raise ValueError(f'Unexpected phase values in pick CSV: {bad}')

	# Create ISO8601 UTC time string for GaMMA
	phase_time_dt = pd.to_datetime(merged['pick_time_utc_ms'], unit='ms', utc=True)
	merged['phase_time'] = phase_time_dt.dt.strftime(PHASE_TIME_FMT)

	merged['phase_score'] = merged['prob'].astype('float64')

	core = merged[['station_id', 'phase_time', 'phase_type', 'phase_score']].copy()

	if INCLUDE_TRACE_COLUMNS:
		trace = merged[TRACE_COLUMNS].copy()
		out = pd.concat([core, trace], axis=1)
	else:
		out = core

	out = out.sort_values(
		['phase_time', 'station_id', 'phase_type'], kind='mergesort'
	).reset_index(drop=True)

	return out


def main() -> None:
	if not IN_PICK_CSV.exists():
		raise FileNotFoundError(f'Pick CSV not found: {IN_PICK_CSV}')
	if not IN_STATION_META_CSV.exists():
		raise FileNotFoundError(f'Station meta CSV not found: {IN_STATION_META_CSV}')

	OUT_GAMMA_PICKS_CSV.parent.mkdir(parents=True, exist_ok=True)

	out = build_gamma_picks(IN_PICK_CSV, IN_STATION_META_CSV)

	# Final sanity check: ensure core columns exist
	validate_columns(out, OUT_COLUMNS, 'output GaMMA picks DF')

	out.to_csv(OUT_GAMMA_PICKS_CSV, index=False)

	print('Wrote:', OUT_GAMMA_PICKS_CSV)
	print('Rows:', int(out.shape[0]))
	print('Time range:', out['phase_time'].iloc[0], '->', out['phase_time'].iloc[-1])
	print('Stations:', int(out['station_id'].nunique()))


if __name__ == '__main__':
	main()
