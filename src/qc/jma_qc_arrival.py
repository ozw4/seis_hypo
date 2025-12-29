# src/qc/jma_arrivetime_qc.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.core import validate_columns
from common.geo import haversine_distance_km

P_PHASES_DEFAULT: set[str] = {'P', 'EP', 'IP', 'PKP'}
S_PHASES_DEFAULT: set[str] = {'S', 'ES', 'IS'}


@dataclass(frozen=True)
class JmaArrivetimeQcArtifacts:
	out_dir: Path
	fig_dir: Path
	picks_csv: Path
	event_summary_csv: Path
	station_summary_csv: Path
	station_sp_csv: Path


def _parse_datetime_series(s: pd.Series, name: str) -> pd.Series:
	t = pd.to_datetime(s, errors='coerce')
	bad = t.isna() & s.notna()
	if bad.any():
		ex = s[bad].head(5).tolist()
		raise ValueError(f'unparseable datetime in {name}: {ex}')
	return t


def _phase_type(phase: str, p_phases: set[str], s_phases: set[str]) -> str:
	ph = str(phase).strip()
	if ph in p_phases:
		return 'P'
	if ph in s_phases:
		return 'S'
	return 'Other'


def read_arrivetime_measurements_csv(path: str | Path) -> pd.DataFrame:
	path = Path(path)
	if not path.is_file():
		raise FileNotFoundError(f'measurements csv not found: {path}')

	df = pd.read_csv(path)
	validate_columns(
		df,
		[
			'event_id',
			'station_code',
			'station_number',
			'sensor_type',
			'phase_name_1',
			'phase1_time',
			'phase_name_2',
			'phase2_time',
		],
		'measurements',
	)

	df = df.copy()
	df['phase1_time'] = _parse_datetime_series(df['phase1_time'], 'phase1_time')
	df['phase2_time'] = _parse_datetime_series(df['phase2_time'], 'phase2_time')
	df['station_code'] = df['station_code'].astype(str)
	df['station_number'] = df['station_number'].astype(int)
	return df


def read_arrivetime_epicenters_csv(path: str | Path) -> pd.DataFrame:
	path = Path(path)
	if not path.is_file():
		raise FileNotFoundError(f'epicenters csv not found: {path}')

	df = pd.read_csv(path)
	validate_columns(
		df,
		[
			'event_id',
			'origin_time',
			'latitude_deg',
			'longitude_deg',
			'depth_km',
			'mag1',
			'mag1_type',
			'tt_table',
			'station_count',
			'hypocenter_flag',
		],
		'epicenters',
	)

	df = df.copy()
	df['origin_time'] = _parse_datetime_series(df['origin_time'], 'origin_time')
	return df


def read_jma_station_csv(path: str | Path) -> pd.DataFrame:
	path = Path(path)
	if not path.is_file():
		raise FileNotFoundError(f'station csv not found: {path}')

	df = pd.read_csv(path)
	validate_columns(
		df,
		[
			'station_code',
			'station_number',
			'Latitude_deg',
			'Longitude_deg',
			'Height',
			'From',
			'To',
		],
		'station.csv',
	)

	df = df.copy()
	df['station_code'] = df['station_code'].astype(str)
	df['station_number'] = df['station_number'].astype(int)
	df['From_dt'] = pd.to_datetime(df['From'], errors='coerce')
	df['To_dt'] = pd.to_datetime(df['To'], errors='coerce')
	return df


def expand_measurements_to_picks(
	meas_df: pd.DataFrame,
	*,
	p_phases: set[str] = P_PHASES_DEFAULT,
	s_phases: set[str] = S_PHASES_DEFAULT,
) -> pd.DataFrame:
	p1 = meas_df[
		[
			'event_id',
			'station_code',
			'station_number',
			'sensor_type',
			'phase_name_1',
			'phase1_time',
		]
	].rename(columns={'phase_name_1': 'phase', 'phase1_time': 'time'})

	p2 = meas_df[
		[
			'event_id',
			'station_code',
			'station_number',
			'sensor_type',
			'phase_name_2',
			'phase2_time',
		]
	].rename(columns={'phase_name_2': 'phase', 'phase2_time': 'time'})

	picks = pd.concat([p1, p2], ignore_index=True)
	picks = picks.dropna(subset=['phase', 'time']).copy()
	picks['phase'] = picks['phase'].astype(str).str.strip()
	picks['phase_type'] = picks['phase'].map(
		lambda ph: _phase_type(ph, p_phases, s_phases)
	)
	picks['event_id'] = picks['event_id'].astype(int)
	picks['station_code'] = picks['station_code'].astype(str)
	picks['station_number'] = picks['station_number'].astype(int)
	return picks


def _filter_epicenters(
	epi_df: pd.DataFrame,
	*,
	exclude_hypocenter_flags: set[str],
) -> pd.DataFrame:
	if not exclude_hypocenter_flags:
		return epi_df.copy()

	mask = ~epi_df['hypocenter_flag'].astype(str).isin(exclude_hypocenter_flags)
	return epi_df.loc[mask].copy()


def _select_station_rows_for_time_range(
	station_df: pd.DataFrame,
	used_pairs: pd.DataFrame,
	*,
	tmin: pd.Timestamp,
	tmax: pd.Timestamp,
) -> pd.DataFrame:
	sta = station_df.copy()

	overlap = (sta['From_dt'].isna() | (sta['From_dt'] <= tmax)) & (
		sta['To_dt'].isna() | (sta['To_dt'] >= tmin)
	)
	sta = sta.loc[overlap].copy()
	sta = sta.merge(used_pairs, on=['station_code', 'station_number'], how='inner')

	if sta.empty:
		raise ValueError('no station rows matched the used station_code and time range')

	sta['_From_sort'] = sta['From_dt'].fillna(pd.Timestamp.min)
	sta = sta.sort_values(
		['station_code', 'station_number', '_From_sort'], kind='mergesort'
	)
	sta = sta.groupby(['station_code', 'station_number'], as_index=False).tail(1)
	sta = sta.drop(columns=['_From_sort'])

	dup = sta.duplicated(subset=['station_code', 'station_number'], keep=False)
	if dup.any():
		ex = (
			sta.loc[dup, ['station_code', 'station_number', 'From', 'To']]
			.head(10)
			.to_dict('records')
		)
		raise ValueError(f'station rows are ambiguous after filtering: {ex}')

	cols = ['station_code', 'station_number', 'Latitude_deg', 'Longitude_deg', 'Height']
	return sta[cols].copy()


def _attach_station_coords(
	picks: pd.DataFrame,
	station_df: pd.DataFrame,
	*,
	tmin: pd.Timestamp,
	tmax: pd.Timestamp,
) -> pd.DataFrame:
	used_pairs = picks[['station_code', 'station_number']].drop_duplicates()
	sta_rows = _select_station_rows_for_time_range(
		station_df, used_pairs, tmin=tmin, tmax=tmax
	)

	out = picks.merge(
		sta_rows, on=['station_code', 'station_number'], how='left', validate='m:1'
	)
	if out[['Latitude_deg', 'Longitude_deg']].isna().any().any():
		raise ValueError('station coordinate join failed for some rows')
	return out


def _compute_dist_km_by_event(df: pd.DataFrame) -> pd.Series:
	df = df.reset_index(drop=True)
	dist = np.empty(len(df), dtype=float)
	for _, g in df.groupby('event_id', sort=False):
		lat0 = float(g['latitude_deg'].iloc[0])
		lon0 = float(g['longitude_deg'].iloc[0])
		lat = g['Latitude_deg'].to_numpy(dtype=float)
		lon = g['Longitude_deg'].to_numpy(dtype=float)
		dist[g.index.to_numpy()] = haversine_distance_km(
			lat0_deg=lat0, lon0_deg=lon0, lat_deg=lat, lon_deg=lon
		)
	return pd.Series(dist, index=df.index)


def build_arrivetime_qc_tables(
	*,
	meas_csv: str | Path,
	epic_csv: str | Path,
	station_csv: str | Path,
	p_phases: set[str] = P_PHASES_DEFAULT,
	s_phases: set[str] = S_PHASES_DEFAULT,
	exclude_hypocenter_flags: set[str] = {'F'},
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	meas = read_arrivetime_measurements_csv(meas_csv)
	epi = read_arrivetime_epicenters_csv(epic_csv)
	sta = read_jma_station_csv(station_csv)

	picks = expand_measurements_to_picks(meas, p_phases=p_phases, s_phases=s_phases)
	epi_ok = _filter_epicenters(epi, exclude_hypocenter_flags=exclude_hypocenter_flags)

	meas_ids = set(meas['event_id'].unique())
	epi_ids = set(epi_ok['event_id'].unique())
	missing_in_meas = epi_ids - meas_ids
	if missing_in_meas:
		raise ValueError(f'event_id missing in measurements: {len(missing_in_meas)}')

	picks = picks.merge(
		epi_ok[
			[
				'event_id',
				'origin_time',
				'latitude_deg',
				'longitude_deg',
				'depth_km',
				'mag1',
				'mag1_type',
				'tt_table',
				'station_count',
				'hypocenter_flag',
			]
		],
		on='event_id',
		how='inner',
		validate='m:1',
	)

	picks['origin_time'] = pd.to_datetime(picks['origin_time'])
	picks['time'] = pd.to_datetime(picks['time'])
	picks['tt_s'] = (picks['time'] - picks['origin_time']).dt.total_seconds()
	if (picks['tt_s'] < 0).any():
		bad = picks.loc[
			picks['tt_s'] < 0,
			['event_id', 'station_code', 'phase', 'origin_time', 'time', 'tt_s'],
		].head(10)
		raise ValueError(f'negative travel time exists: {bad.to_dict("records")}')

	tmin = picks['origin_time'].min()
	tmax = picks['origin_time'].max()
	picks = _attach_station_coords(picks, sta, tmin=tmin, tmax=tmax)

	picks = picks.reset_index(drop=True)
	picks['dist_km'] = _compute_dist_km_by_event(picks).to_numpy(dtype=float)

	# event summary
	evt_counts = picks.pivot_table(
		index='event_id',
		columns='phase_type',
		values='phase',
		aggfunc='count',
		fill_value=0,
	)
	for c in ['P', 'S', 'Other']:
		if c not in evt_counts.columns:
			evt_counts[c] = 0
	evt_counts['total_picks'] = evt_counts[['P', 'S', 'Other']].sum(axis=1)
	evt_counts['n_stations_meas'] = picks.groupby('event_id')['station_code'].nunique()
	evt_counts['pick_time_min'] = picks.groupby('event_id')['time'].min()
	evt_counts['pick_time_max'] = picks.groupby('event_id')['time'].max()
	evt_counts['pick_span_s'] = (
		evt_counts['pick_time_max'] - evt_counts['pick_time_min']
	).dt.total_seconds()

	ps = picks[picks['phase_type'].isin(['P', 'S'])].copy()
	tt_stats = (
		ps.groupby(['event_id', 'phase_type'])['tt_s']
		.agg(['count', 'min', 'median', 'max'])
		.unstack('phase_type')
	)
	tt_stats.columns = [f'{stat}_{ph}' for stat, ph in tt_stats.columns]

	event_summary = evt_counts.join(tt_stats, how='left').reset_index()
	event_summary = event_summary.merge(
		epi_ok[
			[
				'event_id',
				'origin_time',
				'latitude_deg',
				'longitude_deg',
				'depth_km',
				'mag1',
				'mag1_type',
				'tt_table',
				'station_count',
				'hypocenter_flag',
			]
		],
		on='event_id',
		how='left',
		validate='1:1',
	)

	# station summary
	st = (
		picks.groupby(['station_code', 'phase_type'])['phase']
		.count()
		.unstack(fill_value=0)
	)
	for c in ['P', 'S', 'Other']:
		if c not in st.columns:
			st[c] = 0
	st['total'] = st[['P', 'S', 'Other']].sum(axis=1)
	station_summary = (
		st.reset_index().sort_values('total', ascending=False).reset_index(drop=True)
	)

	# station-level S-P
	first_tt = (
		ps.groupby(['event_id', 'station_code', 'station_number', 'phase_type'])['tt_s']
		.min()
		.unstack('phase_type')
	)
	first_tt = first_tt.dropna(subset=['P', 'S']).reset_index()
	first_tt['S_minus_P_s'] = first_tt['S'] - first_tt['P']

	evt_ll = picks[
		['event_id', 'latitude_deg', 'longitude_deg', 'mag1', 'mag1_type']
	].drop_duplicates('event_id')
	sta_ll = picks[
		['station_code', 'station_number', 'Latitude_deg', 'Longitude_deg', 'Height']
	].drop_duplicates(subset=['station_code', 'station_number'])
	station_sp = first_tt.merge(
		evt_ll, on='event_id', how='left', validate='m:1'
	).merge(sta_ll, on=['station_code', 'station_number'], how='left', validate='m:1')
	station_sp = station_sp.reset_index(drop=True)
	station_sp['dist_km'] = _compute_dist_km_by_event(station_sp).to_numpy(dtype=float)

	return picks, event_summary, station_summary, station_sp


def _savefig(path: Path) -> None:
	plt.tight_layout()
	plt.savefig(path, dpi=160)
	plt.close()


def _plot_hist_logy(
	values: pd.Series, *, bins: int, title: str, xlabel: str, outpath: Path
) -> None:
	plt.figure(figsize=(8, 4.5))
	plt.hist(values.dropna(), bins=bins)
	plt.yscale('log')
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel('count')
	_savefig(outpath)


def _plot_hist2d(
	x: pd.Series,
	y: pd.Series,
	*,
	bins: int,
	title: str,
	xlabel: str,
	ylabel: str,
	outpath: Path,
) -> None:
	plt.figure(figsize=(7, 5.5))
	plt.hist2d(x.to_numpy(), y.to_numpy(), bins=bins)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	_savefig(outpath)


def plot_jma_arrivetime_qc(
	*,
	event_summary: pd.DataFrame,
	station_summary: pd.DataFrame,
	station_sp: pd.DataFrame,
	picks: pd.DataFrame,
	fig_dir: str | Path,
	clip_dist_km: float = 200.0,
) -> None:
	fig_dir = Path(fig_dir)
	fig_dir.mkdir(parents=True, exist_ok=True)

	_plot_hist_logy(
		event_summary['n_stations_meas'],
		bins=60,
		title='Stations per event (log-y)',
		xlabel='n_stations_meas',
		outpath=fig_dir / 'hist_n_stations.png',
	)
	_plot_hist_logy(
		event_summary['total_picks'],
		bins=70,
		title='Total picks per event (log-y)',
		xlabel='total_picks',
		outpath=fig_dir / 'hist_total_picks.png',
	)

	_plot_hist2d(
		event_summary['P'].fillna(0),
		event_summary['S'].fillna(0),
		bins=40,
		title='P picks vs S picks (2D hist)',
		xlabel='P picks',
		ylabel='S picks',
		outpath=fig_dir / 'hist2d_P_vs_S.png',
	)

	# travel-time histograms (event level)
	def _tt_clip(col: str, tmax: float, fname: str) -> None:
		v = event_summary[col].dropna()
		v = v[(v >= 0) & (v <= tmax)]
		_plot_hist_logy(
			v,
			bins=int(tmax),
			title=f'{col} clipped 0-{tmax}s (log-y)',
			xlabel=f'{col} [s]',
			outpath=fig_dir / fname,
		)

	for col, tmax, fname in [
		('min_P', 120.0, 'hist_event_min_P_0_120s.png'),
		('max_P', 200.0, 'hist_event_max_P_0_200s.png'),
		('min_S', 200.0, 'hist_event_min_S_0_200s.png'),
		('max_S', 300.0, 'hist_event_max_S_0_300s.png'),
	]:
		if col in event_summary.columns:
			_tt_clip(col, tmax, fname)

	# distance vs travel time (pick level)
	for phase, tmax, fname in [
		('P', 70.0, 'hist2d_dist_vs_tt_P.png'),
		('S', 70.0, 'hist2d_dist_vs_tt_S.png'),
	]:
		df = picks[picks['phase_type'] == phase].copy()
		df = df[
			(df['dist_km'] >= 0)
			& (df['dist_km'] <= clip_dist_km)
			& (df['tt_s'] >= 0)
			& (df['tt_s'] <= tmax)
		]
		if not df.empty:
			_plot_hist2d(
				df['dist_km'],
				df['tt_s'],
				bins=80,
				title=f'Distance vs travel time ({phase})',
				xlabel='distance [km]',
				ylabel='travel time [s]',
				outpath=fig_dir / fname,
			)

	# station-level S-P
	sp = station_sp.copy()
	sp = sp[
		(sp['S_minus_P_s'] >= 0)
		& (sp['S_minus_P_s'] <= 120)
		& (sp['dist_km'] <= clip_dist_km)
	]
	if not sp.empty:
		_plot_hist_logy(
			sp['S_minus_P_s'],
			bins=120,
			title='Station-level S-P (0-120s, log-y)',
			xlabel='S-P [s]',
			outpath=fig_dir / 'hist_station_SP_0_120s.png',
		)
		_plot_hist2d(
			sp['dist_km'],
			sp['S_minus_P_s'],
			bins=80,
			title='Distance vs S-P (2D hist)',
			xlabel='distance [km]',
			ylabel='S-P [s]',
			outpath=fig_dir / 'hist2d_dist_vs_SP.png',
		)

	# top stations
	top = station_summary.head(20).copy()
	plt.figure(figsize=(10, 5))
	plt.bar(top['station_code'], top['total'])
	plt.title('Top 20 stations by total picks')
	plt.xlabel('station_code')
	plt.ylabel('total picks')
	plt.xticks(rotation=60, ha='right')
	_savefig(fig_dir / 'bar_top20_stations_total.png')

	# magnitude views
	if 'mag1' in event_summary.columns:
		m = event_summary['mag1'].dropna()
		if not m.empty:
			plt.figure(figsize=(8, 4.5))
			plt.hist(m, bins=60)
			plt.title('Magnitude (mag1) histogram')
			plt.xlabel('mag1')
			plt.ylabel('count')
			_savefig(fig_dir / 'hist_mag1.png')

			tmp = event_summary.dropna(subset=['mag1', 'total_picks']).copy()
			mmin = np.floor(tmp['mag1'].min() * 2) / 2
			mmax = np.ceil(tmp['mag1'].max() * 2) / 2
			bins = np.arange(mmin, mmax + 0.5, 0.5)
			tmp['m_bin'] = pd.cut(tmp['mag1'], bins=bins, include_lowest=True)
			grp = tmp.groupby('m_bin', observed=True)['total_picks'].median()

			plt.figure(figsize=(10, 5))
			plt.bar([str(x) for x in grp.index], grp.values)
			plt.title('Median total_picks by mag1 bin (0.5 step)')
			plt.xlabel('mag1 bin')
			plt.ylabel('median total_picks')
			plt.xticks(rotation=60, ha='right')
			_savefig(fig_dir / 'bar_mag1_bin_median_total_picks.png')


def run_jma_arrivetime_qc(
	*,
	meas_csv: str | Path,
	epic_csv: str | Path,
	station_csv: str | Path,
	out_dir: str | Path,
	p_phases: set[str] = P_PHASES_DEFAULT,
	s_phases: set[str] = S_PHASES_DEFAULT,
	exclude_hypocenter_flags: set[str] = {'F'},
) -> JmaArrivetimeQcArtifacts:
	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	fig_dir = out_dir / 'fig'

	picks, event_summary, station_summary, station_sp = build_arrivetime_qc_tables(
		meas_csv=meas_csv,
		epic_csv=epic_csv,
		station_csv=station_csv,
		p_phases=p_phases,
		s_phases=s_phases,
		exclude_hypocenter_flags=exclude_hypocenter_flags,
	)

	picks_csv = out_dir / 'picks_long_with_tt_dist.csv'
	event_csv = out_dir / 'event_summary_with_tt.csv'
	station_csv_out = out_dir / 'station_summary.csv'
	station_sp_csv = out_dir / 'station_level_SP_with_dist.csv'

	picks.to_csv(picks_csv, index=False)
	event_summary.to_csv(event_csv, index=False)
	station_summary.to_csv(station_csv_out, index=False)
	station_sp.to_csv(station_sp_csv, index=False)

	plot_jma_arrivetime_qc(
		event_summary=event_summary,
		station_summary=station_summary,
		station_sp=station_sp,
		picks=picks,
		fig_dir=fig_dir,
	)

	return JmaArrivetimeQcArtifacts(
		out_dir=out_dir,
		fig_dir=fig_dir,
		picks_csv=picks_csv,
		event_summary_csv=event_csv,
		station_summary_csv=station_csv_out,
		station_sp_csv=station_sp_csv,
	)
