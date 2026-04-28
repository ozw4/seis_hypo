# %%
"""QC and visualization for Izu 2009 GaMMA association results."""

# file: proc/izu2009/association/qc_gamma_izu2009.py
#
# Required inputs:
# - proc/izu2009/association/out/gamma_events.csv
# - proc/izu2009/association/out/gamma_picks.csv
# - proc/izu2009/association/out/gamma_config.json
# - proc/izu2009/association/in/gamma_stations.csv
# - proc/izu2009/association/in/origin_latlon.json

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _REPO_ROOT / 'src'

for _path in (_REPO_ROOT, _SRC_DIR):
	_path_str = str(_path)
	if _path_str not in sys.path:
		sys.path.insert(0, _path_str)

from common.geo import local_xy_km_to_latlon  # noqa: E402
from common.json_io import read_json  # noqa: E402
from viz.core.fig_io import save_figure  # noqa: E402

IN_DIR = _REPO_ROOT / 'proc/izu2009/association/in'
GAMMA_OUT_DIR = _REPO_ROOT / 'proc/izu2009/association/out'
QC_DIR = _REPO_ROOT / 'proc/izu2009/association/qc'

EVENTS_CSV = GAMMA_OUT_DIR / 'gamma_events.csv'
PICKS_CSV = GAMMA_OUT_DIR / 'gamma_picks.csv'
CONFIG_JSON = GAMMA_OUT_DIR / 'gamma_config.json'
STATIONS_CSV = IN_DIR / 'gamma_stations.csv'
ORIGIN_JSON = IN_DIR / 'origin_latlon.json'

OUT_EVENTS_LATLON_CSV = QC_DIR / 'gamma_events_with_latlon.csv'
OUT_EVENTS_FLAGS_CSV = QC_DIR / 'gamma_events_qc_flags.csv'
OUT_STATION_SUMMARY_CSV = QC_DIR / 'gamma_station_pick_summary.csv'
OUT_HOURLY_COUNTS_CSV = QC_DIR / 'gamma_hourly_counts.csv'
OUT_ASSOCIATED_PICKS_CSV_GZ = QC_DIR / 'gamma_associated_picks_enriched.csv.gz'
OUT_SP_PAIRS_CSV = QC_DIR / 'gamma_station_sp_pairs.csv'
OUT_SUMMARY_MD = QC_DIR / 'gamma_qc_summary.md'

FIG_DIR = QC_DIR / 'figures'
BOUNDARY_TOL_KM = 1.0e-6
NEAR_BOUNDARY_KM = 1.0


def _require_file(path: Path) -> None:
	if not path.is_file():
		raise FileNotFoundError(f'input file not found: {path}')


def _require_input_files() -> None:
	for path in [EVENTS_CSV, PICKS_CSV, CONFIG_JSON, STATIONS_CSV, ORIGIN_JSON]:
		_require_file(path)


def _read_gamma_events() -> pd.DataFrame:
	events = pd.read_csv(EVENTS_CSV)
	if events.empty:
		raise RuntimeError(f'no events in {EVENTS_CSV}')
	required = {
		'time',
		'event_index',
		'x(km)',
		'y(km)',
		'z(km)',
		'num_picks',
		'num_p_picks',
		'num_s_picks',
		'gamma_score',
		'sigma_time',
	}
	missing = required.difference(events.columns)
	if missing:
		raise ValueError(f'{EVENTS_CSV} missing columns: {sorted(missing)}')

	events['origin_time'] = pd.to_datetime(events['time'], utc=True, errors='raise')
	return events


def _read_gamma_picks() -> pd.DataFrame:
	picks = pd.read_csv(PICKS_CSV)
	if picks.empty:
		raise RuntimeError(f'no picks in {PICKS_CSV}')
	required = {'station_id', 'phase_time', 'phase_type', 'event_index'}
	missing = required.difference(picks.columns)
	if missing:
		raise ValueError(f'{PICKS_CSV} missing columns: {sorted(missing)}')

	picks['phase_time_utc'] = pd.to_datetime(
		picks['phase_time'], utc=True, errors='raise'
	)
	picks['phase_type'] = picks['phase_type'].astype(str).str.upper()
	return picks


def _read_stations() -> pd.DataFrame:
	stations = pd.read_csv(STATIONS_CSV)
	required = {'id', 'x(km)', 'y(km)', 'z(km)'}
	missing = required.difference(stations.columns)
	if missing:
		raise ValueError(f'{STATIONS_CSV} missing columns: {sorted(missing)}')
	if stations['id'].duplicated().any():
		dups = stations.loc[stations['id'].duplicated(), 'id'].astype(str).tolist()
		raise ValueError(f'duplicate station ids in {STATIONS_CSV}: {dups}')
	return stations


def _events_with_latlon(events: pd.DataFrame) -> pd.DataFrame:
	origin = read_json(ORIGIN_JSON)
	for key in ['lat0_deg', 'lon0_deg']:
		if key not in origin:
			raise ValueError(f'{ORIGIN_JSON} missing key: {key}')

	lat, lon = local_xy_km_to_latlon(
		events['x(km)'].to_numpy(dtype=float),
		events['y(km)'].to_numpy(dtype=float),
		lat0_deg=float(origin['lat0_deg']),
		lon0_deg=float(origin['lon0_deg']),
	)
	out = events.copy()
	out['latitude_deg'] = lat
	out['longitude_deg'] = lon
	out['depth_km'] = out['z(km)'].astype(float)
	return out.sort_values('origin_time').reset_index(drop=True)


def _events_with_qc_flags(events: pd.DataFrame, config: dict) -> pd.DataFrame:
	out = events.copy()
	x_min, x_max = [float(v) for v in config['x(km)']]
	y_min, y_max = [float(v) for v in config['y(km)']]
	z_min, z_max = [float(v) for v in config['z(km)']]

	out['on_x_min'] = np.isclose(out['x(km)'], x_min, atol=BOUNDARY_TOL_KM)
	out['on_x_max'] = np.isclose(out['x(km)'], x_max, atol=BOUNDARY_TOL_KM)
	out['on_y_min'] = np.isclose(out['y(km)'], y_min, atol=BOUNDARY_TOL_KM)
	out['on_y_max'] = np.isclose(out['y(km)'], y_max, atol=BOUNDARY_TOL_KM)
	out['on_z_min'] = np.isclose(out['z(km)'], z_min, atol=BOUNDARY_TOL_KM)
	out['on_z_max'] = np.isclose(out['z(km)'], z_max, atol=BOUNDARY_TOL_KM)

	out['near_x_min_1km'] = (out['x(km)'] - x_min).abs() <= NEAR_BOUNDARY_KM
	out['near_x_max_1km'] = (out['x(km)'] - x_max).abs() <= NEAR_BOUNDARY_KM
	out['near_y_min_1km'] = (out['y(km)'] - y_min).abs() <= NEAR_BOUNDARY_KM
	out['near_y_max_1km'] = (out['y(km)'] - y_max).abs() <= NEAR_BOUNDARY_KM
	out['near_z_min_1km'] = (out['z(km)'] - z_min).abs() <= NEAR_BOUNDARY_KM
	out['near_z_max_1km'] = (out['z(km)'] - z_max).abs() <= NEAR_BOUNDARY_KM

	boundary_cols = [
		'on_x_min',
		'on_x_max',
		'on_y_min',
		'on_y_max',
		'on_z_min',
		'on_z_max',
	]
	near_cols = [
		'near_x_min_1km',
		'near_x_max_1km',
		'near_y_min_1km',
		'near_y_max_1km',
		'near_z_min_1km',
		'near_z_max_1km',
	]
	out['on_any_boundary'] = out[boundary_cols].any(axis=1)
	out['near_any_boundary_1km'] = out[near_cols].any(axis=1)
	return out.sort_values('origin_time').reset_index(drop=True)


def _station_summary(picks: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
	p = picks.copy()
	p['assigned'] = p['event_index'].astype(int) >= 0
	p['is_p'] = p['phase_type'].eq('P')
	p['is_s'] = p['phase_type'].eq('S')
	p['assigned_p'] = p['assigned'] & p['is_p']
	p['assigned_s'] = p['assigned'] & p['is_s']

	summary = p.groupby('station_id', sort=True).agg(
		total_picks=('station_id', 'size'),
		assigned_picks=('assigned', 'sum'),
		p_picks=('is_p', 'sum'),
		s_picks=('is_s', 'sum'),
		assigned_p_picks=('assigned_p', 'sum'),
		assigned_s_picks=('assigned_s', 'sum'),
	)
	summary = summary.reset_index()
	station_ids = stations[['id']].rename(columns={'id': 'station_id'}).copy()
	out = station_ids.merge(summary, on='station_id', how='left')
	count_cols = [
		'total_picks',
		'assigned_picks',
		'p_picks',
		's_picks',
		'assigned_p_picks',
		'assigned_s_picks',
	]
	out[count_cols] = out[count_cols].fillna(0).astype(int)
	out['assigned_rate'] = np.where(
		out['total_picks'] > 0,
		out['assigned_picks'] / out['total_picks'],
		0.0,
	)
	out = out.merge(
		stations,
		left_on='station_id',
		right_on='id',
		how='left',
		validate='one_to_one',
	)
	out = out.drop(columns=['id'])
	return out.sort_values(['assigned_picks', 'station_id'], ascending=[False, True])


def _hourly_counts(events: pd.DataFrame, picks: pd.DataFrame) -> pd.DataFrame:
	event_counts = events.set_index('origin_time').resample('1h').size()
	assigned_picks = (
		picks[picks['event_index'] >= 0]
		.set_index('phase_time_utc')
		.resample('1h')
		.size()
	)
	all_picks = picks.set_index('phase_time_utc').resample('1h').size()
	out = pd.concat(
		[
			event_counts.rename('events'),
			assigned_picks.rename('assigned_picks'),
			all_picks.rename('all_picks'),
		],
		axis=1,
	).fillna(0)
	return out.astype(int).reset_index().rename(columns={'index': 'time_utc'})


def _associated_picks_enriched(
	picks: pd.DataFrame, events: pd.DataFrame, stations: pd.DataFrame
) -> pd.DataFrame:
	assigned = picks[picks['event_index'] >= 0].copy()
	event_cols = [
		'event_index',
		'origin_time',
		'x(km)',
		'y(km)',
		'z(km)',
		'num_picks',
		'num_p_picks',
		'num_s_picks',
		'gamma_score',
		'sigma_time',
	]
	station_cols = ['id', 'x(km)', 'y(km)', 'z(km)']
	if 'network_code' in stations.columns:
		station_cols.append('network_code')
	if 'station_code' in stations.columns:
		station_cols.append('station_code')

	merged = assigned.merge(
		events[event_cols],
		on='event_index',
		how='left',
		validate='many_to_one',
		suffixes=('', '_event'),
	)
	if merged['origin_time'].isna().any():
		bad = merged.loc[merged['origin_time'].isna(), 'event_index'].unique().tolist()
		raise ValueError(f'assigned picks reference missing event_index values: {bad}')

	merged = merged.merge(
		stations[station_cols],
		left_on='station_id',
		right_on='id',
		how='left',
		validate='many_to_one',
		suffixes=('_event', '_station'),
	)
	if merged['id'].isna().any():
		bad = merged.loc[merged['id'].isna(), 'station_id'].unique().tolist()
		raise ValueError(f'picks reference missing station_id values: {bad}')

	merged = merged.drop(columns=['id'])
	merged = merged.rename(
		columns={
			'x(km)_event': 'event_x_km',
			'y(km)_event': 'event_y_km',
			'z(km)_event': 'event_z_km',
			'x(km)_station': 'station_x_km',
			'y(km)_station': 'station_y_km',
			'z(km)_station': 'station_z_km',
		}
	)
	merged['travel_time_s'] = (
		merged['phase_time_utc'] - merged['origin_time']
	).dt.total_seconds()
	merged['epicentral_distance_km'] = np.hypot(
		merged['event_x_km'] - merged['station_x_km'],
		merged['event_y_km'] - merged['station_y_km'],
	)
	merged['hypocentral_distance_km'] = np.sqrt(
		merged['epicentral_distance_km'].to_numpy(dtype=float) ** 2
		+ (merged['event_z_km'] - merged['station_z_km']).to_numpy(dtype=float) ** 2
	)
	return merged.sort_values(
		['event_index', 'phase_time_utc', 'station_id']
	).reset_index(drop=True)


def _station_sp_pairs(associated: pd.DataFrame) -> pd.DataFrame:
	cols = [
		'event_index',
		'station_id',
		'phase_type',
		'travel_time_s',
		'epicentral_distance_km',
		'hypocentral_distance_km',
	]
	grp = associated[cols].groupby(
		['event_index', 'station_id', 'phase_type'], sort=True
	)
	agg = grp.agg(
		travel_time_s=('travel_time_s', 'min'),
		epicentral_distance_km=('epicentral_distance_km', 'first'),
		hypocentral_distance_km=('hypocentral_distance_km', 'first'),
	).reset_index()
	pivot = agg.pivot_table(
		index=['event_index', 'station_id'],
		columns='phase_type',
		aggfunc='first',
	)
	if ('travel_time_s', 'P') not in pivot.columns or (
		'travel_time_s',
		'S',
	) not in pivot.columns:
		return pd.DataFrame(
			columns=[
				'event_index',
				'station_id',
				'p_travel_time_s',
				's_travel_time_s',
				's_minus_p_s',
				'epicentral_distance_km',
				'hypocentral_distance_km',
			]
		)

	out = pd.DataFrame(
		{
			'p_travel_time_s': pivot[('travel_time_s', 'P')],
			's_travel_time_s': pivot[('travel_time_s', 'S')],
			'epicentral_distance_km': pivot[
				('epicentral_distance_km', 'P')
			].combine_first(pivot[('epicentral_distance_km', 'S')]),
			'hypocentral_distance_km': pivot[
				('hypocentral_distance_km', 'P')
			].combine_first(pivot[('hypocentral_distance_km', 'S')]),
		}
	).reset_index()
	out['s_minus_p_s'] = out['s_travel_time_s'] - out['p_travel_time_s']
	out = out[out['s_minus_p_s'] >= 0.0].copy()
	return out.sort_values(['event_index', 'station_id']).reset_index(drop=True)


def _plot_event_map_3view(
	events: pd.DataFrame, stations: pd.DataFrame, config: dict
) -> None:
	fig = plt.figure(figsize=(11.0, 10.0))
	gs = fig.add_gridspec(
		2,
		2,
		width_ratios=[3.0, 1.2],
		height_ratios=[3.0, 1.2],
		wspace=0.12,
		hspace=0.12,
	)
	ax_xy = fig.add_subplot(gs[0, 0])
	ax_yz = fig.add_subplot(gs[0, 1], sharey=ax_xy)
	ax_xz = fig.add_subplot(gs[1, 0], sharex=ax_xy)
	ax_empty = fig.add_subplot(gs[1, 1])
	ax_empty.axis('off')

	depth = events['depth_km'].to_numpy(dtype=float)
	sc = ax_xy.scatter(
		events['x(km)'],
		events['y(km)'],
		c=depth,
		s=8,
		alpha=0.65,
		linewidths=0,
		cmap='viridis_r',
	)
	ax_xy.scatter(
		stations['x(km)'],
		stations['y(km)'],
		s=42,
		marker='^',
		facecolors='white',
		edgecolors='black',
		linewidths=0.7,
		label='stations',
	)
	label_col = 'station_code' if 'station_code' in stations.columns else 'id'
	for _, row in stations.iterrows():
		ax_xy.text(
			float(row['x(km)']) + 0.3,
			float(row['y(km)']) + 0.3,
			str(row[label_col]),
			fontsize=6,
		)

	ax_xz.scatter(
		events['x(km)'], events['depth_km'], c=depth, s=8, alpha=0.65, cmap='viridis_r'
	)
	ax_yz.scatter(
		events['depth_km'], events['y(km)'], c=depth, s=8, alpha=0.65, cmap='viridis_r'
	)

	x_min, x_max = [float(v) for v in config['x(km)']]
	y_min, y_max = [float(v) for v in config['y(km)']]
	z_min, z_max = [float(v) for v in config['z(km)']]
	ax_xy.set_xlim(x_min, x_max)
	ax_xy.set_ylim(y_min, y_max)
	ax_xz.set_ylim(z_max, z_min)
	ax_yz.set_xlim(z_min, z_max)
	ax_yz.invert_xaxis()

	ax_xy.set_ylabel('North [km]')
	ax_xz.set_xlabel('East [km]')
	ax_xz.set_ylabel('Depth [km]')
	ax_yz.set_xlabel('Depth [km]')
	ax_xy.set_title('GaMMA events and stations, local XY')
	ax_xy.grid(visible=True, linewidth=0.4)
	ax_xz.grid(visible=True, linewidth=0.4)
	ax_yz.grid(visible=True, linewidth=0.4)
	ax_xy.legend(loc='upper right')
	cbar = fig.colorbar(sc, ax=[ax_xy, ax_xz, ax_yz], shrink=0.78, pad=0.03)
	cbar.set_label('Depth [km]')
	save_figure(fig, FIG_DIR / 'gamma_events_3view_localxy.png', dpi=200)


def _plot_event_histograms(events: pd.DataFrame) -> None:
	fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.0))
	items = [
		('num_picks', 'Number of picks per event'),
		('num_p_picks', 'P picks per event'),
		('num_s_picks', 'S picks per event'),
		('sigma_time', 'GaMMA sigma_time [s]'),
		('gamma_score', 'GaMMA score'),
		('depth_km', 'Depth [km]'),
	]
	for ax, (col, title) in zip(axes.ravel(), items, strict=True):
		ax.hist(events[col].to_numpy(dtype=float), bins=50)
		ax.set_title(title)
		ax.set_ylabel('Count')
		ax.grid(visible=True, linewidth=0.4)
	fig.tight_layout()
	save_figure(fig, FIG_DIR / 'gamma_event_histograms.png', dpi=200)


def _plot_depth_time(events: pd.DataFrame) -> None:
	fig, ax = plt.subplots(figsize=(10.0, 5.5))
	sc = ax.scatter(
		events['origin_time'],
		events['depth_km'],
		c=events['num_picks'],
		s=8,
		alpha=0.7,
		linewidths=0,
	)
	ax.invert_yaxis()
	ax.set_xlabel('Origin time [UTC]')
	ax.set_ylabel('Depth [km]')
	ax.set_title('Depth versus origin time')
	ax.grid(visible=True, linewidth=0.4)
	cbar = fig.colorbar(sc, ax=ax)
	cbar.set_label('Number of picks')
	fig.autofmt_xdate()
	save_figure(fig, FIG_DIR / 'gamma_depth_vs_time.png', dpi=200, bbox_inches='tight')


def _plot_hourly_counts(hourly: pd.DataFrame) -> None:
	fig, ax1 = plt.subplots(figsize=(11.0, 5.5))
	t = pd.to_datetime(hourly['time_utc'], utc=True, errors='raise')
	ax1.plot(t, hourly['events'], label='events')
	ax1.set_xlabel('Time [UTC]')
	ax1.set_ylabel('Events per hour')
	ax1.grid(visible=True, linewidth=0.4)
	ax2 = ax1.twinx()
	ax2.plot(t, hourly['assigned_picks'], linestyle='--', label='assigned picks')
	ax2.set_ylabel('Assigned picks per hour')
	lines1, labels1 = ax1.get_legend_handles_labels()
	lines2, labels2 = ax2.get_legend_handles_labels()
	ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
	ax1.set_title('Hourly GaMMA event and assigned-pick counts')
	fig.autofmt_xdate()
	save_figure(fig, FIG_DIR / 'gamma_hourly_counts.png', dpi=200, bbox_inches='tight')


def _plot_station_assignment(summary: pd.DataFrame) -> None:
	df = summary.sort_values('assigned_rate', ascending=True).copy()
	fig, ax = plt.subplots(figsize=(8.0, 11.0))
	labels = df['station_id'].astype(str).to_numpy()
	y = np.arange(len(df))
	ax.barh(y, df['assigned_rate'].to_numpy(dtype=float))
	ax.set_yticks(y)
	ax.set_yticklabels(labels, fontsize=7)
	ax.set_xlabel('Assigned pick fraction')
	ax.set_title('Station-level association rate')
	ax.grid(visible=True, axis='x', linewidth=0.4)
	for i, value in enumerate(df['assigned_picks'].to_numpy(dtype=int)):
		ax.text(
			float(df.iloc[i]['assigned_rate']) + 0.005,
			i,
			str(value),
			va='center',
			fontsize=6,
		)
	save_figure(
		fig, FIG_DIR / 'gamma_station_assignment_rate.png', dpi=200, bbox_inches='tight'
	)


def _plot_travel_time_distance(associated: pd.DataFrame) -> None:
	fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.5), sharex=True, sharey=True)
	for ax, phase in zip(axes, ['P', 'S'], strict=True):
		df = associated[
			associated['phase_type'].eq(phase)
			& (associated['travel_time_s'] >= 0.0)
			& (associated['travel_time_s'] <= 80.0)
		].copy()
		ax.scatter(
			df['hypocentral_distance_km'],
			df['travel_time_s'],
			s=5,
			alpha=0.18,
			linewidths=0,
		)
		ax.set_title(f'{phase} picks')
		ax.set_xlabel('Hypocentral distance [km]')
		ax.grid(visible=True, linewidth=0.4)
	axes[0].set_ylabel('Travel time from GaMMA origin [s]')
	fig.suptitle('Associated-pick travel time versus hypocentral distance')
	fig.tight_layout()
	save_figure(fig, FIG_DIR / 'gamma_travel_time_vs_distance.png', dpi=200)


def _plot_sp_distance(sp_pairs: pd.DataFrame) -> None:
	fig, ax = plt.subplots(figsize=(8.0, 5.5))
	if not sp_pairs.empty:
		df = sp_pairs[
			(sp_pairs['s_minus_p_s'] >= 0.0) & (sp_pairs['s_minus_p_s'] <= 50.0)
		].copy()
		ax.scatter(
			df['hypocentral_distance_km'],
			df['s_minus_p_s'],
			s=6,
			alpha=0.25,
			linewidths=0,
		)
	ax.set_xlabel('Hypocentral distance [km]')
	ax.set_ylabel('S minus P [s]')
	ax.set_title('Station-level S minus P versus distance')
	ax.grid(visible=True, linewidth=0.4)
	save_figure(fig, FIG_DIR / 'gamma_sp_vs_distance.png', dpi=200)


def _fmt_float(value: float, digits: int = 3) -> str:
	return f'{float(value):.{digits}f}'


def _write_summary_md(  # noqa: PLR0913
	*,
	events: pd.DataFrame,
	picks: pd.DataFrame,
	associated: pd.DataFrame,
	station_summary: pd.DataFrame,
	hourly: pd.DataFrame,
	flags: pd.DataFrame,
	sp_pairs: pd.DataFrame,
	config: dict,
) -> None:
	assigned_count = int((picks['event_index'] >= 0).sum())
	picks_count = len(picks)
	assigned_rate = assigned_count / picks_count
	phase_counts = picks['phase_type'].value_counts().to_dict()
	assigned_phase_counts = associated['phase_type'].value_counts().to_dict()
	z_min_count = int(flags['on_z_min'].sum())
	z_max_count = int(flags['on_z_max'].sum())
	xy_boundary_count = int(
		flags[['on_x_min', 'on_x_max', 'on_y_min', 'on_y_max']].any(axis=1).sum()
	)
	near_boundary_count = int(flags['near_any_boundary_1km'].sum())

	best_hour = hourly.sort_values('events', ascending=False).head(1)
	if best_hour.empty:
		best_hour_text = 'not available'
	else:
		best_hour_text = (
			f'{best_hour.iloc[0]["time_utc"]} '
			f'with {int(best_hour.iloc[0]["events"])} events'
		)

	lines = [
		'# Izu 2009 GaMMA QC summary',
		'',
		'## Input and association counts',
		'',
		f'- events: {len(events)}',
		f'- picks: {picks_count}',
		f'- assigned picks: {assigned_count}',
		f'- assigned fraction: {_fmt_float(assigned_rate, 4)}',
		(
			f'- all pick phases: P={int(phase_counts.get("P", 0))}, '
			f'S={int(phase_counts.get("S", 0))}'
		),
		(
			f'- assigned pick phases: P={int(assigned_phase_counts.get("P", 0))}, '
			f'S={int(assigned_phase_counts.get("S", 0))}'
		),
		'',
		'## Time range',
		'',
		(
			f'- event time UTC: {events["origin_time"].min()} '
			f'to {events["origin_time"].max()}'
		),
		(
			f'- pick time UTC: {picks["phase_time_utc"].min()} '
			f'to {picks["phase_time_utc"].max()}'
		),
		f'- busiest hour: {best_hour_text}',
		'',
		'## Event-quality distributions',
		'',
		f'- num_picks median: {_fmt_float(events["num_picks"].median(), 2)}',
		(
			'- num_picks 95th percentile: '
			f'{_fmt_float(events["num_picks"].quantile(0.95), 2)}'
		),
		f'- sigma_time median s: {_fmt_float(events["sigma_time"].median(), 3)}',
		(
			'- sigma_time 95th percentile s: '
			f'{_fmt_float(events["sigma_time"].quantile(0.95), 3)}'
		),
		f'- gamma_score median: {_fmt_float(events["gamma_score"].median(), 3)}',
		f'- depth median km: {_fmt_float(events["depth_km"].median(), 3)}',
		'',
		'## Boundary checks',
		'',
		f'- search x range km: {config["x(km)"]}',
		f'- search y range km: {config["y(km)"]}',
		f'- search z range km: {config["z(km)"]}',
		f'- events on z minimum: {z_min_count}',
		f'- events on z maximum: {z_max_count}',
		f'- events on x or y boundary: {xy_boundary_count}',
		(
			f'- events within {NEAR_BOUNDARY_KM:g} km of any boundary: '
			f'{near_boundary_count}'
		),
		'',
		'## Station checks',
		'',
		f'- stations in GaMMA station table: {len(station_summary)}',
		(
			'- stations with no picks: '
			f'{int((station_summary["total_picks"] == 0).sum())}'
		),
		(
			'- stations with no assigned picks: '
			f'{int((station_summary["assigned_picks"] == 0).sum())}'
		),
		'',
		'## S-P pairs',
		'',
		f'- event-station P/S pairs: {len(sp_pairs)}',
	]
	OUT_SUMMARY_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
	"""Create QC CSVs and figures for Izu 2009 GaMMA outputs."""
	_require_input_files()
	QC_DIR.mkdir(parents=True, exist_ok=True)
	FIG_DIR.mkdir(parents=True, exist_ok=True)

	config = read_json(CONFIG_JSON)
	events_raw = _read_gamma_events()
	picks = _read_gamma_picks()
	stations = _read_stations()
	events = _events_with_latlon(events_raw)
	flags = _events_with_qc_flags(events, config)
	station_summary = _station_summary(picks, stations)
	hourly = _hourly_counts(events, picks)
	associated = _associated_picks_enriched(picks, events, stations)
	sp_pairs = _station_sp_pairs(associated)

	events.to_csv(
		OUT_EVENTS_LATLON_CSV, index=False, date_format='%Y-%m-%dT%H:%M:%S.%fZ'
	)
	flags.to_csv(OUT_EVENTS_FLAGS_CSV, index=False, date_format='%Y-%m-%dT%H:%M:%S.%fZ')
	station_summary.to_csv(OUT_STATION_SUMMARY_CSV, index=False)
	hourly.to_csv(
		OUT_HOURLY_COUNTS_CSV, index=False, date_format='%Y-%m-%dT%H:%M:%S.%fZ'
	)
	associated.to_csv(OUT_ASSOCIATED_PICKS_CSV_GZ, index=False)
	sp_pairs.to_csv(OUT_SP_PAIRS_CSV, index=False)

	_plot_event_map_3view(events, stations, config)
	_plot_event_histograms(events)
	_plot_depth_time(events)
	_plot_hourly_counts(hourly)
	_plot_station_assignment(station_summary)
	_plot_travel_time_distance(associated)
	_plot_sp_distance(sp_pairs)

	_write_summary_md(
		events=events,
		picks=picks,
		associated=associated,
		station_summary=station_summary,
		hourly=hourly,
		flags=flags,
		sp_pairs=sp_pairs,
		config=config,
	)

	print('Wrote:', OUT_EVENTS_LATLON_CSV)
	print('Wrote:', OUT_EVENTS_FLAGS_CSV)
	print('Wrote:', OUT_STATION_SUMMARY_CSV)
	print('Wrote:', OUT_HOURLY_COUNTS_CSV)
	print('Wrote:', OUT_ASSOCIATED_PICKS_CSV_GZ)
	print('Wrote:', OUT_SP_PAIRS_CSV)
	print('Wrote:', OUT_SUMMARY_MD)
	print('Wrote figures:', FIG_DIR)


if __name__ == '__main__':
	main()

# Example:
# python proc/izu2009/association/qc_gamma_izu2009.py
