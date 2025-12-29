# %%
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from jma.chk_network_station import parse_station_names_from_ch
from jma.download import create_hinet_client
from jma.station_reader import read_hinet_channel_table

EPIC = Path('/workspace/data/arrivetime/JMA/arrivetime_epicenters.csv')  # 20年分
START_MONTH, END_MONTH = '200404', '202512'  # inclusive
OUT_ROOT = Path('./snapshots_v2/monthly')
PAD_MINUTES = 1
REGION, MINMAG, MAXMAG, INCLUDE_UNKNOWN_MAG = '00', -5.0, 9.9, True

client = create_hinet_client()
stamp = datetime.now().strftime('%Y%m%d%H%M%S')


def _run_get_event_waveform(outdir: Path, t0: datetime, t1: datetime) -> list[Path]:
	outdir.mkdir(parents=True, exist_ok=True)
	prev = os.getcwd()
	os.chdir(outdir)
	try:
		client.get_event_waveform(
			t0,
			t1,
			region=REGION,
			minmagnitude=MINMAG,
			maxmagnitude=MAXMAG,
			include_unknown_mag=INCLUDE_UNKNOWN_MAG,
		)
	finally:
		os.chdir(prev)
	return sorted(outdir.rglob('*.ch'))


def _download_month_ch_files(
	outdir: Path, ot: datetime
) -> tuple[list[Path], datetime, str]:
	t0 = ot.replace(second=0, microsecond=0)
	t1 = t0 + timedelta(minutes=PAD_MINUTES)

	try:
		ch_files = _run_get_event_waveform(outdir, t0, t1)
		if ch_files:
			return ch_files, t0, 'ok'
	except Exception as e:
		print(
			f'[WARN] get_event_waveform failed (month={outdir.parent.name}) at {t0} : {type(e).__name__}: {e}'
		)

	# 特例：日付を+1日ずらして再トライ
	t0b = t0 + timedelta(days=1)
	t1b = t0b + timedelta(minutes=PAD_MINUTES)
	print(f'[WARN] retry with +1 day: {t0b} (month={outdir.parent.name})')

	try:
		ch_files = _run_get_event_waveform(outdir, t0b, t1b)
		if ch_files:
			return ch_files, t0b, 'retry_ok'
	except Exception as e:
		print(
			f'[WARN] retry failed (month={outdir.parent.name}) at {t0b} : {type(e).__name__}: {e}'
		)

	return [], t0b, 'failed'


ep = pd.read_csv(EPIC)
ep['origin_time'] = pd.to_datetime(ep['origin_time'], errors='raise')
ep['month'] = ep['origin_time'].dt.to_period('M').astype(str)

months = (
	pd.date_range(
		datetime.strptime(START_MONTH, '%Y%m'),
		datetime.strptime(END_MONTH, '%Y%m'),
		freq='MS',
	)
	.strftime('%Y-%m')
	.tolist()
)
ep = ep[ep['month'].isin(months)].copy()
first_ot = ep.groupby('month')['origin_time'].min().to_dict()

month_to_sta: dict[str, set[str]] = {}
meta_rows: list[pd.DataFrame] = []
log_rows: list[dict[str, object]] = []

for m in months:
	if m not in first_ot:
		print(f'[WARN] no events in month: {m} -> presence=0')
		month_to_sta[m] = set()
		log_rows.append(
			{
				'month': m,
				'status': 'no_event',
				'used_time': '',
				'ch_files': 0,
				'outdir': '',
			}
		)
		continue

	ot = pd.Timestamp(first_ot[m]).to_pydatetime()
	outdir = OUT_ROOT / m / stamp

	ch_files, used_t0, status = _download_month_ch_files(outdir, ot)
	if not ch_files:
		month_to_sta[m] = set()
		log_rows.append(
			{
				'month': m,
				'status': status,
				'used_time': used_t0.isoformat(),
				'ch_files': 0,
				'outdir': str(outdir),
			}
		)
		print(f'[WARN] {m} -> no .ch (status={status}) outdir={outdir}')
		continue

	sta_set: set[str] = set()
	for ch in ch_files:
		ct = read_hinet_channel_table(ch)
		ct['station'] = ct['station'].astype(str).str.strip().str.upper()
		ct['component'] = ct['component'].astype(str).str.strip().str.upper()
		name_map = parse_station_names_from_ch(ch)
		ct['station_name'] = ct['station'].map(name_map).fillna('')
		sta_set |= set(ct['station'].unique().tolist())
		meta_rows.append(
			ct[['station', 'station_name', 'lat', 'lon', 'elevation_m', 'component']]
		)

	month_to_sta[m] = sta_set
	log_rows.append(
		{
			'month': m,
			'status': status,
			'used_time': used_t0.isoformat(),
			'ch_files': len(ch_files),
			'outdir': str(outdir),
		}
	)
	print(
		m,
		used_t0,
		f'stations={len(sta_set)}',
		f'ch={len(ch_files)}',
		f'status={status}',
		'->',
		outdir,
	)

if not meta_rows:
	raise ValueError('all months failed; meta_rows is empty')

all_meta = pd.concat(meta_rows, ignore_index=True).drop_duplicates()
station_master = (
	all_meta.groupby('station', as_index=False)
	.agg(
		station_name=('station_name', 'first'),
		lat=('lat', 'first'),
		lon=('lon', 'first'),
		elevation_m=('elevation_m', 'first'),
		components=('component', lambda x: ','.join(sorted(set(x)))),
		n_components=('component', lambda x: len(set(x))),
	)
	.sort_values('station')
	.reset_index(drop=True)
)

presence = station_master[['station']].copy()
for m in months:
	presence[m] = (
		presence['station']
		.map(lambda s: 1 if s in month_to_sta.get(m, set()) else 0)
		.astype(int)
	)
presence = presence.merge(station_master, on='station', how='left')

summary = OUT_ROOT / '_summary'
summary.mkdir(parents=True, exist_ok=True)
presence.to_csv(summary / 'monthly_presence.csv', index=False, encoding='utf-8')
station_master.to_csv(summary / 'station_master.csv', index=False, encoding='utf-8')
pd.DataFrame(log_rows).to_csv(
	summary / 'monthly_seed_events.csv', index=False, encoding='utf-8'
)
