# %%
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import pandas as pd

from common.done_marker import read_done_json
from common.json_io import read_json
from jma.prepare.event_txt import read_event_txt_meta
from jma.prepare.missing_io import read_missing_by_network
from jma.station_reader import read_hinet_channel_table
from jma.stationcode_common import normalize_code, normalize_network_code


def _axis_from_component(raw: str) -> str:
	comp = normalize_code(raw)
	if not comp:
		return ''
	tail = comp[-1]
	if tail == 'Z':
		return 'U'
	if tail == 'Y':
		return 'N'
	if tail == 'X':
		return 'E'
	if tail in {'U', 'N', 'E'}:
		return tail
	return ''


def _merge_axis_maps(dst: dict[str, set[str]], src: dict[str, set[str]]) -> None:
	for sta, axes in src.items():
		dst.setdefault(sta, set()).update(axes)


def _axis_map_from_ch(ch_path: Path) -> tuple[int, dict[str, set[str]]]:
	df = read_hinet_channel_table(ch_path)
	axis_by_sta: dict[str, set[str]] = defaultdict(set)
	for _, r in df.iterrows():
		sta = normalize_code(str(r.get('station', '')))
		if not sta:
			continue
		ax = _axis_from_component(str(r.get('component', '')))
		if ax:
			axis_by_sta[sta].add(ax)
	return len(df), axis_by_sta


def _read_mapping_log_counts(path: Path) -> tuple[int, int, int]:
	if not path.is_file():
		return 0, 0, 0
	with path.open('r', encoding='utf-8', errors='strict', newline='') as f:
		reader = csv.reader(f)
		header = next(reader, None)
		if not header:
			return 0, 0, 0
		try:
			i_status = header.index('map_status')
		except ValueError:
			n = sum(1 for _ in reader)
			return 1 + n, 0, 0

		n_rows = 1
		n_mapped = 0
		n_unmapped = 0
		for row in reader:
			n_rows += 1
			status = (row[i_status] if i_status < len(row) else '').strip().lower()
			if status == 'mapped':
				n_mapped += 1
			else:
				n_unmapped += 1
		return n_rows, n_mapped, n_unmapped


def _pick_best(paths: list[Path]) -> Path | None:
	if not paths:
		return None
	return sorted(paths, key=lambda p: (p.stat().st_mtime, p.name))[-1]


def _continuous_done_ok(path: Path) -> bool:
	obj = read_done_json(path, on_missing='empty', on_error='empty')
	status = str(obj.get('status', '')).lower()
	return status in {'done', 'exists', 'no_missing_file'}


def _fill_done_ok(path: Path) -> bool:
	obj = read_done_json(path, on_missing='empty', on_error='empty')
	status = str(obj.get('status', '')).lower()
	return status in {'downloaded', 'exists', 'satisfied_by_near200'}


def summarize_event_dir(
	event_dir: Path, *, cont_subdir: str = 'continuous'
) -> dict[str, object]:
	row: dict[str, object] = {
		'event_name': event_dir.name,
		'event_dir': str(event_dir),
		'overall_step': '',
		'error': '',
	}

	evt_files = sorted(event_dir.glob('*.evt'))
	evt_path = evt_files[0] if len(evt_files) == 1 else None
	stem = evt_path.stem if evt_path else ''

	row['evt_file'] = evt_path.name if evt_path else ''
	row['step1_evt_exists'] = 1 if evt_path else 0

	ch_path = event_dir / f'{stem}.ch' if stem else None
	txt_path = event_dir / f'{stem}.txt' if stem else None
	row['step1_ch_exists'] = 1 if (ch_path and ch_path.is_file()) else 0
	row['step1_txt_exists'] = 1 if (txt_path and txt_path.is_file()) else 0

	row['origin_jst'] = ''
	row['event_month'] = ''
	row['event_lat'] = ''
	row['event_lon'] = ''
	if txt_path and txt_path.is_file():
		meta = read_event_txt_meta(txt_path)
		row['origin_jst'] = meta.origin_jst.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
		row['event_month'] = meta.event_month
		row['event_lat'] = f'{meta.lat:.6f}'
		row['event_lon'] = f'{meta.lon:.6f}'

	active_ch = (event_dir / f'{stem}_active.ch') if stem else None
	if not (active_ch and active_ch.is_file()):
		cand = sorted(event_dir.glob('*_active.ch'))
		active_ch = cand[0] if len(cand) == 1 else None

	row['active_ch_file'] = active_ch.name if active_ch else ''
	row['step1_active_ch_exists'] = 1 if active_ch else 0
	if active_ch:
		n_ch, axis_map = _axis_map_from_ch(active_ch)
		row['n_active_channels'] = n_ch
		row['n_active_stations'] = len(axis_map)
		row['n_active_stations_3comp'] = sum(
			1 for sta in axis_map if axis_map[sta] >= {'U', 'N', 'E'}
		)
	else:
		row['n_active_channels'] = 0
		row['n_active_stations'] = 0
		row['n_active_stations_3comp'] = 0

	mapping_log = (event_dir / f'{stem}_mapping_log.csv') if stem else None
	row['mapping_log_exists'] = 1 if (mapping_log and mapping_log.is_file()) else 0
	row['n_mapping_rows'] = 0
	row['n_mapping_mapped'] = 0
	row['n_mapping_unmapped'] = 0
	if mapping_log and mapping_log.is_file():
		n_rows, n_mapped, n_unmapped = _read_mapping_log_counts(mapping_log)
		row['n_mapping_rows'] = n_rows
		row['n_mapping_mapped'] = n_mapped
		row['n_mapping_unmapped'] = n_unmapped

	missing_path = (event_dir / f'{stem}_missing_continuous.txt') if stem else None
	row['missing_exists'] = 1 if (missing_path and missing_path.is_file()) else 0

	missing_by_net: dict[str, list[str]] = {}
	if missing_path and missing_path.is_file():
		missing_by_net = read_missing_by_network(
			missing_path,
			normalize_station=normalize_code,
			normalize_network=normalize_network_code,
		)

	row['n_missing_pairs'] = sum(len(v) for v in missing_by_net.values())
	row['missing_networks'] = ','.join(sorted(missing_by_net.keys()))

	step2_expected = sorted(missing_by_net.keys())
	step2_done = []
	step2_pending = []
	for net in step2_expected:
		if not stem:
			step2_pending.append(net)
			continue
		cands = sorted(event_dir.glob(f'{stem}_continuous_done_*_{net}.json'))
		if any(_continuous_done_ok(p) for p in cands):
			step2_done.append(net)
		else:
			step2_pending.append(net)

	if not step2_expected:
		row['step2_status'] = 'not_needed'
	elif not step2_done:
		row['step2_status'] = 'pending'
	elif len(step2_done) == len(step2_expected):
		row['step2_status'] = 'done'
	else:
		row['step2_status'] = 'partial'

	row['step2_done_networks'] = ','.join(step2_done)
	row['step2_pending_networks'] = ','.join(step2_pending)

	cont_dir = event_dir / cont_subdir
	row['continuous_dir_exists'] = 1 if cont_dir.is_dir() else 0
	row['n_cont_cnt_files'] = (
		len(list(cont_dir.glob('*.cnt'))) if cont_dir.is_dir() else 0
	)
	row['n_cont_ch_files'] = (
		len(list(cont_dir.glob('*.ch'))) if cont_dir.is_dir() else 0
	)

	cont_channels_total = 0
	cont_axis_map: dict[str, set[str]] = {}
	if cont_dir.is_dir():
		for chp in sorted(cont_dir.glob('*.ch')):
			n_ch, axis_map = _axis_map_from_ch(chp)
			cont_channels_total += n_ch
			_merge_axis_maps(cont_axis_map, axis_map)

	row['n_cont_channels_total'] = cont_channels_total
	row['n_cont_stations_union'] = len(cont_axis_map)
	row['n_cont_stations_3comp_union'] = sum(
		1 for sta in cont_axis_map if cont_axis_map[sta] >= {'U', 'N', 'E'}
	)

	fill_done = (
		_pick_best(sorted(event_dir.glob(f'{stem}_fill_to_48_done_*.json')))
		if stem
		else None
	)
	row['fill_done_exists'] = 1 if fill_done else 0
	row['fill_done_file'] = fill_done.name if fill_done else ''
	row['fill_status'] = ''
	row['fill_n_after'] = 0
	row['fill_done_ok'] = 0
	if fill_done:
		obj = read_json(fill_done, encoding='utf-8', errors='strict')
		row['fill_status'] = str(obj.get('status', ''))
		row['fill_n_after'] = int(obj.get('n_after', 0))
		row['fill_done_ok'] = 1 if _fill_done_ok(fill_done) else 0

	total_axis_map: dict[str, set[str]] = {}
	if active_ch:
		_, am = _axis_map_from_ch(active_ch)
		_merge_axis_maps(total_axis_map, am)
	if cont_axis_map:
		_merge_axis_maps(total_axis_map, cont_axis_map)

	n_sta3 = sum(1 for sta in total_axis_map if total_axis_map[sta] >= {'U', 'N', 'E'})
	row['n_stations_3comp_union'] = n_sta3
	row['n_traces_3comp_union'] = n_sta3 * 3

	if row['step1_evt_exists'] == 0:
		row['overall_step'] = 'no_evt'
	elif row['fill_done_ok'] == 1:
		row['overall_step'] = 'step3_done'
	elif row['missing_exists'] == 1:
		row['overall_step'] = f'step2_{row["step2_status"]}'
	elif row['step1_active_ch_exists'] == 1:
		row['overall_step'] = 'step1_done'
	else:
		row['overall_step'] = 'step1_partial'

	return row


def summarize_event_root(
	event_root: Path, *, cont_subdir: str = 'continuous', strict: bool = False
) -> list[dict[str, object]]:
	if not event_root.is_dir():
		raise FileNotFoundError(event_root)

	event_dirs = sorted([p for p in event_root.iterdir() if p.is_dir()])
	if not event_dirs:
		raise RuntimeError(f'no event dirs under: {event_root}')

	rows: list[dict[str, object]] = []
	for d in event_dirs:
		if strict:
			rows.append(summarize_event_dir(d, cont_subdir=cont_subdir))
		else:
			try:
				rows.append(summarize_event_dir(d, cont_subdir=cont_subdir))
			except Exception as e:
				rows.append(
					{
						'event_name': d.name,
						'event_dir': str(d),
						'overall_step': 'error',
						'error': f'event_summarize_error={e!r}',
						'n_traces_3comp_union': 0,
						'n_stations_3comp_union': 0,
					}
				)
	return rows


def write_csv(out_path: Path, rows: list[dict[str, object]]) -> None:
	if not rows:
		raise RuntimeError('no rows to write')
	out_path.parent.mkdir(parents=True, exist_ok=True)

	cols = sorted(set().union(*(r.keys() for r in rows)))
	with out_path.open('w', encoding='utf-8', errors='strict', newline='') as f:
		w = csv.DictWriter(f, fieldnames=cols)
		w.writeheader()
		for r in rows:
			w.writerow({k: r.get(k, '') for k in cols})


EVENT_ROOT = Path('/workspace/data/waveform/jma/event')
OUT_CSV = EVENT_ROOT / 'event_progress_summary.csv'
CONT_SUBDIR = 'continuous'
STRICT = False

rows = summarize_event_root(EVENT_ROOT, cont_subdir=CONT_SUBDIR, strict=STRICT)
write_csv(OUT_CSV, rows)


df = pd.read_csv(OUT_CSV)
df[
	[
		'event_name',
		'overall_step',
		'n_traces_3comp_union',
		'fill_status',
		'fill_n_after',
		'step2_status',
	]
].head(20)
