# %%
# proc/prepare_data/jma/run_build_event_inventory.py
from __future__ import annotations

import csv
import json
from pathlib import Path

from jma.prepare.inventory import Candidate, SourceSpec, build_inventory

# =========================
# 設定（ここを直書きでOK）
# =========================

# イベントディレクトリ（添付zipを展開したフォルダを指定）
EVENT_DIR = Path('/workspace/data/waveform/jma/event/D20230118000041_20').resolve()

# continuous サブディレクトリ名
CONT_SUBDIR = 'continuous'

# 出力先
OUTDIR = EVENT_DIR / 'inventory'

# バージョン
SCHEMA_VERSION = 'event_inventory_v1'

# get_evt_info の sampling rate 走査ブロック数（多めにしておく）
EVT_INFO_SCAN_RATE_BLOCKS = 60

# scan_channel_sampling_rate_map_win32 の secondblock 走査上限（None で全走査）
SCAN_MAX_SECOND_BLOCKS = None

# component 揺らぎ吸収の優先度（小さいほど強い）
# axis U/N/E に対して、どの表記を優先するか
COMP_PRIORITY = {
	'U': ['U', 'Z'],
	'N': ['N', 'Y'],
	'E': ['E', 'X'],
}

# 末尾1文字で軸推定を許可する文字（wU, xxN, ...）
AXIS_TAIL_CHARS = set(['U', 'N', 'E', 'Z', 'X', 'Y'])


# =========================
# 実装
# =========================

def _relpath(base: Path, p: Path) -> str:
	try:
		return str(p.relative_to(base))
	except ValueError:
		return str(p)


def _write_sources_csv(
	out_path: Path,
	*,
	event_dir: Path,
	sources: list[SourceSpec],
	sources_meta: dict[str, dict],
) -> None:
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fields = [
		'source_id',
		'kind',
		'network_code',
		'data_path',
		'ch_path',
		'start_time',
		'end_time_exclusive',
		'n_second_blocks',
		'span_seconds',
		'base_sampling_rate_hz',
		'sampling_rates_hz',
		'n_present_channels',
	]
	with out_path.open('w', newline='', encoding='utf-8') as f:
		w = csv.DictWriter(f, fieldnames=fields)
		w.writeheader()
		for s in sources:
			m = sources_meta[s.source_id]
			w.writerow(
				{
					'source_id': s.source_id,
					'kind': s.kind,
					'network_code': s.network_code,
					'data_path': _relpath(event_dir, s.data_path),
					'ch_path': _relpath(event_dir, s.ch_path),
					'start_time': m['start_time'],
					'end_time_exclusive': m['end_time_exclusive'],
					'n_second_blocks': m['n_second_blocks'],
					'span_seconds': m['span_seconds'],
					'base_sampling_rate_hz': m['base_sampling_rate_hz'],
					'sampling_rates_hz': ','.join(
						str(x) for x in m['sampling_rates_hz']
					),
					'n_present_channels': m['n_present_channels'],
				}
			)


def _write_candidates_csv(out_path: Path, candidates: list[Candidate]) -> None:
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fields = [
		'station',
		'axis',
		'component_raw',
		'comp_rank',
		'source_id',
		'kind',
		'network_code',
		'ch_int',
		'fs_hz',
		'lat',
		'lon',
	]
	with out_path.open('w', newline='', encoding='utf-8') as f:
		w = csv.DictWriter(f, fieldnames=fields)
		w.writeheader()
		for c in candidates:
			w.writerow(
				{
					'station': c.station,
					'axis': c.axis,
					'component_raw': c.component_raw,
					'comp_rank': c.comp_rank,
					'source_id': c.source_id,
					'kind': c.kind,
					'network_code': c.network_code,
					'ch_int': c.ch_int,
					'fs_hz': c.fs_hz,
					'lat': f'{c.lat:.6f}',
					'lon': f'{c.lon:.6f}',
				}
			)


def _write_stations_csv(
	out_path: Path,
	*,
	stations: list[str],
	station_meta: dict[str, dict],
) -> None:
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fields = [
		'station',
		'lat',
		'lon',
		'is_usable',
		'U_source_id',
		'U_ch_int',
		'U_component_raw',
		'U_fs_hz',
		'N_source_id',
		'N_ch_int',
		'N_component_raw',
		'N_fs_hz',
		'E_source_id',
		'E_ch_int',
		'E_component_raw',
		'E_fs_hz',
	]
	with out_path.open('w', newline='', encoding='utf-8') as f:
		w = csv.DictWriter(f, fieldnames=fields)
		w.writeheader()
		for sta in stations:
			m = station_meta[sta]
			u = m.get('U')
			n = m.get('N')
			e = m.get('E')
			is_usable = bool(u and n and e)
			w.writerow(
				{
					'station': sta,
					'lat': f'{m["lat"]:.6f}',
					'lon': f'{m["lon"]:.6f}',
					'is_usable': 1 if is_usable else 0,
					'U_source_id': '' if not u else u['source_id'],
					'U_ch_int': '' if not u else u['ch_int'],
					'U_component_raw': '' if not u else u['component_raw'],
					'U_fs_hz': '' if not u else u['fs_hz'],
					'N_source_id': '' if not n else n['source_id'],
					'N_ch_int': '' if not n else n['ch_int'],
					'N_component_raw': '' if not n else n['component_raw'],
					'N_fs_hz': '' if not n else n['fs_hz'],
					'E_source_id': '' if not e else e['source_id'],
					'E_ch_int': '' if not e else e['ch_int'],
					'E_component_raw': '' if not e else e['component_raw'],
					'E_fs_hz': '' if not e else e['fs_hz'],
				}
			)


def _build_inventory_for_event(event_dir: Path) -> dict:
	result = build_inventory(
		event_dir,
		cont_subdir=CONT_SUBDIR,
		schema_version=SCHEMA_VERSION,
		evt_info_scan_rate_blocks=EVT_INFO_SCAN_RATE_BLOCKS,
		scan_max_second_blocks=SCAN_MAX_SECOND_BLOCKS,
		comp_priority=COMP_PRIORITY,
		axis_tail_chars=AXIS_TAIL_CHARS,
	)
	sources = result.sources
	sources_meta = result.sources_meta
	all_candidates = result.candidates
	station_meta = result.station_meta
	out = result.inventory

	# 出力（JSON / CSV）
	OUTDIR.mkdir(parents=True, exist_ok=True)
	json_path = OUTDIR / f'{event_dir.name}_{SCHEMA_VERSION}.json'
	json_path.write_text(
		json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8'
	)

	_write_sources_csv(
		OUTDIR / f'{event_dir.name}_{SCHEMA_VERSION}_sources.csv',
		event_dir=event_dir,
		sources=sources,
		sources_meta=sources_meta,
	)
	_write_candidates_csv(
		OUTDIR / f'{event_dir.name}_{SCHEMA_VERSION}_candidates.csv',
		all_candidates,
	)
	_write_stations_csv(
		OUTDIR / f'{event_dir.name}_{SCHEMA_VERSION}_stations.csv',
		stations=sorted(station_meta.keys()),
		station_meta=station_meta,
	)

	n_total = out['summary']['n_stations_total']
	n_usable = out['summary']['n_stations_usable_3comp']

	print('[inventory]')
	print(f'  event_dir: {event_dir}')
	print(f'  outdir   : {OUTDIR}')
	print(f'  sources  : {len(sources)}')
	print(f'  stations : total={n_total} usable_3comp={n_usable}')
	print(f'  json     : {json_path.name}')
	print(f'  stations_csv  : {event_dir.name}_{SCHEMA_VERSION}_stations.csv')
	print(f'  sources_csv   : {event_dir.name}_{SCHEMA_VERSION}_sources.csv')
	print(f'  candidates_csv: {event_dir.name}_{SCHEMA_VERSION}_candidates.csv')

	return out


def main() -> None:
	_build_inventory_for_event(EVENT_DIR)


if __name__ == '__main__':
	main()

# %%
