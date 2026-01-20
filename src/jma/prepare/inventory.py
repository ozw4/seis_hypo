from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from jma.prepare.event_paths import resolve_evt_and_ch
from jma.station_reader import read_hinet_channel_table
from jma.stationcode_common import normalize_code, normalize_network_code
from jma.win32_reader import get_evt_info, scan_channel_sampling_rate_map_win32

DEFAULT_COMP_PRIORITY = {
	'U': ['U', 'Z'],
	'N': ['N', 'Y'],
	'E': ['E', 'X'],
}
DEFAULT_AXIS_TAIL_CHARS = set(['U', 'N', 'E', 'Z', 'X', 'Y'])


def _iso(dt0: datetime) -> str:
	return dt0.strftime('%Y-%m-%d %H:%M:%S')


def _parse_network_code_from_win_name(p: Path) -> str:
	# expected: win_{network}_{yyyymmddhhmm}_{span}m_{hash}.cnt
	parts = p.stem.split('_')
	if len(parts) < 5 or parts[0] != 'win':
		raise ValueError(f'unexpected win filename: {p.name}')
	return normalize_network_code(parts[1])


def _axis_from_component(raw: str, axis_tail_chars: set[str]) -> str:
	comp = normalize_code(raw)
	if not comp:
		return ''
	tail = comp[-1]
	if tail not in axis_tail_chars:
		return ''
	if tail == 'Z':
		return 'U'
	if tail == 'Y':
		return 'N'
	if tail == 'X':
		return 'E'
	return tail


def _component_rank(
	raw: str, axis: str, comp_priority: dict[str, list[str]]
) -> int:
	comp = normalize_code(raw)
	if not comp:
		return 99

	# exact
	if comp == axis:
		return 0

	# alias exact (Z/Y/X as U/N/E)
	for a in comp_priority.get(axis, []):
		if comp == a and a != axis:
			return 1

	# suffix (wU, xxN, WZ, ...)
	if comp.endswith(axis):
		return 2
	if axis == 'U' and comp.endswith('Z'):
		return 2
	if axis == 'N' and comp.endswith('Y'):
		return 2
	if axis == 'E' and comp.endswith('X'):
		return 2

	return 99


@dataclass(frozen=True)
class SourceSpec:
	source_id: str
	kind: str  # "evt" or "cnt"
	network_code: str
	data_path: Path
	ch_path: Path


@dataclass(frozen=True)
class Candidate:
	station: str
	axis: str  # U/N/E
	component_raw: str
	source_id: str
	kind: str
	network_code: str
	ch_int: int
	fs_hz: int
	lat: float
	lon: float
	comp_rank: int


@dataclass(frozen=True)
class InventoryResult:
	inventory: dict
	sources: list[SourceSpec]
	sources_meta: dict[str, dict]
	candidates: list[Candidate]
	station_meta: dict[str, dict]


def _list_sources(event_dir: Path, cont_subdir: str) -> list[SourceSpec]:
	evt_path, evt_ch = resolve_evt_and_ch(event_dir)

	out: list[SourceSpec] = [
		SourceSpec(
			source_id='evt',
			kind='evt',
			network_code='EVT',
			data_path=evt_path,
			ch_path=evt_ch,
		)
	]

	cont_dir = event_dir / cont_subdir
	if cont_dir.is_dir():
		for cnt_path in sorted(cont_dir.glob('*.cnt')):
			ch_path = cnt_path.with_suffix('.ch')
			if not ch_path.is_file():
				raise FileNotFoundError(
					f'missing .ch for .cnt: {cnt_path} -> {ch_path}'
				)
			net = _parse_network_code_from_win_name(cnt_path)
			sid = f'cnt:{cnt_path.name}'
			out.append(
				SourceSpec(
					source_id=sid,
					kind='cnt',
					network_code=net,
					data_path=cnt_path,
					ch_path=ch_path,
				)
			)

	return out


def _relpath(base: Path, p: Path) -> str:
	try:
		return str(p.relative_to(base))
	except ValueError:
		return str(p)


def _candidate_sort_key(c: Candidate) -> tuple:
	# source優先: evt -> cnt
	source_rank = 0 if c.kind == 'evt' else 1

	# network優先（タイブレーク用）：0101 を少し優先
	net_rank = 0 if c.network_code == '0101' else 1

	return (source_rank, c.comp_rank, net_rank, c.source_id, c.ch_int)


def _pick_best_candidate(cands: list[Candidate]) -> Candidate:
	if not cands:
		raise ValueError('empty candidates')
	return sorted(cands, key=_candidate_sort_key)[0]


def _build_inventory_for_event(
	event_dir: Path,
	*,
	cont_subdir: str,
	schema_version: str,
	evt_info_scan_rate_blocks: int,
	scan_max_second_blocks: int | None,
	comp_priority: dict[str, list[str]],
	axis_tail_chars: set[str],
) -> InventoryResult:
	if not event_dir.is_dir():
		raise FileNotFoundError(event_dir)

	sources = _list_sources(event_dir, cont_subdir)
	if not sources:
		raise RuntimeError(f'no sources found in {event_dir}')

	sources_meta: dict[str, dict] = {}
	all_candidates: list[Candidate] = []

	# station -> axis -> candidates
	cand_map: dict[str, dict[str, list[Candidate]]] = {}

	for s in sources:
		if s.kind == 'evt':
			info = get_evt_info(
				s.data_path, scan_rate_blocks=int(evt_info_scan_rate_blocks)
			)
		else:
			info = get_evt_info(s.data_path, scan_rate_blocks=1)

		fs_by_ch = scan_channel_sampling_rate_map_win32(
			s.data_path, max_second_blocks=scan_max_second_blocks
		)
		present_ch = set(int(x) for x in fs_by_ch.keys())

		df = read_hinet_channel_table(s.ch_path)
		df2 = df[df['ch_int'].isin(present_ch)]
		if df2.empty:
			raise ValueError(f'no present channels matched .ch for {s.data_path.name}')

		sources_meta[s.source_id] = {
			'start_time': _iso(info.start_time),
			'end_time_exclusive': _iso(info.end_time_exclusive),
			'n_second_blocks': int(info.n_second_blocks),
			'span_seconds': int(info.span_seconds),
			'sampling_rates_hz': list(int(x) for x in info.sampling_rates_hz),
			'base_sampling_rate_hz': int(info.base_sampling_rate_hz),
			'n_present_channels': len(present_ch),
		}

		for _, r in df2.iterrows():
			sta = normalize_code(r['station'])
			if not sta:
				continue

			comp_raw = str(r['component'])
			axis = _axis_from_component(comp_raw, axis_tail_chars)
			if axis not in ['U', 'N', 'E']:
				continue

			ch_int = int(r['ch_int'])
			fs_hz = int(fs_by_ch[ch_int])
			lat = float(r['lat'])
			lon = float(r['lon'])
			comp_rank = _component_rank(comp_raw, axis, comp_priority)

			c = Candidate(
				station=sta,
				axis=axis,
				component_raw=comp_raw,
				source_id=s.source_id,
				kind=s.kind,
				network_code=s.network_code,
				ch_int=ch_int,
				fs_hz=fs_hz,
				lat=lat,
				lon=lon,
				comp_rank=comp_rank,
			)
			all_candidates.append(c)
			cand_map.setdefault(sta, {}).setdefault(axis, []).append(c)

	# stationごとに U/N/E の最良候補を確定
	station_meta: dict[str, dict] = {}
	for sta in sorted(cand_map.keys()):
		per_axis = cand_map[sta]
		best: dict[str, dict] = {}

		for axis in ['U', 'N', 'E']:
			cands = per_axis.get(axis, [])
			if not cands:
				continue
			b = _pick_best_candidate(cands)
			best[axis] = {
				'source_id': b.source_id,
				'kind': b.kind,
				'network_code': b.network_code,
				'ch_int': b.ch_int,
				'component_raw': b.component_raw,
				'fs_hz': b.fs_hz,
				'lat': b.lat,
				'lon': b.lon,
			}

		# lat/lon は確定した軸の平均（ズレがあっても過度に増幅しない）
		ll = []
		for axis in ['U', 'N', 'E']:
			if axis in best:
				ll.append((float(best[axis]['lat']), float(best[axis]['lon'])))
		if not ll:
			continue

		lat_mean = sum(x[0] for x in ll) / float(len(ll))
		lon_mean = sum(x[1] for x in ll) / float(len(ll))

		station_meta[sta] = {
			'lat': lat_mean,
			'lon': lon_mean,
			'U': best.get('U'),
			'N': best.get('N'),
			'E': best.get('E'),
		}

	n_total = len(station_meta)
	n_usable = sum(
		1
		for sta in station_meta
		if station_meta[sta].get('U')
		and station_meta[sta].get('N')
		and station_meta[sta].get('E')
	)

	out = {
		'schema_version': schema_version,
		'event_dir': str(event_dir),
		'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
		'summary': {
			'n_sources': len(sources),
			'n_stations_total': n_total,
			'n_stations_usable_3comp': n_usable,
		},
		'sources': [
			{
				'source_id': s.source_id,
				'kind': s.kind,
				'network_code': s.network_code,
				'data_path': _relpath(event_dir, s.data_path),
				'ch_path': _relpath(event_dir, s.ch_path),
				'meta': sources_meta[s.source_id],
			}
			for s in sources
		],
		'stations': station_meta,
	}

	return InventoryResult(
		inventory=out,
		sources=sources,
		sources_meta=sources_meta,
		candidates=all_candidates,
		station_meta=station_meta,
	)


def build_inventory(
	event_dir: Path,
	*,
	cont_subdir: str = 'continuous',
	schema_version: str = 'event_inventory_v1',
	evt_info_scan_rate_blocks: int = 60,
	scan_max_second_blocks: int | None = None,
	comp_priority: dict[str, list[str]] | None = None,
	axis_tail_chars: set[str] | None = None,
) -> InventoryResult:
	if comp_priority is None:
		comp_priority = DEFAULT_COMP_PRIORITY
	if axis_tail_chars is None:
		axis_tail_chars = DEFAULT_AXIS_TAIL_CHARS

	return _build_inventory_for_event(
		event_dir,
		cont_subdir=cont_subdir,
		schema_version=schema_version,
		evt_info_scan_rate_blocks=evt_info_scan_rate_blocks,
		scan_max_second_blocks=scan_max_second_blocks,
		comp_priority=comp_priority,
		axis_tail_chars=axis_tail_chars,
	)
