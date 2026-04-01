from __future__ import annotations

import csv
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from jma.ch_table_util import normalize_ch_table_components_to_une
from jma.prepare.event_paths import resolve_evt_and_ch, resolve_txt_for_evt
from jma.missing_continuous import find_event_id_by_origin
from jma.prepare.event_txt import read_event_txt_meta, read_origin_jst_iso
from jma.station_reader import read_hinet_channel_table
from jma.stationcode_common import normalize_code, normalize_network_code
from jma.win32_reader import (
	EvtInfo,
	get_evt_info,
	read_win32_resampled,
	scan_channel_sampling_rate_map_win32,
)
from jma_model_dataset.paths import (
	active_ch_path,
	continuous_dir,
	export_dir,
	raw_root,
)

__all__ = [
	'Export100HzPaths',
	'Export100HzResult',
	'export_event_100hz',
	'resolve_export_100hz_paths',
]

_COMPONENT_ORDER = ('U', 'N', 'E')
_EVENT_SOURCE_ID = 'evt'
_EVENT_NETWORK_CODE = 'EVT'
_EVENT_MANIFEST_FILENAME = 'event_manifest.jsonl'
_EVENT_STATION_MANIFEST_FILENAME = 'event_station_manifest.csv'
_STATION_FIELDS = [
	'station_code',
	'network_code',
	'sampling_rate_hz',
	'start_time',
	'end_time_exclusive',
	'n_samples',
	'source_kind',
	'source_id',
	'source_file',
	'channel_table_file',
	'source_start_time',
	'source_end_time_exclusive',
	'component_order',
	'u_component_raw',
	'n_component_raw',
	'e_component_raw',
	'u_ch_int',
	'n_ch_int',
	'e_ch_int',
	'u_input_sampling_rate_hz',
	'n_input_sampling_rate_hz',
	'e_input_sampling_rate_hz',
	'lat',
	'lon',
]
_EVENT_STATION_MANIFEST_FIELDS = [
	'event_id',
	'event_file_stem',
	'event_month',
	*_STATION_FIELDS,
]


@dataclass(frozen=True)
class Export100HzPaths:
	event_dir: Path
	raw_dir: Path
	evt_path: Path
	txt_path: Path
	raw_ch_path: Path
	active_path: Path
	cont_dir: Path
	outdir: Path
	waveforms_path: Path
	legacy_stations_path: Path
	legacy_metadata_path: Path


@dataclass(frozen=True)
class ExportManifestPaths:
	event_manifest_path: Path
	event_station_manifest_path: Path


@dataclass(frozen=True)
class PreparedSource:
	source_id: str
	kind: str
	network_code: str
	data_path: Path
	ch_path: Path
	info: EvtInfo
	station_order: tuple[str, ...]
	station_rows: dict[str, pd.DataFrame]


@dataclass(frozen=True)
class SelectedStation:
	station_code: str
	source_id: str
	source_kind: str
	network_code: str
	source_file: str
	channel_table_file: str
	source_start_time: str
	source_end_time_exclusive: str
	rows: pd.DataFrame


@dataclass(frozen=True)
class Export100HzResult:
	event_dir: Path
	outdir: Path
	waveforms_path: Path
	event_manifest_path: Path
	event_station_manifest_path: Path
	event_id: int
	station_count: int | None
	n_samples: int | None
	target_fs_hz: int
	start_time: dt.datetime | None
	end_time_exclusive: dt.datetime | None
	skipped: bool


@dataclass(frozen=True)
class EventIdMatch:
	event_id: int
	event_file_stem: str
	event_time: str
	event_month: str
	method: str
	source: str
	match_key: str


def _iso_datetime(value: dt.datetime) -> str:
	return value.isoformat(timespec='milliseconds')


def _iso_utc_now() -> str:
	return (
		dt.datetime.now(dt.timezone.utc)
		.replace(microsecond=0)
		.isoformat()
		.replace('+00:00', 'Z')
	)


def _relpath(base: Path, path: Path) -> str:
	try:
		return str(path.relative_to(base))
	except ValueError:
		return str(path)


def _waveforms_filename(target_fs_hz: int) -> str:
	return f'waveforms_{int(target_fs_hz)}hz.npz'


def _validate_event_month(value: str) -> str:
	value2 = str(value).strip()
	if (
		len(value2) != 7
		or value2[4] != '-'
		or not value2[:4].isdigit()
		or not value2[5:].isdigit()
	):
		raise ValueError(f'event_month must be YYYY-MM: {value!r}')
	return value2


def _resolve_export_manifest_paths(
	event_dir: Path,
	*,
	event_month: str,
) -> ExportManifestPaths:
	event_dir2 = Path(event_dir).resolve()
	month2 = _validate_event_month(event_month)
	# Monthly manifests are shared across sibling event dirs, not stored per-event.
	month_dir = (
		event_dir2.parent / 'flows' / 'jma_model_dataset' / 'export_manifests' / month2
	)
	return ExportManifestPaths(
		event_manifest_path=month_dir / _EVENT_MANIFEST_FILENAME,
		event_station_manifest_path=month_dir
		/ _EVENT_STATION_MANIFEST_FILENAME,
	)


def _parse_network_code_from_cnt_name(path: Path) -> str:
	parts = path.stem.split('_')
	if len(parts) < 5 or parts[0] != 'win':
		raise ValueError(f'unexpected continuous filename: {path.name}')
	return normalize_network_code(parts[1])


def _components_ok(rows: pd.DataFrame) -> bool:
	return list(rows['component'].tolist()) == list(_COMPONENT_ORDER)


def _build_station_rows(
	df: pd.DataFrame,
) -> tuple[tuple[str, ...], dict[str, pd.DataFrame]]:
	station_order: list[str] = []
	station_rows: dict[str, pd.DataFrame] = {}

	for station_code, rows in df.groupby('station', sort=False):
		rows2 = rows.reset_index(drop=True)
		if len(rows2) != len(_COMPONENT_ORDER):
			continue
		if not _components_ok(rows2):
			continue
		station_order.append(str(station_code))
		station_rows[str(station_code)] = rows2

	return tuple(station_order), station_rows


def _prepare_source(
	*,
	source_id: str,
	kind: str,
	network_code: str,
	data_path: Path,
	ch_path: Path,
	scan_rate_blocks: int,
) -> PreparedSource:
	info = get_evt_info(data_path, scan_rate_blocks=int(scan_rate_blocks))
	df = read_hinet_channel_table(ch_path)
	if df.empty:
		raise ValueError(f'empty channel table: {ch_path}')

	channel_filter = set(int(x) for x in df['ch_int'].astype(int).tolist())
	fs_by_ch = scan_channel_sampling_rate_map_win32(
		data_path,
		channel_filter=channel_filter,
		on_mixed='drop',
	)
	present_ch = set(int(x) for x in fs_by_ch.keys())
	df_present = df[df['ch_int'].astype(int).isin(present_ch)].copy()
	if df_present.empty:
		raise ValueError(f'no WIN32-present channels matched .ch for {data_path.name}')

	df_present['station'] = df_present['station'].astype(str).map(normalize_code)
	df_present['component_raw'] = df_present['component'].astype(str)
	df_present['input_sampling_rate_hz'] = df_present['ch_int'].map(
		lambda x: int(fs_by_ch[int(x)])
	)

	df_norm = normalize_ch_table_components_to_une(
		df_present,
		require_full_une=False,
	)
	station_order, station_rows = _build_station_rows(df_norm)

	return PreparedSource(
		source_id=str(source_id),
		kind=str(kind),
		network_code=str(network_code),
		data_path=Path(data_path),
		ch_path=Path(ch_path),
		info=info,
		station_order=station_order,
		station_rows=station_rows,
	)


def resolve_export_100hz_paths(
	event_dir: Path,
	*,
	target_fs_hz: int,
) -> Export100HzPaths:
	event_dir = Path(event_dir).resolve()
	if not event_dir.is_dir():
		raise NotADirectoryError(f'event directory not found: {event_dir}')

	target_fs_hz2 = int(target_fs_hz)
	if target_fs_hz2 <= 0:
		raise ValueError('target_fs_hz must be >= 1')

	raw_dir = raw_root(event_dir)
	if not raw_dir.is_dir():
		raise NotADirectoryError(f'raw directory not found: {raw_dir}')

	evt_path, raw_ch_path = resolve_evt_and_ch(raw_dir)
	txt_path = resolve_txt_for_evt(evt_path)
	active_path = active_ch_path(event_dir, evt_path.stem)
	if not active_path.is_file():
		raise FileNotFoundError(f'flow active .ch not found: {active_path}')

	outdir = export_dir(event_dir)
	return Export100HzPaths(
		event_dir=event_dir,
		raw_dir=raw_dir,
		evt_path=evt_path,
		txt_path=txt_path,
		raw_ch_path=raw_ch_path,
		active_path=active_path,
		cont_dir=continuous_dir(event_dir),
		outdir=outdir,
		waveforms_path=outdir / _waveforms_filename(target_fs_hz2),
		legacy_stations_path=outdir / 'stations.csv',
		legacy_metadata_path=outdir / 'metadata.json',
	)


def _list_sources(paths: Export100HzPaths) -> list[PreparedSource]:
	out = [
		_prepare_source(
			source_id=_EVENT_SOURCE_ID,
			kind='evt',
			network_code=_EVENT_NETWORK_CODE,
			data_path=paths.evt_path,
			ch_path=paths.active_path,
			scan_rate_blocks=1000,
		)
	]

	if not paths.cont_dir.is_dir():
		return out

	for cnt_path in sorted(paths.cont_dir.glob('*.cnt')):
		ch_path = cnt_path.with_suffix('.ch')
		if not ch_path.is_file():
			raise FileNotFoundError(f'missing .ch for .cnt: {cnt_path} -> {ch_path}')
		out.append(
			_prepare_source(
				source_id=f'cnt:{cnt_path.name}',
				kind='cnt',
				network_code=_parse_network_code_from_cnt_name(cnt_path),
				data_path=cnt_path,
				ch_path=ch_path,
				scan_rate_blocks=1,
			)
		)

	return out


def _select_stations(
	event_dir: Path,
	sources: list[PreparedSource],
) -> tuple[list[SelectedStation], dict[str, pd.DataFrame]]:
	selected: list[SelectedStation] = []
	selected_rows_by_source: dict[str, pd.DataFrame] = {}
	seen_stations: set[str] = set()

	for source in sources:
		rows_for_source: list[pd.DataFrame] = []
		for station_code in source.station_order:
			if station_code in seen_stations:
				continue
			rows = source.station_rows[station_code]
			rows_for_source.append(rows)
			seen_stations.add(station_code)
			selected.append(
				SelectedStation(
					station_code=station_code,
					source_id=source.source_id,
					source_kind=source.kind,
					network_code=source.network_code,
					source_file=_relpath(event_dir, source.data_path),
					channel_table_file=_relpath(event_dir, source.ch_path),
					source_start_time=_iso_datetime(source.info.start_time),
					source_end_time_exclusive=_iso_datetime(
						source.info.end_time_exclusive
					),
					rows=rows,
				)
			)
		if rows_for_source:
			selected_rows_by_source[source.source_id] = pd.concat(
				rows_for_source,
				ignore_index=True,
			)

	if not selected:
		raise ValueError(f'no exportable stations found in {event_dir}')

	return selected, selected_rows_by_source


def _resolve_common_window(
	sources: list[PreparedSource],
	selected_rows_by_source: dict[str, pd.DataFrame],
	target_fs_hz: int,
) -> tuple[dt.datetime, dt.datetime, int]:
	used_sources = [
		source for source in sources if source.source_id in selected_rows_by_source
	]
	if not used_sources:
		raise ValueError('no sources selected for export')

	start_time = max(source.info.start_time for source in used_sources)
	end_time_exclusive = min(
		source.info.end_time_exclusive for source in used_sources
	)
	if end_time_exclusive <= start_time:
		raise ValueError(
			'no common time window across selected sources: '
			f'start={start_time} end={end_time_exclusive}'
		)

	n_samples = int(
		round(
			(end_time_exclusive - start_time).total_seconds() * float(target_fs_hz)
		)
	)
	if n_samples <= 0:
		raise ValueError(
			f'common export window is empty at fs={target_fs_hz}: '
			f'start={start_time} end={end_time_exclusive}'
		)

	return start_time, end_time_exclusive, n_samples


def _crop_resampled_array(
	arr: np.ndarray,
	*,
	source_info: EvtInfo,
	start_time: dt.datetime,
	end_time_exclusive: dt.datetime,
	target_fs_hz: int,
	expected_samples: int,
) -> np.ndarray:
	start_idx = int(
		round((start_time - source_info.start_time).total_seconds() * target_fs_hz)
	)
	end_idx = int(
		round(
			(end_time_exclusive - source_info.start_time).total_seconds()
			* target_fs_hz
		)
	)
	if start_idx < 0 or end_idx <= start_idx:
		raise ValueError(
			f'invalid crop indices: start_idx={start_idx} end_idx={end_idx}'
		)
	if end_idx > arr.shape[1]:
		raise ValueError(
			f'crop exceeds source array: end_idx={end_idx} arr_n={arr.shape[1]}'
		)

	out = arr[:, start_idx:end_idx]
	if out.shape[1] != int(expected_samples):
		raise ValueError(
			f'unexpected cropped sample count: expected={expected_samples} '
			f'got={out.shape[1]}'
		)
	return out


def _read_source_waveforms(
	*,
	source: PreparedSource,
	selected_rows: pd.DataFrame,
	target_fs_hz: int,
	start_time: dt.datetime,
	end_time_exclusive: dt.datetime,
	n_samples: int,
) -> np.ndarray:
	arr = read_win32_resampled(
		source.data_path,
		selected_rows,
		target_sampling_rate_HZ=int(target_fs_hz),
		duration_SECOND=int(source.info.span_seconds),
		missing_channel_policy='raise',
	)
	if np.isnan(arr).any():
		raise ValueError(f'NaN detected in resampled waveform: {source.data_path}')

	arr2 = _crop_resampled_array(
		arr,
		source_info=source.info,
		start_time=start_time,
		end_time_exclusive=end_time_exclusive,
		target_fs_hz=int(target_fs_hz),
		expected_samples=int(n_samples),
	)
	zero_rows = np.where(np.all(arr2 == 0.0, axis=1))[0]
	if len(zero_rows) > 0:
		raise ValueError(
			f'zero waveform row(s) detected after read/crop: '
			f'source={source.data_path.name} rows={zero_rows.tolist()}'
		)
	if arr2.shape[0] % len(_COMPONENT_ORDER) != 0:
		raise ValueError(
			f'unexpected channel row count for {source.data_path.name}: '
			f'{arr2.shape[0]}'
		)

	return arr2.reshape(-1, len(_COMPONENT_ORDER), int(n_samples)).astype(
		np.float32, copy=False
	)


def _write_waveforms_npz(
	path: Path,
	*,
	waveforms: np.ndarray,
	station_codes: list[str],
	target_fs_hz: int,
	start_time: dt.datetime,
	end_time_exclusive: dt.datetime,
) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	np.savez_compressed(
		path,
		waveforms=waveforms,
		stations=np.asarray(station_codes, dtype='<U64'),
		components=np.asarray(_COMPONENT_ORDER, dtype='<U1'),
		target_fs_hz=np.asarray([int(target_fs_hz)], dtype=np.int32),
		start_time=np.asarray([_iso_datetime(start_time)], dtype='<U32'),
		end_time_exclusive=np.asarray(
			[_iso_datetime(end_time_exclusive)], dtype='<U32'
		),
	)


def _station_row(
	selected: SelectedStation,
	*,
	target_fs_hz: int,
	start_time: dt.datetime,
	end_time_exclusive: dt.datetime,
	n_samples: int,
) -> dict[str, object]:
	rows = selected.rows.reset_index(drop=True)
	return {
		'station_code': selected.station_code,
		'network_code': selected.network_code,
		'sampling_rate_hz': int(target_fs_hz),
		'start_time': _iso_datetime(start_time),
		'end_time_exclusive': _iso_datetime(end_time_exclusive),
		'n_samples': int(n_samples),
		'source_kind': selected.source_kind,
		'source_id': selected.source_id,
		'source_file': selected.source_file,
		'channel_table_file': selected.channel_table_file,
		'source_start_time': selected.source_start_time,
		'source_end_time_exclusive': selected.source_end_time_exclusive,
		'component_order': ','.join(_COMPONENT_ORDER),
		'u_component_raw': str(rows.iloc[0]['component_raw']),
		'n_component_raw': str(rows.iloc[1]['component_raw']),
		'e_component_raw': str(rows.iloc[2]['component_raw']),
		'u_ch_int': int(rows.iloc[0]['ch_int']),
		'n_ch_int': int(rows.iloc[1]['ch_int']),
		'e_ch_int': int(rows.iloc[2]['ch_int']),
		'u_input_sampling_rate_hz': int(rows.iloc[0]['input_sampling_rate_hz']),
		'n_input_sampling_rate_hz': int(rows.iloc[1]['input_sampling_rate_hz']),
		'e_input_sampling_rate_hz': int(rows.iloc[2]['input_sampling_rate_hz']),
		'lat': float(rows['lat'].astype(float).mean()),
		'lon': float(rows['lon'].astype(float).mean()),
	}


def _metadata_source_entry(event_dir: Path, source: PreparedSource) -> dict[str, object]:
	return {
		'source_id': source.source_id,
		'kind': source.kind,
		'network_code': source.network_code,
		'data_path': _relpath(event_dir, source.data_path),
		'ch_path': _relpath(event_dir, source.ch_path),
		'start_time': _iso_datetime(source.info.start_time),
		'end_time_exclusive': _iso_datetime(source.info.end_time_exclusive),
		'span_seconds': int(source.info.span_seconds),
		'sampling_rates_hz': [int(x) for x in source.info.sampling_rates_hz],
		'base_sampling_rate_hz': int(source.info.base_sampling_rate_hz),
		'station_count': int(len(source.station_order)),
	}


def _resolve_event_id_match(
	*,
	evt_path: Path,
	txt_path: Path,
	epi_df: pd.DataFrame,
	epi_source: str,
) -> EventIdMatch:
	origin_iso = read_origin_jst_iso(txt_path)
	event_meta = read_event_txt_meta(txt_path)
	epi_source2 = str(epi_source).strip()
	if epi_source2 == '':
		raise ValueError('epi_source must be non-empty')

	exact_hit = epi_df.loc[epi_df['origin_time'] == origin_iso, 'event_id']
	method = 'origin_jst_exact'
	if len(exact_hit) == 0:
		method = 'origin_jst_nearest_within_0p5s'

	event_id = find_event_id_by_origin(epi_df, origin_iso)
	return EventIdMatch(
		event_id=int(event_id),
		event_file_stem=evt_path.stem,
		event_time=origin_iso,
		event_month=event_meta.event_month,
		method=method,
		source=epi_source2,
		match_key=origin_iso,
	)


def _used_sources(
	sources: list[PreparedSource],
	selected_stations: list[SelectedStation],
) -> list[PreparedSource]:
	used_source_ids = {selected.source_id for selected in selected_stations}
	return [source for source in sources if source.source_id in used_source_ids]


def _input_files(
	*,
	paths: Export100HzPaths,
	used_sources: list[PreparedSource],
	epi_source: str,
) -> list[str]:
	input_files = [
		str(epi_source),
		_relpath(paths.event_dir, paths.evt_path),
		_relpath(paths.event_dir, paths.txt_path),
		_relpath(paths.event_dir, paths.raw_ch_path),
		_relpath(paths.event_dir, paths.active_path),
	]
	for source in used_sources:
		if source.kind == 'cnt':
			input_files.append(_relpath(paths.event_dir, source.data_path))
			input_files.append(_relpath(paths.event_dir, source.ch_path))
	return sorted(set(input_files))


def _event_manifest_record(
	*,
	paths: Export100HzPaths,
	sources: list[PreparedSource],
	selected_stations: list[SelectedStation],
	event_id_match: EventIdMatch,
	target_fs_hz: int,
	start_time: dt.datetime,
	end_time_exclusive: dt.datetime,
	n_samples: int,
) -> dict[str, object]:
	used_sources = _used_sources(sources, selected_stations)
	station_codes = [selected.station_code for selected in selected_stations]
	waveforms_relpath = _relpath(paths.event_dir, paths.waveforms_path)

	return {
		'event_id': int(event_id_match.event_id),
		'event_file_stem': event_id_match.event_file_stem,
		'event_dir': str(paths.event_dir),
		'event_time': event_id_match.event_time,
		'event_month': event_id_match.event_month,
		'event_id_match': {
			'method': event_id_match.method,
			'source': event_id_match.source,
			'match_key': event_id_match.match_key,
		},
		'export_time': _iso_utc_now(),
		'target_fs_hz': int(target_fs_hz),
		'station_count': int(len(selected_stations)),
		'n_samples': int(n_samples),
		'start_time': _iso_datetime(start_time),
		'end_time_exclusive': _iso_datetime(end_time_exclusive),
		'input_files': _input_files(
			paths=paths,
			used_sources=used_sources,
			epi_source=event_id_match.source,
		),
		'output_files': {
			'waveforms_npz': waveforms_relpath,
		},
		'sources': [
			_metadata_source_entry(paths.event_dir, source) for source in used_sources
		],
		'stations': station_codes,
	}


def _event_station_manifest_rows(
	*,
	selected_stations: list[SelectedStation],
	event_id_match: EventIdMatch,
	target_fs_hz: int,
	start_time: dt.datetime,
	end_time_exclusive: dt.datetime,
	n_samples: int,
) -> list[dict[str, object]]:
	rows: list[dict[str, object]] = []
	for selected in selected_stations:
		rows.append(
			{
				'event_id': int(event_id_match.event_id),
				'event_file_stem': event_id_match.event_file_stem,
				'event_month': event_id_match.event_month,
				**_station_row(
					selected,
					target_fs_hz=target_fs_hz,
					start_time=start_time,
					end_time_exclusive=end_time_exclusive,
					n_samples=n_samples,
				),
			}
		)
	return rows


def _event_manifest_matches(
	record: dict[str, object],
	*,
	event_id_match: EventIdMatch,
	target_fs_hz: int,
) -> bool:
	return (
		int(record['event_id']) == int(event_id_match.event_id)
		and str(record['event_file_stem']) == event_id_match.event_file_stem
		and int(record['target_fs_hz']) == int(target_fs_hz)
	)


def _read_event_manifest_record(
	path: Path,
	*,
	event_id_match: EventIdMatch,
	target_fs_hz: int,
) -> dict[str, object] | None:
	if not path.is_file():
		return None

	matches: list[dict[str, object]] = []
	with path.open('r', encoding='utf-8') as f:
		for line_no, raw_line in enumerate(f, start=1):
			line = raw_line.strip()
			if line == '':
				continue
			record = json.loads(line)
			if not isinstance(record, dict):
				raise ValueError(
					f'event manifest line {line_no} must be a JSON object: {path}'
				)
			if _event_manifest_matches(
				record,
				event_id_match=event_id_match,
				target_fs_hz=target_fs_hz,
			):
				matches.append(record)

	if len(matches) > 1:
		raise ValueError(
			'multiple event manifest records matched '
			f'event_id={event_id_match.event_id} '
			f'event_file_stem={event_id_match.event_file_stem} '
			f'target_fs_hz={target_fs_hz}: {path}'
		)
	return matches[0] if matches else None


def _normalize_event_manifest_record(
	record: dict[str, object],
) -> dict[str, object]:
	record2 = dict(record)
	record2.pop('export_time', None)
	return record2


def _normalize_station_manifest_row(
	row: dict[str, object],
) -> dict[str, str]:
	return {
		field: '' if row.get(field) is None else str(row.get(field))
		for field in _EVENT_STATION_MANIFEST_FIELDS
	}


def _read_event_station_manifest_rows(
	path: Path,
	*,
	event_id_match: EventIdMatch,
	target_fs_hz: int,
) -> list[dict[str, str]]:
	if not path.is_file():
		return []

	with path.open('r', newline='', encoding='utf-8') as f:
		reader = csv.DictReader(f)
		if reader.fieldnames is None:
			return []
		missing_fields = [
			field
			for field in _EVENT_STATION_MANIFEST_FIELDS
			if field not in reader.fieldnames
		]
		if missing_fields:
			raise ValueError(
				f'event-station manifest missing columns {missing_fields}: {path}'
			)

		rows: list[dict[str, str]] = []
		for row in reader:
			if (
				str(row['event_id']) == str(event_id_match.event_id)
				and str(row['event_file_stem']) == event_id_match.event_file_stem
				and str(row['sampling_rate_hz']) == str(int(target_fs_hz))
			):
				rows.append(
					{
						field: ''
						if row.get(field) is None
						else str(row.get(field))
						for field in _EVENT_STATION_MANIFEST_FIELDS
					}
				)
	return rows


def _append_event_manifest_record(
	path: Path,
	*,
	record: dict[str, object],
) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open('a', encoding='utf-8') as f:
		f.write(json.dumps(record, ensure_ascii=False) + '\n')


def _append_event_station_manifest_rows(
	path: Path,
	*,
	rows: list[dict[str, object]],
) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	need_header = not path.is_file() or path.stat().st_size == 0
	with path.open('a', newline='', encoding='utf-8') as f:
		writer = csv.DictWriter(f, fieldnames=_EVENT_STATION_MANIFEST_FIELDS)
		if need_header:
			writer.writeheader()
		for row in rows:
			writer.writerow(row)


def _write_export_manifests(
	*,
	manifest_paths: ExportManifestPaths,
	paths: Export100HzPaths,
	sources: list[PreparedSource],
	selected_stations: list[SelectedStation],
	event_id_match: EventIdMatch,
	target_fs_hz: int,
	start_time: dt.datetime,
	end_time_exclusive: dt.datetime,
	n_samples: int,
) -> dict[str, object]:
	event_record = _event_manifest_record(
		paths=paths,
		sources=sources,
		selected_stations=selected_stations,
		event_id_match=event_id_match,
		target_fs_hz=target_fs_hz,
		start_time=start_time,
		end_time_exclusive=end_time_exclusive,
		n_samples=n_samples,
	)
	station_rows = _event_station_manifest_rows(
		selected_stations=selected_stations,
		event_id_match=event_id_match,
		target_fs_hz=target_fs_hz,
		start_time=start_time,
		end_time_exclusive=end_time_exclusive,
		n_samples=n_samples,
	)
	normalized_station_rows = [
		_normalize_station_manifest_row(row) for row in station_rows
	]

	existing_record = _read_event_manifest_record(
		manifest_paths.event_manifest_path,
		event_id_match=event_id_match,
		target_fs_hz=target_fs_hz,
	)
	if existing_record is not None:
		if _normalize_event_manifest_record(existing_record) != _normalize_event_manifest_record(
			event_record
		):
			raise ValueError(
				'event manifest record already exists with different content: '
				f'{manifest_paths.event_manifest_path}'
			)

	existing_station_rows = _read_event_station_manifest_rows(
		manifest_paths.event_station_manifest_path,
		event_id_match=event_id_match,
		target_fs_hz=target_fs_hz,
	)
	if existing_station_rows:
		if existing_station_rows != normalized_station_rows:
			raise ValueError(
				'event-station manifest rows already exist with different content: '
				f'{manifest_paths.event_station_manifest_path}'
			)

	if not existing_station_rows:
		_append_event_station_manifest_rows(
			manifest_paths.event_station_manifest_path,
			rows=station_rows,
		)
	if existing_record is None:
		_append_event_manifest_record(
			manifest_paths.event_manifest_path,
			record=event_record,
		)

	return event_record


def _validate_existing_manifests(
	*,
	manifest_paths: ExportManifestPaths,
	paths: Export100HzPaths,
	event_id_match: EventIdMatch,
	target_fs_hz: int,
) -> dict[str, object]:
	record = _read_event_manifest_record(
		manifest_paths.event_manifest_path,
		event_id_match=event_id_match,
		target_fs_hz=target_fs_hz,
	)
	if record is None:
		raise ValueError(
			'waveform export exists but event manifest record is missing: '
			f'{manifest_paths.event_manifest_path}'
		)

	if str(record['event_month']) != event_id_match.event_month:
		raise ValueError(
			'event manifest month mismatch: '
			f'expected={event_id_match.event_month} '
			f'got={record["event_month"]}'
		)

	output_files = record.get('output_files')
	if not isinstance(output_files, dict):
		raise ValueError(
			f'event manifest output_files must be an object: {manifest_paths.event_manifest_path}'
		)
	waveforms_relpath = _relpath(paths.event_dir, paths.waveforms_path)
	if str(output_files.get('waveforms_npz')) != waveforms_relpath:
		raise ValueError(
			'event manifest waveform path mismatch: '
			f'expected={waveforms_relpath} '
			f'got={output_files.get("waveforms_npz")}'
		)

	station_rows = _read_event_station_manifest_rows(
		manifest_paths.event_station_manifest_path,
		event_id_match=event_id_match,
		target_fs_hz=target_fs_hz,
	)
	station_count = int(record['station_count'])
	if len(station_rows) != station_count:
		raise ValueError(
			'event-station manifest row count mismatch: '
			f'expected={station_count} got={len(station_rows)} '
			f'path={manifest_paths.event_station_manifest_path}'
		)

	record_stations = [str(x) for x in record.get('stations', [])]
	if record_stations:
		station_codes = [row['station_code'] for row in station_rows]
		if station_codes != record_stations:
			raise ValueError(
				'event-station manifest station order mismatch: '
				f'expected={record_stations} got={station_codes} '
				f'path={manifest_paths.event_station_manifest_path}'
			)

	return record


def _cleanup_legacy_event_export_files(paths: Export100HzPaths) -> None:
	for legacy_path in (paths.legacy_stations_path, paths.legacy_metadata_path):
		if legacy_path.is_file():
			legacy_path.unlink()


def _result_from_manifest_record(
	*,
	paths: Export100HzPaths,
	manifest_paths: ExportManifestPaths,
	record: dict[str, object],
	skipped: bool,
) -> Export100HzResult:
	return Export100HzResult(
		event_dir=paths.event_dir,
		outdir=paths.outdir,
		waveforms_path=paths.waveforms_path,
		event_manifest_path=manifest_paths.event_manifest_path,
		event_station_manifest_path=manifest_paths.event_station_manifest_path,
		event_id=int(record['event_id']),
		station_count=int(record['station_count']),
		n_samples=int(record['n_samples']),
		target_fs_hz=int(record['target_fs_hz']),
		start_time=dt.datetime.fromisoformat(str(record['start_time'])),
		end_time_exclusive=dt.datetime.fromisoformat(
			str(record['end_time_exclusive'])
		),
		skipped=skipped,
	)


def export_event_100hz(
	event_dir: Path,
	*,
	epi_df: pd.DataFrame,
	epi_source: str,
	target_fs_hz: int = 100,
	skip_if_exists: bool = True,
) -> Export100HzResult:
	paths = resolve_export_100hz_paths(event_dir, target_fs_hz=target_fs_hz)
	target_fs_hz2 = int(target_fs_hz)
	event_id_match = _resolve_event_id_match(
		evt_path=paths.evt_path,
		txt_path=paths.txt_path,
		epi_df=epi_df,
		epi_source=epi_source,
	)
	manifest_paths = _resolve_export_manifest_paths(
		paths.event_dir,
		event_month=event_id_match.event_month,
	)

	if skip_if_exists and paths.waveforms_path.is_file():
		existing_record = _validate_existing_manifests(
			manifest_paths=manifest_paths,
			paths=paths,
			event_id_match=event_id_match,
			target_fs_hz=target_fs_hz2,
		)
		_cleanup_legacy_event_export_files(paths)
		return _result_from_manifest_record(
			paths=paths,
			manifest_paths=manifest_paths,
			record=existing_record,
			skipped=True,
		)

	sources = _list_sources(paths)
	selected_stations, selected_rows_by_source = _select_stations(
		paths.event_dir,
		sources,
	)
	start_time, end_time_exclusive, n_samples = _resolve_common_window(
		sources,
		selected_rows_by_source,
		target_fs_hz2,
	)

	waveform_chunks: list[np.ndarray] = []
	for source in sources:
		if source.source_id not in selected_rows_by_source:
			continue
		waveform_chunks.append(
			_read_source_waveforms(
				source=source,
				selected_rows=selected_rows_by_source[source.source_id],
				target_fs_hz=target_fs_hz2,
				start_time=start_time,
				end_time_exclusive=end_time_exclusive,
				n_samples=n_samples,
			)
		)

	if len(waveform_chunks) == 1:
		waveforms = waveform_chunks[0]
	else:
		waveforms = np.concatenate(waveform_chunks, axis=0)

	if waveforms.shape[0] != len(selected_stations):
		raise ValueError(
			f'station count mismatch: waveforms={waveforms.shape[0]} '
			f'selected={len(selected_stations)}'
		)

	station_codes = [selected.station_code for selected in selected_stations]

	paths.outdir.mkdir(parents=True, exist_ok=True)
	_write_waveforms_npz(
		paths.waveforms_path,
		waveforms=waveforms,
		station_codes=station_codes,
		target_fs_hz=target_fs_hz2,
		start_time=start_time,
		end_time_exclusive=end_time_exclusive,
	)
	event_record = _write_export_manifests(
		manifest_paths=manifest_paths,
		paths=paths,
		sources=sources,
		selected_stations=selected_stations,
		event_id_match=event_id_match,
		target_fs_hz=target_fs_hz2,
		start_time=start_time,
		end_time_exclusive=end_time_exclusive,
		n_samples=n_samples,
	)
	_cleanup_legacy_event_export_files(paths)

	return _result_from_manifest_record(
		paths=paths,
		manifest_paths=manifest_paths,
		record=event_record,
		skipped=False,
	)
