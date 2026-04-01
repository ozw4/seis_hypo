from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import pandas as pd

EVENT_MANIFEST_FILENAME = 'event_manifest.jsonl'
EVENT_STATION_MANIFEST_FILENAME = 'event_station_manifest.csv'
EVENT_STATION_MANIFEST_FIELDS = [
	'event_id',
	'event_file_stem',
	'event_month',
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
EVENT_MANIFEST_REQUIRED_FIELDS = [
	'event_id',
	'event_file_stem',
	'event_dir',
	'event_time',
	'event_month',
	'event_id_match',
	'export_time',
	'target_fs_hz',
	'station_count',
	'n_samples',
	'start_time',
	'end_time_exclusive',
	'input_files',
	'output_files',
	'sources',
]
EVENT_ID_MATCH_REQUIRED_FIELDS = ['method', 'source', 'match_key']
_EVENT_KEY_COLUMNS = ['event_id', 'event_file_stem', 'event_month', 'sampling_rate_hz']

__all__ = [
	'ManifestAuditSummary',
	'audit_export_manifest_dirs',
	'format_manifest_audit_summary',
]


@dataclass(frozen=True)
class ManifestMonthPaths:
	month_dir: Path
	event_manifest_path: Path
	event_station_manifest_path: Path


@dataclass(frozen=True)
class ManifestAuditSummary:
	month_dirs: tuple[Path, ...]
	event_count: int
	unique_event_id_count: int
	event_station_row_count: int
	station_count_min: int
	station_count_median: float
	station_count_max: int
	target_fs_hz_counts: dict[int, int]
	event_month_counts: dict[str, int]


def _resolve_month_dir(path: Path) -> ManifestMonthPaths:
	month_dir = Path(path).resolve()
	if not month_dir.is_dir():
		raise NotADirectoryError(f'manifest month directory not found: {month_dir}')

	event_manifest_path = month_dir / EVENT_MANIFEST_FILENAME
	event_station_manifest_path = month_dir / EVENT_STATION_MANIFEST_FILENAME
	if not event_manifest_path.is_file():
		raise FileNotFoundError(f'event manifest not found: {event_manifest_path}')
	if not event_station_manifest_path.is_file():
		raise FileNotFoundError(
			f'event-station manifest not found: {event_station_manifest_path}'
		)

	return ManifestMonthPaths(
		month_dir=month_dir,
		event_manifest_path=event_manifest_path,
		event_station_manifest_path=event_station_manifest_path,
	)


def _month_paths(month_dirs: list[Path]) -> list[ManifestMonthPaths]:
	if not month_dirs:
		raise ValueError('month_dirs must be non-empty')

	seen: set[Path] = set()
	out: list[ManifestMonthPaths] = []
	for path in month_dirs:
		month_paths = _resolve_month_dir(path)
		if month_paths.month_dir in seen:
			continue
		seen.add(month_paths.month_dir)
		out.append(month_paths)

	if not out:
		raise ValueError('no manifest month directories resolved')

	return sorted(out, key=lambda x: str(x.month_dir))


def _event_manifest_context(
	month_paths: ManifestMonthPaths,
) -> list[dict[str, object]]:
	rows: list[dict[str, object]] = []
	with month_paths.event_manifest_path.open('r', encoding='utf-8') as f:
		for line_no, raw_line in enumerate(f, start=1):
			line = raw_line.strip()
			if line == '':
				continue
			record = json.loads(line)
			if not isinstance(record, dict):
				raise ValueError(
					f'event manifest line {line_no} must be a JSON object: '
					f'{month_paths.event_manifest_path}'
				)
			missing_fields = [
				field
				for field in EVENT_MANIFEST_REQUIRED_FIELDS
				if field not in record
			]
			if missing_fields:
				raise ValueError(
					f'event manifest missing fields {missing_fields}: '
					f'{month_paths.event_manifest_path}:{line_no}'
				)

			event_id_match = record['event_id_match']
			if not isinstance(event_id_match, dict):
				raise ValueError(
					f'event_id_match must be an object: '
					f'{month_paths.event_manifest_path}:{line_no}'
				)
			missing_match_fields = [
				field
				for field in EVENT_ID_MATCH_REQUIRED_FIELDS
				if field not in event_id_match
			]
			if missing_match_fields:
				raise ValueError(
					f'event_id_match missing fields {missing_match_fields}: '
					f'{month_paths.event_manifest_path}:{line_no}'
				)

			output_files = record['output_files']
			if not isinstance(output_files, dict):
				raise ValueError(
					f'output_files must be an object: '
					f'{month_paths.event_manifest_path}:{line_no}'
				)
			if 'waveforms_npz' not in output_files:
				raise ValueError(
					f'output_files.waveforms_npz missing: '
					f'{month_paths.event_manifest_path}:{line_no}'
				)

			rows.append(
				{
					'event_id': int(record['event_id']),
					'event_file_stem': str(record['event_file_stem']),
					'event_dir': str(record['event_dir']),
					'event_time': str(record['event_time']),
					'event_month': str(record['event_month']),
					'target_fs_hz': int(record['target_fs_hz']),
					'station_count': int(record['station_count']),
					'n_samples': int(record['n_samples']),
					'start_time': str(record['start_time']),
					'end_time_exclusive': str(record['end_time_exclusive']),
					'waveforms_npz': str(output_files['waveforms_npz']),
					'manifest_month_dir': str(month_paths.month_dir),
					'event_manifest_path': str(month_paths.event_manifest_path),
					'event_manifest_line_no': int(line_no),
				}
			)

	if not rows:
		raise ValueError(f'event manifest is empty: {month_paths.event_manifest_path}')
	return rows


def _load_event_manifest_df(month_paths_list: list[ManifestMonthPaths]) -> pd.DataFrame:
	rows: list[dict[str, object]] = []
	for month_paths in month_paths_list:
		rows.extend(_event_manifest_context(month_paths))
	return pd.DataFrame(rows)


def _load_event_station_df(
	month_paths_list: list[ManifestMonthPaths],
) -> pd.DataFrame:
	frames: list[pd.DataFrame] = []
	for month_paths in month_paths_list:
		df = pd.read_csv(month_paths.event_station_manifest_path, low_memory=False)
		missing_fields = [
			field for field in EVENT_STATION_MANIFEST_FIELDS if field not in df.columns
		]
		if missing_fields:
			raise ValueError(
				f'event-station manifest missing columns {missing_fields}: '
				f'{month_paths.event_station_manifest_path}'
			)
		df2 = df.loc[:, EVENT_STATION_MANIFEST_FIELDS].copy()
		df2['event_id'] = df2['event_id'].astype(int)
		df2['sampling_rate_hz'] = df2['sampling_rate_hz'].astype(int)
		df2['event_file_stem'] = df2['event_file_stem'].astype(str)
		df2['event_month'] = df2['event_month'].astype(str)
		df2['station_code'] = df2['station_code'].astype(str)
		df2['manifest_month_dir'] = str(month_paths.month_dir)
		df2['event_station_manifest_path'] = str(
			month_paths.event_station_manifest_path
		)
		frames.append(df2)

	if not frames:
		raise ValueError('no event-station manifest data loaded')
	return pd.concat(frames, ignore_index=True)


def _raise_on_duplicates(
	df: pd.DataFrame,
	*,
	column: str,
) -> None:
	dups = df[df.duplicated(subset=[column], keep=False)].sort_values(
		[column, 'event_manifest_path', 'event_manifest_line_no']
	)
	if dups.empty:
		return
	row = dups.iloc[0]
	raise ValueError(
		f'duplicate {column} in event manifest: {column}={row[column]!r} '
		f'path={row["event_manifest_path"]} '
		f'line={int(row["event_manifest_line_no"])}'
	)


def _waveforms_path(event_row: pd.Series) -> Path:
	event_dir = Path(str(event_row['event_dir']))
	waveforms_npz = Path(str(event_row['waveforms_npz']))
	if waveforms_npz.is_absolute():
		return waveforms_npz
	return event_dir / waveforms_npz


def _validate_waveform_files(event_df: pd.DataFrame) -> None:
	for _, row in event_df.iterrows():
		waveforms_path = _waveforms_path(row)
		if not waveforms_path.is_file():
			raise FileNotFoundError(
				f'waveforms_npz not found: {waveforms_path} '
				f'(event_id={int(row["event_id"])}, '
				f'event_file_stem={row["event_file_stem"]})'
			)


def _validate_station_event_links(
	event_df: pd.DataFrame,
	station_df: pd.DataFrame,
) -> pd.DataFrame:
	station_refs = (
		station_df.groupby(_EVENT_KEY_COLUMNS, sort=False)
		.agg(
			station_row_count=('station_code', 'size'),
			event_station_manifest_path=('event_station_manifest_path', 'first'),
			manifest_month_dir=('manifest_month_dir', 'first'),
		)
		.reset_index()
	)

	event_keys = event_df[
		[
			'event_id',
			'event_file_stem',
			'event_month',
			'target_fs_hz',
			'station_count',
			'event_manifest_path',
			'event_manifest_line_no',
		]
	].copy()
	event_keys = event_keys.rename(columns={'target_fs_hz': 'sampling_rate_hz'})

	station_join = station_refs.merge(
		event_keys[_EVENT_KEY_COLUMNS],
		on=_EVENT_KEY_COLUMNS,
		how='left',
		indicator=True,
	)
	missing_event = station_join[station_join['_merge'] != 'both']
	if not missing_event.empty:
		row = missing_event.iloc[0]
		raise ValueError(
			'event-station manifest references event missing from event manifest: '
			f'event_id={int(row["event_id"])} '
			f'event_file_stem={row["event_file_stem"]} '
			f'event_month={row["event_month"]} '
			f'sampling_rate_hz={int(row["sampling_rate_hz"])} '
			f'path={row["event_station_manifest_path"]}'
		)

	event_join = event_keys.merge(
		station_refs,
		on=_EVENT_KEY_COLUMNS,
		how='left',
	)
	missing_station_rows = event_join[event_join['station_row_count'].isna()]
	if not missing_station_rows.empty:
		row = missing_station_rows.iloc[0]
		raise ValueError(
			'event manifest has zero station rows: '
			f'event_id={int(row["event_id"])} '
			f'event_file_stem={row["event_file_stem"]} '
			f'event_month={row["event_month"]} '
			f'path={row["event_manifest_path"]}'
		)

	station_mismatch = event_join[
		event_join['station_row_count'] != event_join['station_count']
	]
	if not station_mismatch.empty:
		row = station_mismatch.iloc[0]
		raise ValueError(
			'station row count mismatch: '
			f'event_id={int(row["event_id"])} '
			f'event_file_stem={row["event_file_stem"]} '
			f'event_month={row["event_month"]} '
			f'expected={int(row["station_count"])} '
			f'got={int(row["station_row_count"])} '
			f'path={row["event_manifest_path"]}'
		)

	return event_join


def audit_export_manifest_dirs(month_dirs: list[Path]) -> ManifestAuditSummary:
	month_paths_list = _month_paths(month_dirs)
	event_df = _load_event_manifest_df(month_paths_list)
	station_df = _load_event_station_df(month_paths_list)

	_raise_on_duplicates(event_df, column='event_id')
	_raise_on_duplicates(event_df, column='event_file_stem')
	_validate_waveform_files(event_df)
	event_join = _validate_station_event_links(event_df, station_df)

	station_counts = event_join['station_row_count'].astype(int).tolist()
	if not station_counts:
		raise ValueError('no station rows found across loaded manifests')

	return ManifestAuditSummary(
		month_dirs=tuple(month_paths.month_dir for month_paths in month_paths_list),
		event_count=int(len(event_df)),
		unique_event_id_count=int(event_df['event_id'].nunique()),
		event_station_row_count=int(len(station_df)),
		station_count_min=int(min(station_counts)),
		station_count_median=float(median(station_counts)),
		station_count_max=int(max(station_counts)),
		target_fs_hz_counts={
			int(k): int(v)
			for k, v in event_df.groupby('target_fs_hz').size().sort_index().items()
		},
		event_month_counts={
			str(k): int(v)
			for k, v in event_df.groupby('event_month').size().sort_index().items()
		},
	)


def format_manifest_audit_summary(summary: ManifestAuditSummary) -> str:
	lines = [
		f'OK: audited {len(summary.month_dirs)} manifest month dir(s)',
		f'events={summary.event_count}',
		f'unique_event_ids={summary.unique_event_id_count}',
		f'event_station_rows={summary.event_station_row_count}',
		(
			'stations_per_event='
			f'min={summary.station_count_min} '
			f'median={summary.station_count_median:g} '
			f'max={summary.station_count_max}'
		),
		'target_fs_hz_counts='
		+ ', '.join(
			f'{target_fs_hz}:{count}'
			for target_fs_hz, count in summary.target_fs_hz_counts.items()
		),
		'event_month_counts='
		+ ', '.join(
			f'{event_month}:{count}'
			for event_month, count in summary.event_month_counts.items()
		),
	]
	return '\n'.join(lines)
