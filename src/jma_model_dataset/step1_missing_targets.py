from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from jma.missing_continuous import P_PHASES, S_PHASES, find_event_id_by_origin
from jma.picks import build_pick_table_for_event
from jma.prepare.event_paths import resolve_evt_and_ch, resolve_txt_for_evt
from jma.prepare.event_txt import read_origin_jst_iso
from jma.station_reader import read_hinet_channel_table
from jma.stationcode_common import normalize_code, pick_one_network_code
from jma.stationcode_mappingdb import MappingDB
from jma.stationcode_presence import PresenceDB
from jma_model_dataset.paths import (
	active_ch_path,
	mapping_log_path,
	missing_txt_path,
	raw_root,
)

__all__ = [
	'MissingTargetPaths',
	'MissingTargetResult',
	'resolve_missing_target_paths',
	'build_missing_targets_for_event',
]

MAPPING_LOG_COLUMNS = [
	'event_id',
	'event_month',
	'mea_station_code',
	'mea_norm',
	'map_status',
	'ch_station',
	'network_code',
	'map_rule',
	'candidates_in_month',
]


@dataclass(frozen=True)
class MissingTargetPaths:
	event_dir: Path
	raw_dir: Path
	evt_path: Path
	txt_path: Path
	ch_path: Path
	active_path: Path
	missing_path: Path
	mapping_log_path: Path


@dataclass(frozen=True)
class MissingTargetResult:
	event_dir: Path
	evt_path: Path
	txt_path: Path
	ch_path: Path
	active_path: Path
	missing_path: Path
	mapping_log_path: Path
	event_id: int
	origin_time: str
	event_month: str
	n_missing: int


def resolve_missing_target_paths(event_dir: Path) -> MissingTargetPaths:
	event_dir = Path(event_dir).resolve()
	if not event_dir.is_dir():
		raise NotADirectoryError(f'event directory not found: {event_dir}')

	raw_dir = raw_root(event_dir)
	if not raw_dir.is_dir():
		raise NotADirectoryError(f'raw directory not found: {raw_dir}')

	evt_path, ch_path = resolve_evt_and_ch(raw_dir)
	txt_path = resolve_txt_for_evt(evt_path)
	active_path = active_ch_path(event_dir, evt_path.stem)
	if not active_path.is_file():
		raise FileNotFoundError(f'flow active .ch not found: {active_path}')

	return MissingTargetPaths(
		event_dir=event_dir,
		raw_dir=raw_dir,
		evt_path=evt_path,
		txt_path=txt_path,
		ch_path=ch_path,
		active_path=active_path,
		missing_path=missing_txt_path(event_dir, evt_path.stem),
		mapping_log_path=mapping_log_path(event_dir, evt_path.stem),
	)


def _active_station_keys(active_path: Path) -> set[str]:
	station_df = read_hinet_channel_table(active_path)
	return set(station_df['station'].astype(str).map(normalize_code).tolist())


def _missing_pairs_by_network(
	pick_by_ch: pd.DataFrame,
	*,
	active_station_keys: set[str],
) -> dict[str, list[str]]:
	pick_idx = pick_by_ch.copy()
	pick_idx.index = pick_idx.index.astype(str).map(normalize_code)
	missing_keys = sorted(set(pick_idx.index.tolist()) - active_station_keys)

	out: dict[str, list[str]] = {}
	for ch_station in missing_keys:
		row = pick_idx.loc[ch_station]
		network_code = pick_one_network_code(row.get('preferred_network_code'))
		if not network_code:
			raise ValueError(
				f'preferred_network_code not found for missing station: {ch_station}'
			)
		out.setdefault(network_code, []).append(ch_station)

	for network_code in list(out.keys()):
		out[network_code] = sorted(out[network_code])

	return out


def _write_mapping_log(path: Path, map_log: list[dict[str, object]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	pd.DataFrame(map_log, columns=MAPPING_LOG_COLUMNS).to_csv(path, index=False)


def _write_missing_txt(path: Path, missing_by_network: dict[str, list[str]]) -> int:
	path.parent.mkdir(parents=True, exist_ok=True)

	lines: list[str] = []
	for network_code in sorted(missing_by_network):
		for ch_station in missing_by_network[network_code]:
			lines.append(f'{ch_station}\t{network_code}')

	if not lines:
		if path.is_file():
			path.unlink()
		return 0

	path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
	return len(lines)


def build_missing_targets_for_event(
	event_dir: Path,
	*,
	meas_df: pd.DataFrame,
	epi_df: pd.DataFrame,
	pdb: PresenceDB,
	mdb: MappingDB,
) -> MissingTargetResult:
	paths = resolve_missing_target_paths(event_dir)
	origin_iso = read_origin_jst_iso(paths.txt_path)
	event_month = origin_iso[:7]
	event_id = find_event_id_by_origin(epi_df, origin_iso)

	pick_by_ch, map_log = build_pick_table_for_event(
		meas_df,
		event_id=event_id,
		event_month=event_month,
		mdb=mdb,
		pdb=pdb,
		p_phases=P_PHASES,
		s_phases=S_PHASES,
	)

	missing_by_network = _missing_pairs_by_network(
		pick_by_ch,
		active_station_keys=_active_station_keys(paths.active_path),
	)
	n_missing = _write_missing_txt(paths.missing_path, missing_by_network)
	_write_mapping_log(paths.mapping_log_path, map_log)

	return MissingTargetResult(
		event_dir=paths.event_dir,
		evt_path=paths.evt_path,
		txt_path=paths.txt_path,
		ch_path=paths.ch_path,
		active_path=paths.active_path,
		missing_path=paths.missing_path,
		mapping_log_path=paths.mapping_log_path,
		event_id=int(event_id),
		origin_time=origin_iso,
		event_month=event_month,
		n_missing=n_missing,
	)
