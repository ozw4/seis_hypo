from __future__ import annotations

# ruff: noqa: INP001
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from io_util import stream as stream_mod
from io_util.stream import build_stream_from_downloaded_win32


def _touch(path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_bytes(b'')


def _write_event_json(
	event_dir: Path,
	*,
	cnt_files: list[str],
	ch_files: list[str],
) -> None:
	obj = {
		'event_id': 1,
		'origin_time_jst': '2009-12-17T00:09:50+09:00',
		'window': {'pre_sec': 10, 'post_sec': 20},
		'win32': {
			'format': 'groups',
			'span_min': 10,
			'threads': 1,
			'groups': [
				{
					'network_code': '0101',
					'select_used': True,
					'stations': ['STA1', 'STA2'],
					'cnt_files': cnt_files,
					'ch_file': ch_files[0],
					'ch_files': ch_files,
				}
			],
		},
	}
	(event_dir / 'event.json').write_text(json.dumps(obj), encoding='utf-8')


def _channel_table(stations: list[str]) -> pd.DataFrame:
	rows: list[dict[str, object]] = []
	ch_int = 1
	for station in stations:
		for component in ('U', 'N', 'E'):
			rows.append(
				{
					'ch_hex': f'{ch_int:04X}',
					'ch_int': ch_int,
					'conv_coeff': 1.0,
					'station': station,
					'component': component,
				}
			)
			ch_int += 1
	return pd.DataFrame(rows)


def test_build_stream_from_downloaded_win32_uses_parallel_ch_files(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	event_dir = tmp_path / '000001'
	event_dir.mkdir()
	cnt_files = [
		'win_0101_200912170000_10m_aaaaaaaa.cnt',
		'win_0101_200912170010_10m_bbbbbbbb.cnt',
	]
	ch_files = [
		'win_0101_200912170000_10m_aaaaaaaa.ch',
		'win_0101_200912170010_10m_bbbbbbbb.ch',
	]
	for name in [*cnt_files, *ch_files]:
		_touch(event_dir / name)
	_write_event_json(event_dir, cnt_files=cnt_files, ch_files=ch_files)

	def read_hinet_channel_table(path: str | Path) -> pd.DataFrame:
		name = Path(path).name
		if name == ch_files[0]:
			return _channel_table(['STA1', 'STA2'])
		if name == ch_files[1]:
			return _channel_table(['STA2', 'STA1'])
		raise AssertionError(f'unexpected ch path: {path}')

	def read_win32(
		file_path: str | Path,
		ch_df: pd.DataFrame,
		**kwargs: object,
	) -> np.ndarray:
		n = int(kwargs['base_sampling_rate_HZ']) * int(kwargs['duration_SECOND'])
		tile_offset = 0.0 if Path(file_path).name == cnt_files[0] else 100.0
		component_offset = {'U': 0.0, 'N': 1.0, 'E': 2.0}
		station_base = {'STA1': 10.0, 'STA2': 40.0}
		rows = []
		for station, component in zip(
			ch_df['station'].astype(str),
			ch_df['component'].astype(str),
			strict=False,
		):
			value = station_base[station] + component_offset[component] + tile_offset
			rows.append(np.full(n, value, dtype=np.float32))
		return np.asarray(rows, dtype=np.float32)

	monkeypatch.setattr(
		stream_mod, 'read_hinet_channel_table', read_hinet_channel_table
	)
	monkeypatch.setattr(stream_mod, 'read_win32', read_win32)

	st = stream_mod.build_stream_from_downloaded_win32(
		event_dir,
		base_sampling_rate_hz=1,
	)

	sta1_u = st.select(station='STA1', channel='HHU')[0]
	sta2_u = st.select(station='STA2', channel='HHU')[0]
	assert len(sta1_u.data) == 30
	assert np.all(sta1_u.data[:20] == 10.0)
	assert np.all(sta1_u.data[20:] == 110.0)
	assert np.all(sta2_u.data[:20] == 40.0)
	assert np.all(sta2_u.data[20:] == 140.0)


def test_build_stream_from_downloaded_win32_rejects_ch_files_length_mismatch(
	tmp_path: Path,
) -> None:
	event_dir = tmp_path / '000001'
	event_dir.mkdir()
	cnt_files = [
		'win_0101_200912170000_10m_aaaaaaaa.cnt',
		'win_0101_200912170010_10m_bbbbbbbb.cnt',
	]
	ch_files = ['win_0101_200912170000_10m_aaaaaaaa.ch']
	for name in [*cnt_files, *ch_files]:
		_touch(event_dir / name)
	_write_event_json(event_dir, cnt_files=cnt_files, ch_files=ch_files)

	with pytest.raises(ValueError, match='ch_files length must match cnt_files length'):
		build_stream_from_downloaded_win32(event_dir, base_sampling_rate_hz=1)
