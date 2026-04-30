from __future__ import annotations

# ruff: noqa: E402, INP001, PLR0913
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(_REPO_ROOT))

from proc.izu2009.loki import (
	build_event_dirs_from_gamma_izu2009 as builder,
)

_JST = dt.timezone(dt.timedelta(hours=9))


def _naive_jst(
	year: int,
	month: int,
	day: int,
	hour: int,
	minute: int,
	second: int = 0,
) -> dt.datetime:
	return dt.datetime(year, month, day, hour, minute, second, tzinfo=_JST).replace(
		tzinfo=None
	)


def test_build_win32_groups_records_ch_files_parallel_to_cnt_files(
	monkeypatch,
	tmp_path: Path,
) -> None:
	event_dir = tmp_path / 'event'
	cnt_dir = tmp_path / 'cnt'
	ch_dir = tmp_path / 'ch'
	event_dir.mkdir()
	cnt_dir.mkdir()
	ch_dir.mkdir()

	cnt_names = [
		'win_0101_200912170000_10m_aaaaaaaa.cnt',
		'win_0101_200912170010_10m_bbbbbbbb.cnt',
	]
	cnt_paths = []
	for name in cnt_names:
		path = cnt_dir / name
		path.write_bytes(b'cnt')
		cnt_paths.append(path)
		(ch_dir / f'{path.stem}.ch').write_bytes(b'ch')

	monkeypatch.setitem(builder.CH47_BASE_DIR_BY_NETWORK, '0101', ch_dir)
	monkeypatch.setattr(builder, 'USE_SYMLINK', False)
	monkeypatch.setattr(builder, '_station_set_from_ch', lambda _path: {'STA1'})

	records = [
		builder.CntRecord(
			path=cnt_paths[0],
			network_code='0101',
			start_jst=_naive_jst(2009, 12, 17, 0, 0),
			end_jst=_naive_jst(2009, 12, 17, 0, 10),
			span_min=10,
		),
		builder.CntRecord(
			path=cnt_paths[1],
			network_code='0101',
			start_jst=_naive_jst(2009, 12, 17, 0, 10),
			end_jst=_naive_jst(2009, 12, 17, 0, 20),
			span_min=10,
		),
	]
	ev_picks = pd.DataFrame(
		{
			'network_code': ['0101'],
			'station_code': ['STA1'],
		}
	)

	groups, networks = builder._build_win32_groups(  # noqa: SLF001
		event_dir,
		ev_picks,
		{'0101': records},
		_naive_jst(2009, 12, 17, 0, 9, 40),
		_naive_jst(2009, 12, 17, 0, 10, 10),
	)

	assert networks == ['0101']
	assert len(groups) == 1
	group = groups[0]
	expected_ch_names = [f'{Path(name).stem}.ch' for name in cnt_names]
	assert group['cnt_files'] == cnt_names
	assert group['ch_file'] == expected_ch_names[0]
	assert group['ch_files'] == expected_ch_names
	for name in [*cnt_names, *expected_ch_names]:
		assert (event_dir / name).is_file()
