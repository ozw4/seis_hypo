"""Tests for common.csv_util."""

from __future__ import annotations

from pathlib import Path

from common.csv_util import open_dict_writer


def test_open_dict_writer_writes_header_and_rows(tmp_path: Path) -> None:
	path = tmp_path / 'out' / 'a.csv'
	f, w = open_dict_writer(path, fieldnames=['a', 'b'], write_header=True)
	try:
		w.writerow({'a': 1, 'b': 2})
	finally:
		f.close()

	lines = path.read_text(encoding='utf-8').splitlines()
	assert lines[0] == 'a,b'
	assert lines[1] == '1,2'
