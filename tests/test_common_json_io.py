"""Tests for common.json_io."""

from __future__ import annotations

from pathlib import Path

from common.json_io import read_json, write_json


def test_read_write_json_roundtrip(tmp_path: Path) -> None:
	p = tmp_path / 'a.json'
	obj = {
		's': 'あ',
		'i': 1,
		'f': 1.25,
		'list': [1, 2, 3],
		'd': {'k': 'v'},
	}
	write_json(p, obj, indent=2, ensure_ascii=False)
	out = read_json(p)
	assert out == obj
