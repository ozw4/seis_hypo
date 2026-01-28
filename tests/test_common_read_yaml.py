"""Tests for common.read_yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from common.read_yaml import fieldnames, read_yaml_preset


def test_read_yaml_preset_file_missing(tmp_path: Path) -> None:
	p = tmp_path / 'cfg.yaml'
	with pytest.raises(FileNotFoundError):
		read_yaml_preset(p, 'p')


def test_read_yaml_preset_root_not_mapping(write_text):
	p = write_text('cfg.yaml', '- 1\n- 2\n')
	with pytest.raises(ValueError):
		read_yaml_preset(p, 'p')


def test_read_yaml_preset_missing_preset(write_text):
	p = write_text('cfg.yaml', 'a: {x: 1}\n')
	with pytest.raises(KeyError):
		read_yaml_preset(p, 'p')


def test_read_yaml_preset_value_not_mapping(write_text):
	p = write_text('cfg.yaml', 'p: 123\n')
	with pytest.raises(ValueError):
		read_yaml_preset(p, 'p')


def test_read_yaml_preset_ok(write_text):
	p = write_text('cfg.yaml', 'p: {x: 1, y: 2}\n')
	out = read_yaml_preset(p, 'p')
	assert out == {'x': 1, 'y': 2}


@dataclass(frozen=True)
class _D:
	a: int
	b: str


def test_fieldnames_dataclass() -> None:
	assert fieldnames(_D) == {'a', 'b'}
