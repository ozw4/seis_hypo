"""Tests for common.load_config."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

from common.load_config import load_config


@dataclass(frozen=True)
class _Cfg:
	base_dir: Path
	name: str
	out_path: Path
	maybe: Path | None = None


def test_load_config_requires_dataclass(tmp_path: Path) -> None:
	class _X:
		pass

	p = tmp_path / 'cfg.yaml'
	p.write_text('p: {a: 1}\n', encoding='utf-8')
	with pytest.raises(TypeError):
		load_config(_X, p, 'p')


def test_load_config_file_missing(tmp_path: Path) -> None:
	p = tmp_path / 'missing.yaml'
	with pytest.raises(FileNotFoundError):
		load_config(_Cfg, p, 'p')


def test_load_config_ok_templates_and_path_cast(write_text) -> None:
	yaml_text = textwrap.dedent(
		"""\
		p:
		  base_dir: /tmp/base
		  name: run01
		  out_path: "{base_dir}/out/{name}.json"
		  maybe: null
		"""
	)
	yaml_path = write_text('cfg.yaml', yaml_text)
	cfg = load_config(_Cfg, yaml_path, 'p')
	assert cfg.base_dir == Path('/tmp/base')
	assert cfg.out_path == Path('/tmp/base/out/run01.json')
	assert cfg.name == 'run01'
	assert cfg.maybe is None


def test_load_config_unknown_key(write_text) -> None:
	yaml_path = write_text(
		'cfg.yaml',
		'p: {base_dir: /tmp, name: a, out_path: x, extra: 1}\n',
	)
	with pytest.raises(ValueError):
		load_config(_Cfg, yaml_path, 'p')


def test_load_config_missing_preset(write_text) -> None:
	yaml_path = write_text(
		'cfg.yaml',
		'q: {base_dir: /tmp, name: a, out_path: x}\n',
	)
	with pytest.raises(KeyError):
		load_config(_Cfg, yaml_path, 'p')


def test_load_config_template_unknown_key(write_text) -> None:
	yaml_text = textwrap.dedent(
		"""\
		p:
		  base_dir: /tmp/base
		  name: run01
		  out_path: "{not_defined}/out/{name}.json"
		"""
	)
	yaml_path = write_text('cfg.yaml', yaml_text)
	with pytest.raises(KeyError):
		load_config(_Cfg, yaml_path, 'p')
