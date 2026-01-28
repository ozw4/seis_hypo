"""Tests for common.done_marker."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.done_marker import read_done_json, should_skip_done, write_done_json


def test_read_done_json_missing_empty(tmp_path: Path) -> None:
	p = tmp_path / 'done.json'
	assert read_done_json(p, on_missing='empty') == {}


def test_read_done_json_missing_raise(tmp_path: Path) -> None:
	p = tmp_path / 'done.json'
	with pytest.raises(FileNotFoundError):
		read_done_json(p, on_missing='raise')


def test_read_done_json_invalid_on_error_empty(tmp_path: Path) -> None:
	p = tmp_path / 'done.json'
	p.write_text('{invalid json}', encoding='utf-8')
	assert read_done_json(p, on_error='empty') == {}


def test_read_done_json_invalid_on_error_raise(tmp_path: Path) -> None:
	p = tmp_path / 'done.json'
	p.write_text('{invalid json}', encoding='utf-8')
	with pytest.raises(json.JSONDecodeError):
		read_done_json(p, on_error='raise')


def test_write_done_json_appends_newline(tmp_path: Path) -> None:
	p = tmp_path / 'nested' / 'done.json'
	write_done_json(p, {'status': 'ok'})
	text = p.read_text(encoding='utf-8')
	assert text.endswith('\n')


def test_should_skip_done_run_tag_mismatch() -> None:
	data = {'run_tag': 'A', 'status': 'ok'}
	assert should_skip_done(data, run_tag='B', ok_statuses={'ok'}) is False


def test_should_skip_done_ok_statuses_none() -> None:
	data = {'run_tag': 'A', 'status': 'bad'}
	assert should_skip_done(data, run_tag='A', ok_statuses=None) is True


def test_should_skip_done_ok_statuses_match() -> None:
	data = {'run_tag': 'A', 'status': 'ok'}
	assert should_skip_done(data, run_tag='A', ok_statuses={'ok'}) is True


def test_should_skip_done_ok_statuses_no_match() -> None:
	data = {'run_tag': 'A', 'status': 'bad'}
	assert should_skip_done(data, run_tag='A', ok_statuses={'ok'}) is False
