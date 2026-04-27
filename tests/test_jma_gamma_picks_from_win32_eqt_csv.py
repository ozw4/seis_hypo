# ruff: noqa: INP001
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_eqt2gamma_module():
	repo_root = Path(__file__).resolve().parents[1]
	module_path = (
		repo_root
		/ 'proc'
		/ 'prepare_data'
		/ 'jma'
		/ 'build_gamma_picks_from_win32_eqt_csv.py'
	)
	spec = importlib.util.spec_from_file_location(
		'build_gamma_picks_from_win32_eqt_csv_test', module_path
	)
	assert spec is not None
	assert spec.loader is not None

	module = importlib.util.module_from_spec(spec)
	sys.modules[spec.name] = module
	try:
		spec.loader.exec_module(module)
	finally:
		sys.modules.pop(spec.name, None)

	return module


def _write_pick_csv(path: Path) -> None:
	path.write_text(
		'station_code,Phase,pick_time,w_conf,network_code\n'
		'N.ITOH,P,2009-12-17 00:00:00.000,0.75,0101\n',
		encoding='utf-8',
	)


def test_network_station_separator_can_preserve_dotted_station_codes(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	eqt2gamma = _load_eqt2gamma_module()
	pick_csv = tmp_path / 'picks.csv'
	_write_pick_csv(pick_csv)
	monkeypatch.setattr(eqt2gamma, 'STATION_ID_MODE', 'network_station')
	monkeypatch.setattr(eqt2gamma, 'NETWORK_STATION_SEPARATOR', '__')
	monkeypatch.setattr(eqt2gamma, 'INCLUDE_TRACE_COLUMNS', False)

	out = eqt2gamma.build_gamma_picks_from_win32_eqt_csv([pick_csv])

	assert out['station_id'].tolist() == ['0101__N.ITOH']


def test_network_station_separator_must_be_non_empty(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	eqt2gamma = _load_eqt2gamma_module()
	pick_csv = tmp_path / 'picks.csv'
	_write_pick_csv(pick_csv)
	monkeypatch.setattr(eqt2gamma, 'STATION_ID_MODE', 'network_station')
	monkeypatch.setattr(eqt2gamma, 'NETWORK_STATION_SEPARATOR', '')

	with pytest.raises(
		ValueError, match='NETWORK_STATION_SEPARATOR must be a non-empty string'
	):
		eqt2gamma.build_gamma_picks_from_win32_eqt_csv([pick_csv])
