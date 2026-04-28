# ruff: noqa: INP001
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_izu2009_gamma_stations_module():
	repo_root = Path(__file__).resolve().parents[1]
	module_path = (
		repo_root
		/ 'proc'
		/ 'izu2009'
		/ 'association'
		/ 'build_gamma_stations_izu2009.py'
	)
	spec = importlib.util.spec_from_file_location(
		'build_gamma_stations_izu2009_test', module_path
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


def test_build_gamma_stations_izu2009_uses_network_station_ids(
	tmp_path: Path,
) -> None:
	module = _load_izu2009_gamma_stations_module()
	stations_csv = tmp_path / 'stations.csv'
	stations_csv.write_text(
		'network_code,station,station_name,lat,lon,elevation_m,dist_km\n'
		'0203, N.B ,Beta,34.0,140.0,20,2.5\n'
		'0101,N.A,Alpha,35.0,139.0,-10,1.5\n',
		encoding='utf-8',
	)

	out, origin = module.build_gamma_stations_izu2009(stations_csv)

	assert origin == {'lat0_deg': 34.5, 'lon0_deg': 139.5}
	assert out['id'].tolist() == ['0101__N.A', '0203__N.B']
	assert out['station_code'].tolist() == ['N.A', 'N.B']
	assert out['z(km)'].tolist() == [0.0, 0.0]
	assert out['network_code'].tolist() == ['0101', '0203']


def test_build_gamma_stations_izu2009_rejects_duplicate_station_ids(
	tmp_path: Path,
) -> None:
	module = _load_izu2009_gamma_stations_module()
	stations_csv = tmp_path / 'stations.csv'
	stations_csv.write_text(
		'network_code,station,lat,lon,elevation_m\n'
		'0101,N.A,35.0,139.0,0\n'
		'0101,N.A,35.1,139.1,1\n',
		encoding='utf-8',
	)

	with pytest.raises(ValueError, match='duplicated station id detected'):
		module.build_gamma_stations_izu2009(stations_csv)
