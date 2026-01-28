from __future__ import annotations

from pathlib import Path

import yaml

from hypo.synth_eval.pipeline import load_config


def _base_cfg_dict(
	*, dataset_dir: Path, template_cmd: Path, hypoinverse_exe: Path
) -> dict:
	return {
		'dataset_dir': str(dataset_dir),
		'sim_yaml': 'sim.yaml',
		'outputs_dir': 'out',
		'template_cmd': str(template_cmd),
		'hypoinverse_exe': str(hypoinverse_exe),
		'receiver_geometry': 'recv.npy',
		'station_set': 'all',
		'lat0': 35.0,
		'lon0': 140.0,
		'origin0': '2020-01-01T00:00:00Z',
		'dt_sec': 0.01,
		'max_events': 1,
		'default_depth_km': 10.0,
		'fix_depth': False,
		'arc_use_jma_flag': False,
		'arc_p_centroid_top_n': 1,
		'arc_origin_time_offset_sec': 0.0,
	}


def _write_yaml(path: Path, obj: dict) -> None:
	path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding='utf-8')


def test_load_config_defaults_model_type_and_related_fields(tmp_path: Path) -> None:
	cfg_path = tmp_path / 'cfg.yaml'
	obj = _base_cfg_dict(
		dataset_dir=tmp_path / 'dataset',
		template_cmd=tmp_path / 'template.cmd',
		hypoinverse_exe=tmp_path / 'hyp1',
	)
	obj.pop('model_type', None)
	obj.pop('use_station_elev', None)
	obj.pop('cre_reference_margin_m', None)
	obj.pop('cre_typical_station_elevation_m', None)
	obj.pop('cre_n_layers', None)
	obj.pop('z_is_depth_positive', None)

	_write_yaml(cfg_path, obj)
	cfg = load_config(cfg_path)

	assert cfg.model_type == 'CRH'
	assert cfg.use_station_elev is False
	assert cfg.cre_reference_margin_m == 0.0
	assert cfg.cre_typical_station_elevation_m is None
	assert cfg.cre_n_layers == 1
	assert cfg.z_is_depth_positive is True


def test_load_config_normalizes_model_type_and_parses_typical_elevation(
	tmp_path: Path,
) -> None:
	cfg_path = tmp_path / 'cfg.yaml'
	obj = _base_cfg_dict(
		dataset_dir=tmp_path / 'dataset',
		template_cmd=tmp_path / 'template.cmd',
		hypoinverse_exe=tmp_path / 'hyp1',
	)
	obj['model_type'] = ' cre '
	obj['cre_typical_station_elevation_m'] = 123.4

	_write_yaml(cfg_path, obj)
	cfg = load_config(cfg_path)

	assert cfg.model_type == 'CRE'
	assert cfg.cre_typical_station_elevation_m == 123.4
	assert isinstance(cfg.cre_typical_station_elevation_m, float)


def test_load_config_rejects_invalid_model_type(tmp_path: Path) -> None:
	cfg_path = tmp_path / 'cfg.yaml'
	obj = _base_cfg_dict(
		dataset_dir=tmp_path / 'dataset',
		template_cmd=tmp_path / 'template.cmd',
		hypoinverse_exe=tmp_path / 'hyp1',
	)
	obj['model_type'] = 'XYZ'

	_write_yaml(cfg_path, obj)

	try:
		load_config(cfg_path)
		raise AssertionError('expected ValueError')
	except ValueError:
		pass


def test_load_config_default_use_station_elev_depends_on_model_type(
	tmp_path: Path,
) -> None:
	cfg_path = tmp_path / 'cfg.yaml'
	obj = _base_cfg_dict(
		dataset_dir=tmp_path / 'dataset',
		template_cmd=tmp_path / 'template.cmd',
		hypoinverse_exe=tmp_path / 'hyp1',
	)

	obj['model_type'] = 'CRE'
	obj.pop('use_station_elev', None)
	_write_yaml(cfg_path, obj)
	cfg = load_config(cfg_path)
	assert cfg.use_station_elev is True

	obj['model_type'] = 'CRH'
	obj.pop('use_station_elev', None)
	_write_yaml(cfg_path, obj)
	cfg = load_config(cfg_path)
	assert cfg.use_station_elev is False
