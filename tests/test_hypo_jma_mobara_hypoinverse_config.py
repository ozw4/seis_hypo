from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hypo.jma_mobara_hypoinverse_config import load_jma_mobara_hypoinverse_config


def _touch_file(path: Path) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text('', encoding='utf-8')
	return path


def _base_config_dict(tmp_path: Path) -> dict:
	files = {
		'sta_file': _touch_file(tmp_path / 'inputs' / 'stations.sta'),
		'pcrh_file': _touch_file(tmp_path / 'inputs' / 'P.crh'),
		'scrh_file': _touch_file(tmp_path / 'inputs' / 'S.crh'),
		'exe_file': _touch_file(tmp_path / 'inputs' / 'hypoinverse.exe'),
		'cmd_file': _touch_file(tmp_path / 'inputs' / 'template.cmd'),
		'measurement_csv': _touch_file(tmp_path / 'inputs' / 'measurement.csv'),
		'das_measurement_csv': _touch_file(
			tmp_path / 'inputs' / 'das_measurement.csv'
		),
		'das_epicenter_csv': _touch_file(tmp_path / 'inputs' / 'das_epicenter.csv'),
		'pick_npz': _touch_file(tmp_path / 'inputs' / 'pick_input.npz'),
		'station_with_das_csv': _touch_file(tmp_path / 'inputs' / 'station_with_das.csv'),
		'prefecture_shp': _touch_file(tmp_path / 'inputs' / 'prefecture.shp'),
		'plot_config_yaml': _touch_file(tmp_path / 'inputs' / 'plot_config.yaml'),
	}
	return {
		'paths': {
			**{key: str(path) for key, path in files.items()},
			'run_dir': str(tmp_path / 'runs' / 'mobara'),
		},
		'plot': {
			'plot_setting': 'mobara_default',
		},
		'time_filter': {
			'target_start': '2020-02-15 00:00:00',
			'target_end': '2020-03-02 00:00:00',
		},
		'initial_event': {
			'use_jma_flag': False,
			'fix_depth': False,
			'default_depth_km': 10.0,
			'p_centroid_top_n': 5,
			'origin_time_offset_sec': 3.0,
		},
		'das_filter': {
			'dt_sec': 0.01,
			'fiber_spacing_m': 1.0,
			'channel_start': 200,
			'win_half_samples': 500,
			'residual_thresh_s': 0.05,
			'spacing_m': 25.0,
		},
		'das_phase': {
			'max_dt_sec': 10.0,
		},
		'plot_quality_filter': {
			'max_erh_km': 5.0,
			'max_erz_km': 5.0,
			'max_origin_time_err_sec': None,
		},
	}


def _write_yaml(path: Path, obj: dict) -> None:
	path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding='utf-8')


def test_load_jma_mobara_hypoinverse_config_ok(tmp_path: Path) -> None:
	cfg_path = tmp_path / 'cfg.yaml'
	obj = _base_config_dict(tmp_path)
	_write_yaml(cfg_path, obj)

	cfg = load_jma_mobara_hypoinverse_config(cfg_path)

	assert cfg.paths.measurement_csv == (tmp_path / 'inputs' / 'measurement.csv').resolve()
	assert cfg.paths.run_dir == (tmp_path / 'runs' / 'mobara').resolve()
	assert cfg.plot.plot_setting == 'mobara_default'
	assert str(cfg.time_filter.target_start) == '2020-02-15 00:00:00'
	assert str(cfg.time_filter.target_end) == '2020-03-02 00:00:00'
	assert cfg.initial_event.default_depth_km == 10.0
	assert cfg.initial_event.use_jma_flag is False
	assert cfg.das_filter.spacing_m == 25.0
	assert cfg.das_phase.max_dt_sec == 10.0
	assert cfg.plot_quality_filter.max_erh_km == 5.0
	assert cfg.plot_quality_filter.max_erz_km == 5.0
	assert cfg.plot_quality_filter.max_origin_time_err_sec is None


def test_load_jma_mobara_hypoinverse_config_resolves_relative_paths_from_config_dir(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	config_dir = tmp_path / 'configs'
	inputs_dir = config_dir / 'inputs'
	runs_dir = config_dir / 'runs'
	cfg_path = config_dir / 'cfg.yaml'

	files = {
		'sta_file': _touch_file(inputs_dir / 'stations.sta'),
		'pcrh_file': _touch_file(inputs_dir / 'P.crh'),
		'scrh_file': _touch_file(inputs_dir / 'S.crh'),
		'exe_file': _touch_file(inputs_dir / 'hypoinverse.exe'),
		'cmd_file': _touch_file(inputs_dir / 'template.cmd'),
		'measurement_csv': _touch_file(inputs_dir / 'measurement.csv'),
		'das_measurement_csv': _touch_file(inputs_dir / 'das_measurement.csv'),
		'das_epicenter_csv': _touch_file(inputs_dir / 'das_epicenter.csv'),
		'pick_npz': _touch_file(inputs_dir / 'pick_input.npz'),
		'station_with_das_csv': _touch_file(inputs_dir / 'station_with_das.csv'),
		'prefecture_shp': _touch_file(inputs_dir / 'prefecture.shp'),
		'plot_config_yaml': _touch_file(inputs_dir / 'plot_config.yaml'),
	}
	obj = _base_config_dict(tmp_path)
	obj['paths'] = {
		**{key: str(path.relative_to(config_dir)) for key, path in files.items()},
		'run_dir': str((runs_dir / 'mobara').relative_to(config_dir)),
	}
	_write_yaml(cfg_path, obj)

	other_cwd = tmp_path / 'other_cwd'
	other_cwd.mkdir()
	monkeypatch.chdir(other_cwd)

	cfg = load_jma_mobara_hypoinverse_config(cfg_path)

	assert cfg.paths.measurement_csv == (inputs_dir / 'measurement.csv').resolve()
	assert cfg.paths.das_measurement_csv == (
		inputs_dir / 'das_measurement.csv'
	).resolve()
	assert cfg.paths.run_dir == (runs_dir / 'mobara').resolve()


def test_load_jma_mobara_hypoinverse_config_requires_missing_key(
	tmp_path: Path,
) -> None:
	cfg_path = tmp_path / 'cfg.yaml'
	obj = _base_config_dict(tmp_path)
	del obj['paths']['measurement_csv']
	_write_yaml(cfg_path, obj)

	with pytest.raises(KeyError, match='measurement_csv'):
		load_jma_mobara_hypoinverse_config(cfg_path)


def test_load_jma_mobara_hypoinverse_config_rejects_invalid_time_range(
	tmp_path: Path,
) -> None:
	cfg_path = tmp_path / 'cfg.yaml'
	obj = _base_config_dict(tmp_path)
	obj['time_filter']['target_start'] = '2020-03-02 00:00:00'
	obj['time_filter']['target_end'] = '2020-03-02 00:00:00'
	_write_yaml(cfg_path, obj)

	with pytest.raises(ValueError, match='target_start < target_end'):
		load_jma_mobara_hypoinverse_config(cfg_path)


def test_load_jma_mobara_hypoinverse_config_rejects_negative_default_depth(
	tmp_path: Path,
) -> None:
	cfg_path = tmp_path / 'cfg.yaml'
	obj = _base_config_dict(tmp_path)
	obj['initial_event']['default_depth_km'] = -1.0
	_write_yaml(cfg_path, obj)

	with pytest.raises(ValueError, match='default_depth_km'):
		load_jma_mobara_hypoinverse_config(cfg_path)


def test_load_jma_mobara_hypoinverse_config_rejects_non_positive_spacing(
	tmp_path: Path,
) -> None:
	cfg_path = tmp_path / 'cfg.yaml'
	obj = _base_config_dict(tmp_path)
	obj['das_filter']['spacing_m'] = 0.0
	_write_yaml(cfg_path, obj)

	with pytest.raises(ValueError, match='spacing_m'):
		load_jma_mobara_hypoinverse_config(cfg_path)


def test_load_jma_mobara_hypoinverse_config_rejects_negative_plot_quality_filter(
	tmp_path: Path,
) -> None:
	cfg_path = tmp_path / 'cfg.yaml'
	obj = _base_config_dict(tmp_path)
	obj['plot_quality_filter']['max_erh_km'] = -0.1
	_write_yaml(cfg_path, obj)

	with pytest.raises(ValueError, match='max_erh_km'):
		load_jma_mobara_hypoinverse_config(cfg_path)


def test_load_jma_mobara_hypoinverse_config_reads_origin_time_error_threshold(
	tmp_path: Path,
) -> None:
	cfg_path = tmp_path / 'cfg.yaml'
	obj = _base_config_dict(tmp_path)
	obj['plot_quality_filter']['max_origin_time_err_sec'] = 0.5
	_write_yaml(cfg_path, obj)

	cfg = load_jma_mobara_hypoinverse_config(cfg_path)

	assert cfg.plot_quality_filter.max_origin_time_err_sec == 0.5


def test_load_jma_mobara_hypoinverse_config_rejects_negative_origin_time_error_threshold(
	tmp_path: Path,
) -> None:
	cfg_path = tmp_path / 'cfg.yaml'
	obj = _base_config_dict(tmp_path)
	obj['plot_quality_filter']['max_origin_time_err_sec'] = -0.1
	_write_yaml(cfg_path, obj)

	with pytest.raises(ValueError, match='max_origin_time_err_sec'):
		load_jma_mobara_hypoinverse_config(cfg_path)
