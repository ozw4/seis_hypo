from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import hypo.jma_mobara_hypoinverse_das as mod
from hypo.jma_mobara_hypoinverse_das import (
	JmaWithDasHypoinverseDasFilter,
	JmaWithDasHypoinverseDasPhase,
	JmaWithDasHypoinverseInitialEvent,
	JmaWithDasHypoinversePlot,
	JmaWithDasHypoinverseRunConfig,
	JmaWithDasHypoinverseRunPaths,
	JmaWithDasHypoinverseSweep,
	JmaWithDasHypoinverseTimeFilter,
)


def _touch_file(path: Path, content: str = '') -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(content, encoding='utf-8')
	return path


def _write_csv(path: Path, df: pd.DataFrame) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(path, index=False)
	return path


def _make_config(tmp_path: Path) -> JmaWithDasHypoinverseRunConfig:
	template_cmd = '\n'.join(
		[
			"STA 'stations.sta'",
			"CRH 1 'P.crh'",
			"CRH 2 'S.crh'",
			'WET 1.0 0.5 0.3 0.2',
			'LOC',
			"PRT 'old.prt'",
			"SUM 'old.sum'",
			"ARC 'old.arc'",
			"PHS 'old_input.arc'",
		]
	)

	paths = JmaWithDasHypoinverseRunPaths(
		sta_file=_touch_file(tmp_path / 'inputs' / 'stations_with_das.sta'),
		station_csv=_touch_file(tmp_path / 'inputs' / 'station_with_das.csv'),
		pcrh_file=_touch_file(tmp_path / 'inputs' / 'P.crh'),
		scrh_file=_touch_file(tmp_path / 'inputs' / 'S.crh'),
		hypoinverse_exe=_touch_file(tmp_path / 'inputs' / 'hypoinverse.exe'),
		cmd_template_file=_touch_file(tmp_path / 'inputs' / 'template.cmd', template_cmd),
		epicenter_csv=_write_csv(
			tmp_path / 'inputs' / 'epic.csv',
			pd.DataFrame(
				[
					{
						'event_id': 1,
						'origin_time': '2020-02-15 00:00:10',
						'das_score': 0.5,
					},
					{
						'event_id': 2,
						'origin_time': '2020-02-20 00:00:10',
						'das_score': 0.2,
					},
				]
			),
		),
		measurement_csv=_write_csv(
			tmp_path / 'inputs' / 'meas.csv',
			pd.DataFrame(
				[
					{'event_id': 1},
					{'event_id': 2},
				]
			),
		),
		das_measurement_csv=_write_csv(
			tmp_path / 'inputs' / 'das_meas.csv',
			pd.DataFrame(
				[
					{
						'event_id': 10,
						'event_time_peak': '2020-02-15 00:00:11',
					},
					{
						'event_id': 20,
						'event_time_peak': '2020-02-20 00:00:12',
					},
				]
			),
		),
		das_epicenter_csv=_write_csv(
			tmp_path / 'inputs' / 'das_epic.csv',
			pd.DataFrame(
				[
					{
						'event_id': 10,
						'event_time': '2020-02-15 00:00:10',
					},
					{
						'event_id': 20,
						'event_time': '2020-02-20 00:00:10',
					},
				]
			),
		),
		prefecture_shp=_touch_file(tmp_path / 'inputs' / 'prefecture.shp'),
		plot_config_yaml=_touch_file(tmp_path / 'inputs' / 'plot_config.yaml'),
		run_dir=tmp_path / 'runs' / 'mobara_das',
	)

	return JmaWithDasHypoinverseRunConfig(
		paths=paths,
		time_filter=JmaWithDasHypoinverseTimeFilter(
			target_start='2020-02-15 00:00:00',
			target_end='2020-03-02 00:00:00',
			max_das_score=1,
		),
		initial_event=JmaWithDasHypoinverseInitialEvent(
			use_jma_flag=False,
			fix_depth=False,
			default_depth_km=10.0,
			p_centroid_top_n=5,
			origin_time_offset_sec=3.0,
		),
		das_filter=JmaWithDasHypoinverseDasFilter(
			dt_sec=0.01,
			fiber_spacing_m=1.0,
			channel_start=200,
			win_half_samples=500,
			residual_thresh_s=0.05,
			decimation_base_spacing_m=500.0,
		),
		das_phase=JmaWithDasHypoinverseDasPhase(max_dt_sec=10.0),
		sweep=JmaWithDasHypoinverseSweep(
			das_total_weights=(1, 3),
			use_das_channels=(5, 10),
		),
		plot=JmaWithDasHypoinversePlot(plot_setting='mobara_default'),
	)


def test_filter_epicenter_and_measurements_applies_time_and_max_das_score() -> None:
	df_epic = pd.DataFrame(
		[
			{
				'event_id': 1,
				'origin_time': '2020-02-14 23:59:59',
				'das_score': 0.1,
			},
			{
				'event_id': 2,
				'origin_time': '2020-02-15 00:00:00',
				'das_score': 0.9,
			},
			{
				'event_id': 3,
				'origin_time': '2020-02-16 00:00:00',
				'das_score': 1.1,
			},
			{
				'event_id': 4,
				'origin_time': '2020-03-02 00:00:00',
				'das_score': 0.3,
			},
		]
	)
	df_meas = pd.DataFrame(
		[
			{'event_id': 1, 'station_code': 'AAA'},
			{'event_id': 2, 'station_code': 'BBB'},
			{'event_id': 3, 'station_code': 'CCC'},
			{'event_id': 4, 'station_code': 'DDD'},
		]
	)

	filtered_epic, filtered_meas = mod._filter_epicenter_and_measurements(
		df_epic,
		df_meas,
		target_start=pd.Timestamp('2020-02-15 00:00:00'),
		target_end=pd.Timestamp('2020-03-02 00:00:00'),
		max_das_score=1.0,
	)

	assert filtered_epic['event_id'].tolist() == [2]
	assert filtered_meas['event_id'].tolist() == [2]


def test_filter_by_time_range_keeps_start_and_excludes_end() -> None:
	df = pd.DataFrame(
		[
			{'event_id': 1, 'event_time': '2020-02-15 00:00:00'},
			{'event_id': 2, 'event_time': '2020-02-20 00:00:00'},
			{'event_id': 3, 'event_time': '2020-03-02 00:00:00'},
		]
	)

	filtered = mod._filter_by_time_range(
		df,
		time_col='event_time',
		parsed_col='parsed_time',
		target_start=pd.Timestamp('2020-02-15 00:00:00'),
		target_end=pd.Timestamp('2020-03-02 00:00:00'),
	)

	assert filtered['event_id'].tolist() == [1, 2]
	assert 'parsed_time' in filtered.columns


def test_patch_wet_line_replaces_weight_and_requires_line() -> None:
	lines = ['AAA', 'WET 1.0 0.5 0.3 0.2', 'BBB']

	patched = mod._patch_wet_line(lines, codeweight=0.6)

	assert patched == ['AAA', 'WET 1.0 0.5 0.3 0.6', 'BBB']


def test_patch_wet_line_raises_when_missing() -> None:
	with pytest.raises(ValueError, match='WET'):
		mod._patch_wet_line(['AAA'], codeweight=0.6)


def test_run_single_pipeline_validates_public_parameters_before_division(
	tmp_path: Path,
) -> None:
	config = _make_config(tmp_path)

	with pytest.raises(ValueError, match='use_das_channels must be >= 1'):
		mod.run_single_pipeline(
			config,
			das_total_weight=1,
			use_das_channels=0,
		)


def test_run_parameter_sweep_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	config = _make_config(tmp_path)
	script_path = _touch_file(tmp_path / 'pipeline_with_das.py', 'print("snapshot")\n')

	arc_calls: list[dict[str, object]] = []
	joined_calls: list[dict[str, object]] = []
	plot_quality_calls: list[Path] = []
	map_calls: list[Path] = []
	subprocess_calls: list[dict[str, object]] = []

	def fake_load_plot_params(_yaml: Path, _setting: str) -> SimpleNamespace:
		return SimpleNamespace(
			lon_range=(139.0, 141.0),
			lat_range=(34.0, 36.0),
			depth_range=(0.0, 30.0),
			well_coord=(35.4, 140.1),
		)

	def fake_filter_and_decimate_das_picks(
		df_das_epic: pd.DataFrame,
		df_das_meas: pd.DataFrame,
		**kwargs: object,
	) -> pd.DataFrame:
		assert sorted(df_das_epic['event_id'].tolist()) == [10, 20]
		assert sorted(df_das_meas['event_id'].tolist()) == [10, 20]
		return pd.DataFrame(
			[
				{
					'event_id': 10,
					'channel': 205,
					'pick_time': '2020-02-15 00:00:12',
					'invalid': False,
				}
			]
		)

	def fake_extract_phase_records(_df_meas: pd.DataFrame) -> list[dict[str, object]]:
		return [
			{
				'event_id': 1,
				'station_code': 'AAA',
				'phase_type': 'P',
				'weight': 0,
				'time': pd.Timestamp('2020-02-15 00:00:12'),
			}
		]

	def fake_extract_das_phase_records(
		_df_epic: pd.DataFrame,
		_df_das_meas_filtered: pd.DataFrame,
		*,
		max_dt_sec: float,
	) -> list[dict[str, object]]:
		assert max_dt_sec == 10.0
		return [
			{
				'event_id': 1,
				'station_code': 'D0205',
				'phase_type': 'P',
				'weight': 3,
				'time': pd.Timestamp('2020-02-15 00:00:13'),
			}
		]

	def fake_write_hypoinverse_arc_from_phases(
		df_epic: pd.DataFrame,
		phases: list[dict[str, object]],
		station_csv: Path,
		output_arc: Path,
		**kwargs: object,
	) -> None:
		arc_calls.append(
			{
				'event_ids': df_epic['event_id'].tolist(),
				'phase_count': len(phases),
				'station_csv': Path(station_csv),
				'output_arc': Path(output_arc),
				'spacing': kwargs,
			}
		)
		output_arc.write_text('arc\n', encoding='utf-8')

	def fake_subprocess_run(
		args: list[str],
		*,
		stdin: object | None = None,
		cwd: Path | None = None,
		capture_output: bool | None = None,
		text: bool | None = None,
		check: bool | None = None,
		**kwargs: object,
	) -> subprocess.CompletedProcess[str]:
		del capture_output, text, check, kwargs
		assert stdin is not None
		assert cwd is not None
		cmd_text = stdin.read().decode('utf-8')
		subprocess_calls.append(
			{
				'args': args,
				'cwd': Path(cwd),
				'cmd_text': cmd_text,
			}
		)
		(Path(cwd) / 'hypoinverse_run.prt').write_text('prt\n', encoding='utf-8')
		return subprocess.CompletedProcess(args=args, returncode=0, stdout='ok', stderr='')

	def fake_build_joined_jma_hypo_csv(
		df_epic: pd.DataFrame,
		df_meas: pd.DataFrame,
		prt_path: Path,
		out_join_csv: Path,
	) -> pd.DataFrame:
		joined_calls.append(
			{
				'event_ids': df_epic['event_id'].tolist(),
				'measurement_ids': df_meas['event_id'].tolist(),
				'prt_path': Path(prt_path),
				'out_join_csv': Path(out_join_csv),
			}
		)
		df = pd.DataFrame(
			[
				{
					'origin_time_hyp': '2020-02-15 00:00:10',
					'lat_deg_hyp': 35.4,
					'lon_deg_hyp': 140.1,
					'depth_km_hyp': 10.0,
					'origin_time_jma': '2020-02-15 00:00:10',
					'lat_deg_jma': 35.4,
					'lon_deg_jma': 140.1,
					'depth_km_jma': 10.0,
					'mag1_jma': 1.2,
				}
			]
		)
		df.to_csv(out_join_csv, index=False)
		return df

	def fake_plot_event_quality(df: pd.DataFrame, *, out_dir: Path, **kwargs: object) -> None:
		del df, kwargs
		plot_quality_calls.append(Path(out_dir))

	def fake_plot_events_map_and_sections(*, out_png: Path, **kwargs: object) -> None:
		del kwargs
		map_calls.append(Path(out_png))

	monkeypatch.setattr(mod, '_load_plot_params', fake_load_plot_params)
	monkeypatch.setattr(
		'das.picks_filter.filter_and_decimate_das_picks',
		fake_filter_and_decimate_das_picks,
	)
	monkeypatch.setattr('hypo.phase_jma.extract_phase_records', fake_extract_phase_records)
	monkeypatch.setattr(
		'hypo.phase_ml_das.extract_das_phase_records',
		fake_extract_das_phase_records,
	)
	monkeypatch.setattr(
		'hypo.arc.write_hypoinverse_arc_from_phases',
		fake_write_hypoinverse_arc_from_phases,
	)
	monkeypatch.setattr(
		'hypo.join_jma_hypoinverse.build_joined_jma_hypo_csv',
		fake_build_joined_jma_hypo_csv,
	)
	monkeypatch.setattr(
		'viz.hypo.event_quality.plot_event_quality',
		fake_plot_event_quality,
	)
	monkeypatch.setattr(
		'viz.events_map.plot_events_map_and_sections',
		fake_plot_events_map_and_sections,
	)
	monkeypatch.setattr(subprocess, 'run', fake_subprocess_run)

	mod.run_parameter_sweep(config, script_path=script_path)

	expected_run_dirs = [
		tmp_path / 'runs' / 'mobara_das_wet_1_ch_5',
		tmp_path / 'runs' / 'mobara_das_wet_1_ch_10',
		tmp_path / 'runs' / 'mobara_das_wet_3_ch_5',
		tmp_path / 'runs' / 'mobara_das_wet_3_ch_10',
	]
	for run_dir in expected_run_dirs:
		assert run_dir.is_dir()
		assert (run_dir / 'bak_pipeline_with_das.py').is_file()
		assert (run_dir / 'hypoinverse_input.arc').is_file()
		assert (run_dir / 'hypoinverse_run.prt').is_file()
		assert (run_dir / 'hypoinverse_events_jma_join.csv').is_file()

	assert [call['output_arc'] for call in arc_calls] == [
		run_dir / 'hypoinverse_input.arc' for run_dir in expected_run_dirs
	]
	assert [call['out_join_csv'] for call in joined_calls] == [
		run_dir / 'hypoinverse_events_jma_join.csv' for run_dir in expected_run_dirs
	]
	assert plot_quality_calls == [run_dir / 'img' for run_dir in expected_run_dirs]
	assert map_calls == [
		path
		for run_dir in expected_run_dirs
		for path in (
			run_dir / 'img' / 'Hypoinv_event_location.png',
			run_dir / 'img' / 'jma_event_location.png',
		)
	]
	assert len(subprocess_calls) == 4
	assert "WET 1.0 0.5 0.3 0.2\n" in subprocess_calls[0]['cmd_text']
	assert "WET 1.0 0.5 0.3 0.1\n" in subprocess_calls[1]['cmd_text']
	assert "WET 1.0 0.5 0.3 0.6\n" in subprocess_calls[2]['cmd_text']
	assert "WET 1.0 0.5 0.3 0.3\n" in subprocess_calls[3]['cmd_text']


def test_pipeline_with_das_import_has_no_top_level_execution(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	module_path = Path(
		'/workspace/proc/hypocenter_determination/jma_mobara_hypoinverse/pipeline_with_das.py'
	)
	spec = importlib.util.spec_from_file_location('pipeline_with_das_import_test', module_path)
	assert spec is not None
	assert spec.loader is not None

	monkeypatch.chdir(tmp_path)

	module = importlib.util.module_from_spec(spec)
	sys.modules[spec.name] = module
	try:
		spec.loader.exec_module(module)
	finally:
		sys.modules.pop(spec.name, None)

	assert hasattr(module, 'main')
	assert not (tmp_path / 'result').exists()
