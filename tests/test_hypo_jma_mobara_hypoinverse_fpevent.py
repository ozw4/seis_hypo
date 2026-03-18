from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import hypo.jma_mobara_hypoinverse_fpevent as mod
from hypo.jma_mobara_hypoinverse_fpevent import (
	JmaFpEventHypoinverseDasFilter,
	JmaFpEventHypoinverseDasPhase,
	JmaFpEventHypoinverseInitialEvent,
	JmaFpEventHypoinversePlot,
	JmaFpEventHypoinverseRunConfig,
	JmaFpEventHypoinverseRunPaths,
	JmaFpEventHypoinverseTimeFilter,
)


def _touch_file(path: Path, content: str = '') -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(content, encoding='utf-8')
	return path


def _write_csv(path: Path, df: pd.DataFrame) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(path, index=False)
	return path


def _write_pick_npz(path: Path) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	np.savez(
		path,
		sta_code=np.array(['AAA', 'BBB']),
		sta_lat=np.array([35.1, 35.2]),
		sta_lon=np.array([140.1, 140.2]),
	)
	return path


def _make_config(tmp_path: Path) -> JmaFpEventHypoinverseRunConfig:
	template_cmd = '\n'.join(
		[
			"STA 'stations.sta'",
			"CRH 1 'P.crh'",
			"CRH 2 'S.crh'",
			'LOC',
			"PRT 'old.prt'",
			"SUM 'old.sum'",
			"ARC 'old.arc'",
			"PHS 'old_input.arc'",
		]
	)

	paths = JmaFpEventHypoinverseRunPaths(
		sta_file=_touch_file(tmp_path / 'inputs' / 'stations_with_das.sta'),
		pcrh_file=_touch_file(tmp_path / 'inputs' / 'P.crh'),
		scrh_file=_touch_file(tmp_path / 'inputs' / 'S.crh'),
		hypoinverse_exe=_touch_file(tmp_path / 'inputs' / 'hypoinverse.exe'),
		cmd_template_file=_touch_file(
			tmp_path / 'inputs' / 'template.cmd', template_cmd
		),
		measurement_csv=_write_csv(
			tmp_path / 'inputs' / 'eqt.csv',
			pd.DataFrame(
				[
					{
						'event_id': 1,
						'station_code': 'AAA',
						'Phase': 'P',
						'pick_time': '2020-02-15 00:00:11',
						'event_time_peak': '2020-02-15 00:00:10',
						'w_conf': 0.9,
					},
					{
						'event_id': 2,
						'station_code': 'BBB',
						'Phase': 'P',
						'pick_time': '2020-02-20 00:00:11',
						'event_time_peak': '2020-02-20 00:00:10',
						'w_conf': 0.8,
					},
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
						'channel': 205,
						'pick_time': '2020-02-15 00:00:12',
						'invalid': False,
					},
					{
						'event_id': 20,
						'event_time_peak': '2020-02-20 00:00:11',
						'channel': 305,
						'pick_time': '2020-02-20 00:00:12',
						'invalid': False,
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
		pick_npz=_write_pick_npz(tmp_path / 'inputs' / 'pick.npz'),
		station_with_das_csv=_touch_file(tmp_path / 'inputs' / 'station_with_das.csv'),
		prefecture_shp=_touch_file(tmp_path / 'inputs' / 'prefecture.shp'),
		plot_config_yaml=_touch_file(tmp_path / 'inputs' / 'plot_config.yaml'),
		run_dir=tmp_path / 'runs' / 'mobara_fpevent',
	)

	return JmaFpEventHypoinverseRunConfig(
		paths=paths,
		time_filter=JmaFpEventHypoinverseTimeFilter(
			target_start='2020-02-15 00:00:00',
			target_end='2020-03-02 00:00:00',
		),
		initial_event=JmaFpEventHypoinverseInitialEvent(
			use_jma_flag=False,
			fix_depth=False,
			default_depth_km=10.0,
			p_centroid_top_n=5,
			origin_time_offset_sec=3.0,
		),
		das_filter=JmaFpEventHypoinverseDasFilter(
			dt_sec=0.01,
			fiber_spacing_m=1.0,
			channel_start=200,
			win_half_samples=500,
			residual_thresh_s=0.05,
			spacing_m=25.0,
		),
		das_phase=JmaFpEventHypoinverseDasPhase(max_dt_sec=10.0),
		plot=JmaFpEventHypoinversePlot(
			plot_setting='mobara_default',
			max_erh_km=5.0,
			max_erz_km=5.0,
			max_origin_time_err_sec=2.0,
		),
	)


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


def test_filter_plot_df_by_quality_applies_all_thresholds() -> None:
	df = pd.DataFrame(
		[
			{'seq': 1, 'ERH': 1.0, 'ERZ': 1.5, 'origin_time_err_sec': 0.5},
			{'seq': 2, 'ERH': 6.0, 'ERZ': 1.0, 'origin_time_err_sec': 0.5},
			{'seq': 3, 'ERH': 1.0, 'ERZ': 6.0, 'origin_time_err_sec': 0.5},
			{'seq': 4, 'ERH': 1.0, 'ERZ': 1.0, 'origin_time_err_sec': 3.0},
		]
	)

	filtered = mod._filter_plot_df_by_quality(
		df,
		max_erh_km=5.0,
		max_erz_km=5.0,
		max_origin_time_err_sec=2.0,
	)

	assert filtered['seq'].tolist() == [1]


def test_write_plot_filter_event_csvs_writes_expected_rows(tmp_path: Path) -> None:
	initial_event_df = pd.DataFrame(
		[
			{
				'event_id': 1,
				'origin_time': '2020-02-15 00:00:10',
				'latitude_deg': 35.1,
				'longitude_deg': 140.1,
				'depth_km': 10.0,
			},
			{
				'event_id': 2,
				'origin_time': '2020-02-20 00:00:10',
				'latitude_deg': 35.2,
				'longitude_deg': 140.2,
				'depth_km': 11.0,
			},
		]
	)
	prt_df = pd.DataFrame(
		[
			{
				'sequence_no_prt': 1,
				'id_no_prt': 1,
				'seq': 1,
				'origin_time_hyp': '2020-02-15 00:00:12',
				'lat_deg_hyp': 35.1,
				'lon_deg_hyp': 140.1,
				'depth_km_hyp': 10.0,
				'RMS': 0.2,
				'ERH': 1.0,
				'ERZ': 1.0,
				'origin_time_err_sec': 0.1,
			},
			{
				'sequence_no_prt': 2,
				'id_no_prt': 2,
				'seq': 2,
				'origin_time_hyp': '2020-02-20 00:00:12',
				'lat_deg_hyp': 35.2,
				'lon_deg_hyp': 140.2,
				'depth_km_hyp': 11.0,
				'RMS': 0.3,
				'ERH': 1.0,
				'ERZ': 1.0,
				'origin_time_err_sec': 0.2,
			},
		]
	)
	prt_plot_df = prt_df.iloc[[1]].copy()

	mod._write_plot_filter_event_csvs(
		tmp_path,
		initial_event_df=initial_event_df,
		prt_df=prt_df,
		prt_plot_df=prt_plot_df,
	)

	before_df = pd.read_csv(
		tmp_path / 'hypoinverse_events_before_plot_quality_filter.csv'
	)
	after_df = pd.read_csv(
		tmp_path / 'hypoinverse_events_after_plot_quality_filter.csv'
	)

	assert before_df['event_id'].tolist() == [1, 2]
	assert before_df['passed_plot_quality_filter'].tolist() == [False, True]
	assert after_df['event_id'].tolist() == [2]
	assert after_df['passed_plot_quality_filter'].tolist() == [True]


def test_run_pipeline_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	config = _make_config(tmp_path)
	script_path = _touch_file(
		tmp_path / 'pipeline_with_jma_fpevent.py', 'print("snapshot")\n'
	)
	config_path = _touch_file(tmp_path / 'pipeline_with_jma_fpevent.yaml', 'plot:\n')

	arc_calls: list[dict[str, object]] = []
	joined_builder_calls: list[dict[str, object]] = []
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
		assert kwargs['spacing_m'] == 25.0
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

	def fake_extract_ml_pick_phase_records(df: pd.DataFrame) -> list[dict[str, object]]:
		assert sorted(df['event_id'].tolist()) == [1, 2]
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
		df_epic: pd.DataFrame,
		df_das_meas_filtered: pd.DataFrame,
		*,
		max_dt_sec: float,
	) -> list[dict[str, object]]:
		assert df_epic['event_id'].tolist() == [1, 2]
		assert df_das_meas_filtered['event_id'].tolist() == [10]
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
				'kwargs': kwargs,
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
		return subprocess.CompletedProcess(
			args=args, returncode=0, stdout='ok', stderr=''
		)

	def fake_load_hypoinverse_summary_from_prt(prt_path: Path) -> pd.DataFrame:
		assert Path(prt_path).name == 'hypoinverse_run.prt'
		return pd.DataFrame(
			[
				{
					'sequence_no_prt': 1,
					'id_no_prt': 1,
					'seq': 1,
					'origin_time_hyp': '2020-02-15 00:00:12',
					'lat_deg_hyp': 35.1,
					'lon_deg_hyp': 140.1,
					'depth_km_hyp': 10.0,
					'RMS': 0.2,
					'ERH': 1.0,
					'ERZ': 1.0,
					'origin_time_err_sec': 0.1,
				},
				{
					'sequence_no_prt': 2,
					'id_no_prt': 2,
					'seq': 2,
					'origin_time_hyp': '2020-02-20 00:00:12',
					'lat_deg_hyp': 35.2,
					'lon_deg_hyp': 140.2,
					'depth_km_hyp': 11.0,
					'RMS': 0.3,
					'ERH': 8.0,
					'ERZ': 1.0,
					'origin_time_err_sec': 0.1,
				},
			]
		)

	def fake_build_hypoinverse_event_export_df(
		initial_event_df: pd.DataFrame,
		hyp_df: pd.DataFrame,
	) -> pd.DataFrame:
		joined_builder_calls.append(
			{
				'event_ids': initial_event_df['event_id'].tolist(),
				'seqs': hyp_df['seq'].tolist(),
			}
		)
		df = hyp_df.copy()
		df['event_id'] = df['id_no_prt']
		df['origin_time_init'] = [
			'2020-02-15 00:00:10' if seq == 1 else '2020-02-20 00:00:10'
			for seq in df['seq']
		]
		df['lat_deg_init'] = [35.1 if seq == 1 else 35.2 for seq in df['seq']]
		df['lon_deg_init'] = [140.1 if seq == 1 else 140.2 for seq in df['seq']]
		df['depth_km_init'] = [10.0 if seq == 1 else 11.0 for seq in df['seq']]
		return df

	def fake_plot_event_quality(
		df: pd.DataFrame, *, out_dir: Path, **kwargs: object
	) -> None:
		assert df['seq'].tolist() == [1]
		assert kwargs['lat_col'] == 'lat_deg_init'
		plot_quality_calls.append(Path(out_dir))

	def fake_plot_events_map_and_sections(
		*, df: pd.DataFrame, out_png: Path, **kwargs: object
	) -> None:
		assert df['seq'].tolist() == [1]
		assert kwargs['origin_time_col'] == 'origin_time_hyp'
		map_calls.append(Path(out_png))

	monkeypatch.setattr(mod, '_load_plot_params', fake_load_plot_params)
	monkeypatch.setattr(
		'das.picks_filter.filter_and_decimate_das_picks',
		fake_filter_and_decimate_das_picks,
	)
	monkeypatch.setattr(
		'hypo.phase_ml.extract_ml_pick_phase_records',
		fake_extract_ml_pick_phase_records,
	)
	monkeypatch.setattr(
		'hypo.phase_ml_das.extract_das_phase_records',
		fake_extract_das_phase_records,
	)
	monkeypatch.setattr(
		'hypo.arc.write_hypoinverse_arc_from_phases',
		fake_write_hypoinverse_arc_from_phases,
	)
	monkeypatch.setattr(
		'hypo.hypoinverse_prt.load_hypoinverse_summary_from_prt',
		fake_load_hypoinverse_summary_from_prt,
	)
	monkeypatch.setattr(
		'hypo.hypoinverse_event_export.build_hypoinverse_event_export_df',
		fake_build_hypoinverse_event_export_df,
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

	mod.run_pipeline(config, script_path=script_path, config_path=config_path)

	run_dir = config.paths.run_dir
	assert run_dir.is_dir()
	assert (run_dir / 'bak_pipeline_with_jma_fpevent.py').is_file()
	assert (run_dir / 'config_used.yaml').is_file()
	assert (run_dir / 'hypoinverse_input.arc').is_file()
	assert (run_dir / 'hypoinverse_run.prt').is_file()
	assert (run_dir / 'hypoinverse_events_jma_join.csv').is_file()
	assert (run_dir / 'hypoinverse_events_before_plot_quality_filter.csv').is_file()
	assert (run_dir / 'hypoinverse_events_after_plot_quality_filter.csv').is_file()

	assert arc_calls == [
		{
			'event_ids': [1, 2],
			'phase_count': 2,
			'station_csv': config.paths.station_with_das_csv,
			'output_arc': run_dir / 'hypoinverse_input.arc',
			'kwargs': {
				'default_depth_km': 10.0,
				'use_jma_flag': False,
				'p_centroid_top_n': 5,
				'origin_time_offset_sec': 3.0,
				'fix_depth': False,
			},
		}
	]
	assert len(joined_builder_calls) == 3
	assert plot_quality_calls == [run_dir / 'img']
	assert map_calls == [run_dir / 'img' / 'Hypoinv_event_location.png']
	assert len(subprocess_calls) == 1
	assert subprocess_calls[0]['args'] == [str(config.paths.hypoinverse_exe)]
	assert "STA '" in subprocess_calls[0]['cmd_text']


def test_pipeline_with_jma_fpevent_import_has_no_top_level_execution(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:

	repo_root = Path(__file__).resolve().parents[1]
	module_path = (
		repo_root
		/ 'proc'
		/ 'hypocenter_determination'
		/ 'jma_mobara_hypoinverse'
		/ 'pipeline_with_jma_fpevent.py'
	)
	spec = importlib.util.spec_from_file_location(
		'pipeline_with_jma_fpevent_import_test',
		module_path,
	)
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
