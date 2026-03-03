from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import hypo.synth_eval.pipeline as pl


def _write_sim_yaml(
	path: Path, *, vp_mps: float = 6000.0, vs_mps: float = 3500.0
) -> None:
	obj = {'model': {'vp_mps': float(vp_mps), 'vs_mps': float(vs_mps)}}
	path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding='utf-8')


def _prepare_dataset(tmp_path: Path) -> tuple[Path, str, str]:
	dataset_dir = tmp_path / 'dataset'
	geom_dir = dataset_dir / 'geometry'
	events_dir = dataset_dir / 'events'
	prov_dir = dataset_dir / 'provenance'

	dataset_dir.mkdir(parents=True, exist_ok=True)
	geom_dir.mkdir(parents=True, exist_ok=True)
	events_dir.mkdir(parents=True, exist_ok=True)
	prov_dir.mkdir(parents=True, exist_ok=True)

	sim_yaml_name = 'sim.yaml'
	recv_name = 'recv.npy'
	recv_catalog_name = 'recv.receivers.csv'

	_write_sim_yaml(prov_dir / sim_yaml_name)

	np.save(geom_dir / recv_name, np.zeros((4, 3), dtype=float))
	(dataset_dir / 'dataset_meta.json').write_text(
		json.dumps(
			{'optional': {'receiver_catalog_csv_rel': f'geometry/{recv_catalog_name}'}},
			sort_keys=False,
		),
		encoding='utf-8',
		newline='\n',
	)
	(geom_dir / recv_catalog_name).write_text(
		'receiver_index,station_code\n0,G0001\n1,G0002\n2,D0001\n3,D0002\n',
		encoding='utf-8',
		newline='\n',
	)
	(dataset_dir / 'index.csv').write_text('', encoding='utf-8')

	return dataset_dir, sim_yaml_name, recv_name


def _write_config(
	path: Path,
	*,
	dataset_dir: Path,
	sim_yaml: str,
	receiver_geometry: str,
	outputs_dir: str,
	template_cmd: Path,
	hypoinverse_exe: Path,
	model_type: str,
	use_station_elev: bool | None,
	apply_station_elevation_delay: bool,
	z_is_depth_positive: bool,
	cre_reference_margin_m: float = 0.0,
	cre_typical_station_elevation_m: float | None = None,
	cre_n_layers: int = 1,
	event_filter: dict | None = None,
	event_subsample: dict | None = None,
) -> None:
	obj: dict = {
		'dataset_dir': str(dataset_dir),
		'sim_yaml': str(sim_yaml),
		'outputs_dir': str(outputs_dir),
		'template_cmd': str(template_cmd),
		'hypoinverse_exe': str(hypoinverse_exe),
		'receiver_geometry': str(receiver_geometry),
		'station_subset': {'surface_indices': 'all', 'das_indices': 'all'},
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
		'model_type': str(model_type),
		'apply_station_elevation_delay': bool(apply_station_elevation_delay),
		'z_is_depth_positive': bool(z_is_depth_positive),
		'cre_reference_margin_m': float(cre_reference_margin_m),
		'cre_typical_station_elevation_m': cre_typical_station_elevation_m,
		'cre_n_layers': int(cre_n_layers),
	}
	if use_station_elev is not None:
		obj['use_station_elev'] = bool(use_station_elev)
	if event_filter is not None:
		obj['event_filter'] = event_filter
	if event_subsample is not None:
		obj['event_subsample'] = event_subsample

	path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding='utf-8')


def _stub_eval_df() -> pd.DataFrame:
	return pd.DataFrame(
		{
			'horiz_m': [1.0],
			'dz_m': [2.0],
			'err3d_m': [3.0],
			'RMS': [0.1],
			'ERH': [0.2],
			'ERZ': [0.3],
			'ell_s1_km': [0.1],
			'ell_az1_deg': [90],
			'ell_dip1_deg': [0],
			'ell_s2_km': [0.1],
			'ell_az2_deg': [0],
			'ell_dip2_deg': [0],
			'ell_s3_km': [0.1],
			'ell_az3_deg': [0],
			'ell_dip3_deg': [90],
		}
	)


def test_run_synth_eval_calls_validate_elevation_correction_config(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	cfg_path = tmp_path / 'cfg.yaml'
	_write_config(
		cfg_path,
		dataset_dir=tmp_path / 'dataset',
		sim_yaml='sim.yaml',
		receiver_geometry='recv.npy',
		outputs_dir='out',
		template_cmd=tmp_path / 'template.cmd',
		hypoinverse_exe=tmp_path / 'hyp1',
		model_type='CRH',
		use_station_elev=False,
		apply_station_elevation_delay=False,
		z_is_depth_positive=True,
	)

	called = {'ok': False}

	def _validate(**_kwargs: object) -> None:
		called['ok'] = True
		raise RuntimeError('validate_called')

	monkeypatch.setattr(pl, 'validate_elevation_correction_config', _validate)

	with pytest.raises(RuntimeError, match='validate_called'):
		pl.run_synth_eval(cfg_path, runs_root=tmp_path / 'runs')

	assert called['ok'] is True


def test_run_synth_eval_passes_z_is_depth_positive_to_build_station_df(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	dataset_dir, sim_yaml_name, recv_name = _prepare_dataset(tmp_path)

	template_cmd = tmp_path / 'template.cmd'
	hyp_exe = tmp_path / 'hyp1'
	template_cmd.write_text("STA 'x'\nCRH 1 'x'\nCRH 2 'x'\n", encoding='utf-8')
	hyp_exe.write_text('', encoding='utf-8')

	cfg_path = tmp_path / 'cfg.yaml'
	_write_config(
		cfg_path,
		dataset_dir=dataset_dir,
		sim_yaml=sim_yaml_name,
		receiver_geometry=recv_name,
		outputs_dir='out',
		template_cmd=template_cmd,
		hypoinverse_exe=hyp_exe,
		model_type='CRH',
		use_station_elev=False,
		apply_station_elevation_delay=False,
		z_is_depth_positive=False,
	)

	monkeypatch.setattr(pl, 'validate_elevation_correction_config', lambda **_k: None)

	seen: dict[str, object] = {}

	def _build_station_df(
		_recv: np.ndarray, *_args: object, z_is_depth_positive: bool
	) -> pd.DataFrame:
		seen['z_is_depth_positive'] = bool(z_is_depth_positive)
		return pd.DataFrame(
			{
				'station_code': ['S0001'],
				'Latitude_deg': [35.0],
				'Longitude_deg': [140.0],
				'Elevation_m': [100],
			}
		)

	monkeypatch.setattr(pl, 'build_station_df', _build_station_df)
	monkeypatch.setattr(
		pl, 'build_truth_df', lambda *_a, **_k: pd.DataFrame({'id': [1]})
	)
	monkeypatch.setattr(
		pl, 'build_epic_df', lambda *_a, **_k: pd.DataFrame({'id': [1]})
	)
	monkeypatch.setattr(
		pl, 'build_meas_df', lambda *_a, **_k: pd.DataFrame({'id': [1]})
	)
	monkeypatch.setattr(pl, 'write_station_csv', lambda *_a, **_k: None)

	def _write_sta(
		_csv: Path, _out: Path, *, force_zero_pdelays: bool = False, **_k: object
	) -> None:
		seen['force_zero_pdelays'] = bool(force_zero_pdelays)

	monkeypatch.setattr(pl, 'write_hypoinverse_sta', _write_sta)
	monkeypatch.setattr(
		pl, 'extract_phase_records', lambda *_a, **_k: pd.DataFrame({'x': [1]})
	)
	monkeypatch.setattr(
		pl, 'override_phase_weight_by_station_prefix', lambda phases, **_k: phases
	)
	monkeypatch.setattr(pl, 'write_hypoinverse_arc_from_phases', lambda *_a, **_k: None)

	def _write_cmd(_tmpl: Path, out_cmd: Path) -> None:
		out_cmd.write_text('\n', encoding='utf-8', newline='\n')

	monkeypatch.setattr(pl, 'write_cmd_from_template', _write_cmd)

	write_crh_calls: list[str] = []

	def _write_crh(_path: Path, name: str, _layers: object) -> None:
		write_crh_calls.append(str(name))

	monkeypatch.setattr(pl, 'write_crh', _write_crh)

	def _run_hyp(_exe: Path, _cmd: Path, run_dir: Path) -> object:
		(run_dir / 'hypoinverse_run.prt').write_text('prt', encoding='utf-8')
		(run_dir / 'hypoinverse_run.sum').write_text('sum', encoding='utf-8')
		(run_dir / 'hypoinverse_run_out.arc').write_text('arc', encoding='utf-8')
		return object()

	monkeypatch.setattr(pl, 'run_hypoinverse', _run_hyp)
	monkeypatch.setattr(pl, 'evaluate', lambda *_a, **_k: _stub_eval_df())

	pl.run_synth_eval(cfg_path, runs_root=tmp_path / 'runs')

	assert seen['z_is_depth_positive'] is False
	assert seen['force_zero_pdelays'] is False
	assert write_crh_calls == ['SYNTH_P', 'SYNTH_S']


def test_run_synth_eval_passes_event_subsample_to_build_truth_df(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	dataset_dir, sim_yaml_name, recv_name = _prepare_dataset(tmp_path)

	template_cmd = tmp_path / 'template.cmd'
	hyp_exe = tmp_path / 'hyp1'
	template_cmd.write_text("STA 'x'\nCRH 1 'x'\nCRH 2 'x'\n", encoding='utf-8')
	hyp_exe.write_text('', encoding='utf-8')

	cfg_path = tmp_path / 'cfg.yaml'
	_write_config(
		cfg_path,
		dataset_dir=dataset_dir,
		sim_yaml=sim_yaml_name,
		receiver_geometry=recv_name,
		outputs_dir='out',
		template_cmd=template_cmd,
		hypoinverse_exe=hyp_exe,
		model_type='CRH',
		use_station_elev=False,
		apply_station_elevation_delay=False,
		z_is_depth_positive=True,
		event_subsample={'stride_ijk': [5, 5, 5]},
	)

	monkeypatch.setattr(pl, 'validate_elevation_correction_config', lambda **_k: None)
	monkeypatch.setattr(
		pl,
		'build_station_df',
		lambda *_a, **_k: pd.DataFrame(
			{
				'station_code': ['S0001'],
				'Latitude_deg': [35.0],
				'Longitude_deg': [140.0],
				'Elevation_m': [100],
			}
		),
	)

	seen: dict[str, object] = {}

	def _build_truth_df(*_a: object, **kwargs: object) -> pd.DataFrame:
		seen['event_stride_ijk'] = kwargs.get('event_stride_ijk')
		seen['event_keep_n_xyz'] = kwargs.get('event_keep_n_xyz')
		return pd.DataFrame({'id': [1]})

	monkeypatch.setattr(pl, 'build_truth_df', _build_truth_df)
	monkeypatch.setattr(
		pl, 'build_epic_df', lambda *_a, **_k: pd.DataFrame({'id': [1]})
	)
	monkeypatch.setattr(
		pl, 'build_meas_df', lambda *_a, **_k: pd.DataFrame({'id': [1]})
	)
	monkeypatch.setattr(pl, 'write_station_csv', lambda *_a, **_k: None)
	monkeypatch.setattr(pl, 'write_hypoinverse_sta', lambda *_a, **_k: None)
	monkeypatch.setattr(
		pl, 'extract_phase_records', lambda *_a, **_k: pd.DataFrame({'x': [1]})
	)
	monkeypatch.setattr(
		pl, 'override_phase_weight_by_station_prefix', lambda phases, **_k: phases
	)
	monkeypatch.setattr(pl, 'write_hypoinverse_arc_from_phases', lambda *_a, **_k: None)
	monkeypatch.setattr(
		pl,
		'write_cmd_from_template',
		lambda _tmpl, out_cmd: out_cmd.write_text('\n', encoding='utf-8', newline='\n'),
	)
	monkeypatch.setattr(pl, 'write_crh', lambda *_a, **_k: None)
	monkeypatch.setattr(
		pl,
		'run_hypoinverse',
		lambda _exe, _cmd, run_dir: (
			(run_dir / 'hypoinverse_run.prt').write_text('prt', encoding='utf-8'),
			(run_dir / 'hypoinverse_run.sum').write_text('sum', encoding='utf-8'),
			(run_dir / 'hypoinverse_run_out.arc').write_text('arc', encoding='utf-8'),
		),
	)
	monkeypatch.setattr(pl, 'evaluate', lambda *_a, **_k: _stub_eval_df())

	pl.run_synth_eval(cfg_path, runs_root=tmp_path / 'runs')

	assert seen['event_stride_ijk'] == [5, 5, 5]
	assert seen['event_keep_n_xyz'] is None


def test_run_synth_eval_passes_event_filter_to_build_truth_df(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	dataset_dir, sim_yaml_name, recv_name = _prepare_dataset(tmp_path)

	template_cmd = tmp_path / 'template.cmd'
	hyp_exe = tmp_path / 'hyp1'
	template_cmd.write_text("STA 'x'\nCRH 1 'x'\nCRH 2 'x'\n", encoding='utf-8')
	hyp_exe.write_text('', encoding='utf-8')

	cfg_path = tmp_path / 'cfg.yaml'
	_write_config(
		cfg_path,
		dataset_dir=dataset_dir,
		sim_yaml=sim_yaml_name,
		receiver_geometry=recv_name,
		outputs_dir='out',
		template_cmd=template_cmd,
		hypoinverse_exe=hyp_exe,
		model_type='CRH',
		use_station_elev=False,
		apply_station_elevation_delay=False,
		z_is_depth_positive=True,
		event_filter={'z_range_m': [100.0, None]},
	)

	monkeypatch.setattr(pl, 'validate_elevation_correction_config', lambda **_k: None)
	monkeypatch.setattr(
		pl,
		'build_station_df',
		lambda *_a, **_k: pd.DataFrame(
			{
				'station_code': ['S0001'],
				'Latitude_deg': [35.0],
				'Longitude_deg': [140.0],
				'Elevation_m': [100],
			}
		),
	)

	seen: dict[str, object] = {}

	def _build_truth_df(*_a: object, **kwargs: object) -> pd.DataFrame:
		seen['event_z_range_m'] = kwargs.get('event_z_range_m')
		return pd.DataFrame({'id': [1]})

	monkeypatch.setattr(pl, 'build_truth_df', _build_truth_df)
	monkeypatch.setattr(
		pl, 'build_epic_df', lambda *_a, **_k: pd.DataFrame({'id': [1]})
	)
	monkeypatch.setattr(
		pl, 'build_meas_df', lambda *_a, **_k: pd.DataFrame({'id': [1]})
	)
	monkeypatch.setattr(pl, 'write_station_csv', lambda *_a, **_k: None)
	monkeypatch.setattr(pl, 'write_hypoinverse_sta', lambda *_a, **_k: None)
	monkeypatch.setattr(
		pl, 'extract_phase_records', lambda *_a, **_k: pd.DataFrame({'x': [1]})
	)
	monkeypatch.setattr(
		pl, 'override_phase_weight_by_station_prefix', lambda phases, **_k: phases
	)
	monkeypatch.setattr(pl, 'write_hypoinverse_arc_from_phases', lambda *_a, **_k: None)
	monkeypatch.setattr(
		pl,
		'write_cmd_from_template',
		lambda _tmpl, out_cmd: out_cmd.write_text('\n', encoding='utf-8', newline='\n'),
	)
	monkeypatch.setattr(pl, 'write_crh', lambda *_a, **_k: None)
	monkeypatch.setattr(
		pl,
		'run_hypoinverse',
		lambda _exe, _cmd, run_dir: (
			(run_dir / 'hypoinverse_run.prt').write_text('prt', encoding='utf-8'),
			(run_dir / 'hypoinverse_run.sum').write_text('sum', encoding='utf-8'),
			(run_dir / 'hypoinverse_run_out.arc').write_text('arc', encoding='utf-8'),
		),
	)
	monkeypatch.setattr(pl, 'evaluate', lambda *_a, **_k: _stub_eval_df())

	pl.run_synth_eval(cfg_path, runs_root=tmp_path / 'runs')

	assert seen['event_z_range_m'] == [100.0, None]


@pytest.mark.parametrize(
	'use_station_elev,expected_force_zero',
	[
		(True, True),
		(False, False),
	],
)
def test_run_synth_eval_cre_branch_calls_expected_functions_and_force_zero_pdelays(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	use_station_elev: bool,
	expected_force_zero: bool,
) -> None:
	dataset_dir, sim_yaml_name, recv_name = _prepare_dataset(tmp_path)

	template_cmd = tmp_path / 'template.cmd'
	hyp_exe = tmp_path / 'hyp1'
	template_cmd.write_text("STA 'x'\nCRH 1 'x'\nCRH 2 'x'\n", encoding='utf-8')
	hyp_exe.write_text('', encoding='utf-8')

	cfg_path = tmp_path / 'cfg.yaml'
	_write_config(
		cfg_path,
		dataset_dir=dataset_dir,
		sim_yaml=sim_yaml_name,
		receiver_geometry=recv_name,
		outputs_dir='out',
		template_cmd=template_cmd,
		hypoinverse_exe=hyp_exe,
		model_type='CRE',
		use_station_elev=use_station_elev,
		apply_station_elevation_delay=False,
		z_is_depth_positive=True,
		cre_reference_margin_m=250.0,
		cre_typical_station_elevation_m=100.0,
		cre_n_layers=3,
	)

	monkeypatch.setattr(pl, 'validate_elevation_correction_config', lambda **_k: None)

	station_df = pd.DataFrame(
		{
			'station_code': ['S0001'],
			'Latitude_deg': [35.0],
			'Longitude_deg': [140.0],
			'Elevation_m': [100],
		}
	)

	seen: dict[str, object] = {}

	monkeypatch.setattr(pl, 'build_station_df', lambda *_a, **_k: station_df)
	monkeypatch.setattr(
		pl, 'build_truth_df', lambda *_a, **_k: pd.DataFrame({'id': [1]})
	)
	monkeypatch.setattr(
		pl, 'build_epic_df', lambda *_a, **_k: pd.DataFrame({'id': [1]})
	)
	monkeypatch.setattr(
		pl, 'build_meas_df', lambda *_a, **_k: pd.DataFrame({'id': [1]})
	)
	monkeypatch.setattr(pl, 'write_station_csv', lambda *_a, **_k: None)

	def _write_sta(
		_csv: Path, _out: Path, *, force_zero_pdelays: bool = False, **_k: object
	) -> None:
		seen['force_zero_pdelays'] = bool(force_zero_pdelays)

	monkeypatch.setattr(pl, 'write_hypoinverse_sta', _write_sta)
	monkeypatch.setattr(
		pl, 'extract_phase_records', lambda *_a, **_k: pd.DataFrame({'x': [1]})
	)
	monkeypatch.setattr(
		pl, 'override_phase_weight_by_station_prefix', lambda phases, **_k: phases
	)
	monkeypatch.setattr(pl, 'write_hypoinverse_arc_from_phases', lambda *_a, **_k: None)

	def _ref(df: pd.DataFrame, *, elevation_col: str, margin_m: float) -> float:
		assert df is station_df
		assert elevation_col == 'Elevation_m'
		seen['margin_m'] = float(margin_m)
		return 1.5

	def _typ(*, explicit_m: float | None) -> float:
		seen['typical_m'] = explicit_m
		return 0.1

	def _shift(ref: float, typical: float) -> float:
		seen['ref_km'] = float(ref)
		seen['typ_km'] = float(typical)
		return float(ref) - float(typical)

	def _meta(
		run_dir: Path, *, ref_elev_km: float, typical_elev_km: float, shift_km: float
	) -> None:
		seen['meta'] = (float(ref_elev_km), float(typical_elev_km), float(shift_km))

	def _write_cre_models(
		run_dir: Path,
		*,
		vp_kms: float,
		vs_kms: float,
		shift_km: float,
		n_layers: int,
	) -> tuple[Path, Path]:
		seen['cre_models'] = (
			float(vp_kms),
			float(vs_kms),
			float(shift_km),
			int(n_layers),
		)
		p = Path(run_dir) / 'P.cre'
		s = Path(run_dir) / 'S.cre'
		return p, s

	def _patch_cmd(
		_template: Path,
		out_cmd: Path,
		*,
		sta_file: str,
		p_model: str,
		s_model: str,
		ref_elev_km: float,
		use_station_elev: bool,
	) -> None:
		assert '/' not in sta_file and '\\' not in sta_file
		assert '/' not in p_model and '\\' not in p_model
		assert '/' not in s_model and '\\' not in s_model
		seen['patch_cmd'] = (
			sta_file,
			p_model,
			s_model,
			float(ref_elev_km),
			bool(use_station_elev),
		)
		out_cmd.write_text('\n', encoding='utf-8', newline='\n')

	monkeypatch.setattr(pl, 'compute_reference_elevation_km', _ref)
	monkeypatch.setattr(pl, 'compute_typical_station_elevation_km', _typ)
	monkeypatch.setattr(pl, 'compute_cre_layer_top_shift_km', _shift)
	monkeypatch.setattr(pl, 'write_cre_meta', _meta)
	monkeypatch.setattr(pl, 'write_synth_cre_models', _write_cre_models)
	monkeypatch.setattr(pl, 'patch_cmd_template_for_cre', _patch_cmd)

	def _never(*_a: object, **_k: object) -> None:
		raise AssertionError('unexpected call')

	monkeypatch.setattr(pl, 'write_cmd_from_template', _never)
	monkeypatch.setattr(pl, 'write_crh', _never)

	def _run_hyp(_exe: Path, _cmd: Path, run_dir: Path) -> object:
		(Path(run_dir) / 'hypoinverse_run.prt').write_text('prt', encoding='utf-8')
		(Path(run_dir) / 'hypoinverse_run.sum').write_text('sum', encoding='utf-8')
		(Path(run_dir) / 'hypoinverse_run_out.arc').write_text('arc', encoding='utf-8')
		return object()

	monkeypatch.setattr(pl, 'run_hypoinverse', _run_hyp)
	monkeypatch.setattr(pl, 'evaluate', lambda *_a, **_k: _stub_eval_df())

	run_dir, _df_eval, _stats = pl.run_synth_eval(cfg_path, runs_root=tmp_path / 'runs')

	assert seen['margin_m'] == 250.0
	assert seen['typical_m'] == 100.0
	assert seen['meta'] == (1.5, 0.1, 1.4)
	assert seen['cre_models'] == (6.0, 3.5, 1.4, 3)
	assert seen['force_zero_pdelays'] is expected_force_zero

	sta_file, p_model, s_model, ref_km, uflag = seen['patch_cmd']
	assert sta_file == 'stations_synth.sta'
	assert p_model == 'P.cre'
	assert s_model == 'S.cre'
	assert ref_km == 1.5
	assert uflag is use_station_elev
	assert Path(run_dir).is_dir()


def test_run_synth_eval_crh_branch_calls_expected_functions(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	dataset_dir, sim_yaml_name, recv_name = _prepare_dataset(tmp_path)

	template_cmd = tmp_path / 'template.cmd'
	hyp_exe = tmp_path / 'hyp1'
	template_cmd.write_text("STA 'x'\nCRH 1 'x'\nCRH 2 'x'\n", encoding='utf-8')
	hyp_exe.write_text('', encoding='utf-8')

	cfg_path = tmp_path / 'cfg.yaml'
	_write_config(
		cfg_path,
		dataset_dir=dataset_dir,
		sim_yaml=sim_yaml_name,
		receiver_geometry=recv_name,
		outputs_dir='out',
		template_cmd=template_cmd,
		hypoinverse_exe=hyp_exe,
		model_type='CRH',
		use_station_elev=False,
		apply_station_elevation_delay=False,
		z_is_depth_positive=True,
	)

	monkeypatch.setattr(pl, 'validate_elevation_correction_config', lambda **_k: None)

	seen: dict[str, object] = {}

	monkeypatch.setattr(
		pl,
		'build_station_df',
		lambda *_a, **_k: pd.DataFrame(
			{
				'station_code': ['S0001'],
				'Latitude_deg': [35.0],
				'Longitude_deg': [140.0],
				'Elevation_m': [0],
			}
		),
	)
	monkeypatch.setattr(
		pl, 'build_truth_df', lambda *_a, **_k: pd.DataFrame({'id': [1]})
	)
	monkeypatch.setattr(
		pl, 'build_epic_df', lambda *_a, **_k: pd.DataFrame({'id': [1]})
	)
	monkeypatch.setattr(
		pl, 'build_meas_df', lambda *_a, **_k: pd.DataFrame({'id': [1]})
	)
	monkeypatch.setattr(pl, 'write_station_csv', lambda *_a, **_k: None)

	def _write_sta(
		_csv: Path, _out: Path, *, force_zero_pdelays: bool = False, **_k: object
	) -> None:
		seen['force_zero_pdelays'] = bool(force_zero_pdelays)

	monkeypatch.setattr(pl, 'write_hypoinverse_sta', _write_sta)
	monkeypatch.setattr(
		pl, 'extract_phase_records', lambda *_a, **_k: pd.DataFrame({'x': [1]})
	)
	monkeypatch.setattr(
		pl, 'override_phase_weight_by_station_prefix', lambda phases, **_k: phases
	)
	monkeypatch.setattr(pl, 'write_hypoinverse_arc_from_phases', lambda *_a, **_k: None)

	write_crh_calls: list[str] = []

	def _write_crh(_path: Path, name: str, _layers: object) -> None:
		write_crh_calls.append(str(name))

	monkeypatch.setattr(pl, 'write_crh', _write_crh)

	def _write_cmd(_tmpl: Path, out_cmd: Path) -> None:
		seen['write_cmd'] = True
		out_cmd.write_text('\n', encoding='utf-8', newline='\n')

	monkeypatch.setattr(pl, 'write_cmd_from_template', _write_cmd)

	def _never(*_a: object, **_k: object) -> None:
		raise AssertionError('unexpected call')

	monkeypatch.setattr(pl, 'compute_reference_elevation_km', _never)
	monkeypatch.setattr(pl, 'compute_typical_station_elevation_km', _never)
	monkeypatch.setattr(pl, 'compute_cre_layer_top_shift_km', _never)
	monkeypatch.setattr(pl, 'write_cre_meta', _never)
	monkeypatch.setattr(pl, 'write_synth_cre_models', _never)
	monkeypatch.setattr(pl, 'patch_cmd_template_for_cre', _never)

	def _run_hyp(_exe: Path, _cmd: Path, run_dir: Path) -> object:
		(Path(run_dir) / 'hypoinverse_run.prt').write_text('prt', encoding='utf-8')
		(Path(run_dir) / 'hypoinverse_run.sum').write_text('sum', encoding='utf-8')
		(Path(run_dir) / 'hypoinverse_run_out.arc').write_text('arc', encoding='utf-8')
		return object()

	monkeypatch.setattr(pl, 'run_hypoinverse', _run_hyp)
	monkeypatch.setattr(pl, 'evaluate', lambda *_a, **_k: _stub_eval_df())

	pl.run_synth_eval(cfg_path, runs_root=tmp_path / 'runs')

	assert seen['force_zero_pdelays'] is False
	assert seen.get('write_cmd', False) is True
	assert write_crh_calls == ['SYNTH_P', 'SYNTH_S']
