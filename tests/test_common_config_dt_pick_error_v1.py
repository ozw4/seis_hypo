"""Minimal tests for dt_pick_error v1 config loader/expander in common.config."""

from __future__ import annotations

import pytest

from common.config import expand_dt_pick_error_experiments, load_dt_pick_error_config_v1


def _yaml_base(
	*,
	out_dir: str = 'runs/${run.run_id}',
	include_stalta: bool = True,
	experiments: str = '',
) -> str:
	lines: list[str] = [
		'version: 1',
		'run:',
		'  run_id: dt_stalta_p_u_20260203_01',
		f'  out_dir: "{out_dir}"',
		'  overwrite: false',
		'  notes: ""',
		'',
		'inputs:',
		'  event_root: /workspace/data/jma/events',
		'  epicenters_csv: /workspace/data/jma/epicenters.csv',
		'  measurements_csv: /workspace/data/jma/arrivetime_measurements.csv',
		'  mapping_report_csv: /workspace/data/jma/mapping_report.csv',
		'  near0_csv: /workspace/data/jma/near0.csv',
		'  monthly_presence_csv: /workspace/data/jma/monthly_presence.csv',
		'  mag1_types_allowed: ["v", "V"]',
		'  distance: hypocentral',
		'  phase_defs:',
		'    P: ["P", "EP", "IP"]',
		'    S: ["S", "ES", "IS"]',
		'  stations_allowlist: null',
		'  event_id_allowlist: null',
		'',
		'preprocess:',
		'  preprocess_preset: jma_snr_pick_table_v1',
		'  fs_target_hz: 100',
		'  detrend: linear',
		'  bandpass:',
		'    fstop_lo: 0.5',
		'    fpass_lo: 1.0',
		'    fpass_hi: 20.0',
		'    fstop_hi: 25.0',
		'    gpass: 1.0',
		'    gstop: 40.0',
		'',
		'picker:',
		'  picker_name: stalta',
		'  picker_preset: stalta_p_v1',
		'  phase: P',
		'  component: U',
	]
	if include_stalta:
		lines += [
			'  stalta:',
			'    transform: raw',
			'    sta_sec: 0.2',
			'    lta_sec: 2.0',
		]
	lines += [
		'',
		'pick_extract:',
		'  search_pre_sec: 1.0',
		'  search_post_sec: 3.0',
		'  clip_search_window: true',
		'  choose: max',
		'  tie_break: min_t',
		'  thr: 0.20',
		'  min_sep_sec: 0.20',
		'  search_i1_inclusive: true',
		'',
		'eval:',
		'  tol_sec: [0.05, 0.10, 0.20]',
		'  keep_missing_rows: true',
		'',
		'output:',
		'  dt_table_csv: dt_table.csv',
		'  skips_csv: skips.csv',
		'  save_config_snapshot: true',
	]
	if experiments:
		lines.append('')
		lines += experiments.strip('\n').splitlines()
		lines.append('')
	return '\n'.join(lines)


def test_load_dt_pick_error_config_v1_parses_minimal_v1(write_text) -> None:
	yaml_path = write_text('cfg.yaml', _yaml_base())
	cfg = load_dt_pick_error_config_v1(yaml_path)

	assert cfg.version == 1
	assert cfg.run.run_id == 'dt_stalta_p_u_20260203_01'
	assert str(cfg.run.out_dir) == 'runs/${run.run_id}'
	assert cfg.picker.picker_name == 'stalta'
	assert cfg.picker.stalta is not None
	assert cfg.experiments == []


def test_load_dt_pick_error_config_v1_rejects_unknown_top_level_key(write_text) -> None:
	yaml_path = write_text('cfg.yaml', _yaml_base() + '\nunknown_top: 1\n')
	with pytest.raises(ValueError, match='未知のトップレベルキーがあります'):
		load_dt_pick_error_config_v1(yaml_path)


def test_load_dt_pick_error_config_v1_requires_stalta_block_when_picker_is_stalta(
	write_text,
) -> None:
	yaml_path = write_text('cfg.yaml', _yaml_base(include_stalta=False))
	with pytest.raises(ValueError, match='picker\\.stalta is required'):
		load_dt_pick_error_config_v1(yaml_path)


def test_expand_dt_pick_error_experiments_appends_run_id_and_rerenders_out_dir(
	write_text,
) -> None:
	exps = """\
experiments:
  - name: thr_0p10
    pick_extract: {thr: 0.10}
    picker:
      stalta:
        sta_sec: 0.10
"""
	yaml_path = write_text('cfg.yaml', _yaml_base(experiments=exps))
	cfg = load_dt_pick_error_config_v1(yaml_path)

	all_cfgs = expand_dt_pick_error_experiments(cfg)
	assert [c.run.run_id for c in all_cfgs] == [
		'dt_stalta_p_u_20260203_01',
		'dt_stalta_p_u_20260203_01__thr_0p10',
	]
	assert str(all_cfgs[0].run.out_dir) == 'runs/dt_stalta_p_u_20260203_01'
	assert str(all_cfgs[1].run.out_dir) == 'runs/dt_stalta_p_u_20260203_01__thr_0p10'

	assert all_cfgs[1].pick_extract.thr == 0.10
	assert all_cfgs[1].picker.stalta is not None
	assert all_cfgs[1].picker.stalta.sta_sec == 0.10
	assert all_cfgs[1].picker.stalta.lta_sec == 2.0
	assert all_cfgs[1].picker.stalta.transform == 'raw'


@pytest.mark.parametrize(
	'yaml_text, stage, msg',
	[
		(
			'version: 1\nrun: {run_id: x, out_dir: runs/x}\n',
			'load',
			'inputs が必要です',
		),
		(
			_yaml_base().replace('  overwrite: false', '  overwrite: "false"'),
			'load',
			'run\\.overwrite must be bool',
		),
		(
			_yaml_base().replace('  distance: hypocentral', '  distance: epicentral'),
			'load',
			'inputs\\.distance must be "hypocentral"',
		),
		(
			_yaml_base().replace('    transform: raw', '    transform: bad'),
			'load',
			'picker\\.stalta\\.transform must be "raw" or "envelope"',
		),
		(
			_yaml_base(out_dir='runs/${run.no_such_key}'),
			'expand',
			'template key',
		),
	],
)
def test_dt_pick_error_config_v1_rejects_invalid_values(
	write_text, yaml_text: str, stage: str, msg: str
) -> None:
	yaml_path = write_text('cfg.yaml', yaml_text)

	if stage == 'load':
		with pytest.raises(ValueError, match=msg):
			load_dt_pick_error_config_v1(yaml_path)
		return

	cfg = load_dt_pick_error_config_v1(yaml_path)
	with pytest.raises(ValueError, match=msg):
		expand_dt_pick_error_experiments(cfg)
