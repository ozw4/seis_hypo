from __future__ import annotations

from pathlib import Path

import pytest

from hypo.synth_eval.hypoinverse_runner import patch_cmd_template_for_cre


def test_patch_cmd_template_for_cre_replaces_sta_and_models_and_inserts_sal(
	tmp_path: Path,
) -> None:
	template = tmp_path / 'template.cmd'
	out_cmd = tmp_path / 'out.cmd'

	template.write_text(
		'\n'.join(
			[
				'* comment line',
				'',
				"STA 'old.sta'",
				"CRH 1 'old_p.crh'",
				"CRT 2 'old_s.crh'",
				"CRH 3 'keep_this.crh'",
				'UNK something',
				"PHS 'old_phs.arc'",
				"PRT 'old.prt'",
				"SUM 'old.sum'",
				"ARC 'old.arc'",
			]
		)
		+ '\n',
		encoding='utf-8',
		newline='\n',
	)

	patch_cmd_template_for_cre(
		template,
		out_cmd,
		sta_file='new.sta',
		p_model='P.cre',
		s_model='S.cre',
		ref_elev_km=0.1234567,
		use_station_elev=True,
	)

	text = out_cmd.read_text(encoding='utf-8')
	assert text.endswith('\n')
	lines = text.splitlines()

	assert lines[0] == '* comment line'
	assert lines[1] == ''

	assert "STA 'new.sta'" in lines
	assert "CRE 1 'P.cre' 0.123457 T" in lines
	assert "CRE 2 'S.cre' 0.123457 T" in lines

	# unrelated model definition kept
	assert "CRH 3 'keep_this.crh'" in lines

	# fixed output/input names kept compatible with existing patcher
	assert "PHS 'hypoinverse_input.arc'" in lines
	assert "PRT 'hypoinverse_run.prt'" in lines
	assert "SUM 'hypoinverse_run.sum'" in lines
	assert "ARC 'hypoinverse_run_out.arc'" in lines

	# SAL inserted (because template has none)
	assert lines.count('SAL 1 2') == 1

	# SAL inserted right after the last replaced crust-model line (model 2)
	i_m2 = lines.index("CRE 2 'S.cre' 0.123457 T")
	assert lines[i_m2 + 1] == 'SAL 1 2'


def test_patch_cmd_template_for_cre_rewrites_existing_sal(tmp_path: Path) -> None:
	template = tmp_path / 'template.cmd'
	out_cmd = tmp_path / 'out.cmd'

	template.write_text(
		'\n'.join(
			[
				"STA 'old.sta'",
				"CRH 1 'old_p.crh'",
				"CRT 2 'old_s.crh'",
				'SAL 9 9',
			]
		)
		+ '\n',
		encoding='utf-8',
		newline='\n',
	)

	patch_cmd_template_for_cre(
		template,
		out_cmd,
		sta_file='new.sta',
		p_model='P.cre',
		s_model='S.cre',
		ref_elev_km=0.0,
		use_station_elev=False,
	)

	lines = out_cmd.read_text(encoding='utf-8').splitlines()
	assert "CRE 1 'P.cre' 0.000000 F" in lines
	assert "CRE 2 'S.cre' 0.000000 F" in lines
	assert lines.count('SAL 1 2') == 1


def test_patch_cmd_template_for_cre_missing_sta_raises(tmp_path: Path) -> None:
	template = tmp_path / 'template.cmd'
	out_cmd = tmp_path / 'out.cmd'

	template.write_text(
		'\n'.join(
			[
				"CRH 1 'old_p.crh'",
				"CRT 2 'old_s.crh'",
			]
		)
		+ '\n',
		encoding='utf-8',
		newline='\n',
	)

	with pytest.raises(ValueError):
		patch_cmd_template_for_cre(
			template,
			out_cmd,
			sta_file='new.sta',
			p_model='P.cre',
			s_model='S.cre',
			ref_elev_km=0.0,
			use_station_elev=True,
		)


def test_patch_cmd_template_for_cre_missing_model2_raises(tmp_path: Path) -> None:
	template = tmp_path / 'template.cmd'
	out_cmd = tmp_path / 'out.cmd'

	template.write_text(
		'\n'.join(
			[
				"STA 'old.sta'",
				"CRH 1 'old_p.crh'",
			]
		)
		+ '\n',
		encoding='utf-8',
		newline='\n',
	)

	with pytest.raises(ValueError):
		patch_cmd_template_for_cre(
			template,
			out_cmd,
			sta_file='new.sta',
			p_model='P.cre',
			s_model='S.cre',
			ref_elev_km=0.0,
			use_station_elev=True,
		)
