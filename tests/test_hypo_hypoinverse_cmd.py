from __future__ import annotations

from pathlib import Path

from hypo.hypoinverse_cmd import (
	cmd_token,
	force_err_erc,
	patch_cmd_template_paths,
)
from hypo.synth_eval.hypoinverse_runner import (
	patch_cmd_template,
	patch_cmd_template_for_cre,
)


def test_cmd_token_handles_blank_and_comment_lines() -> None:
	assert cmd_token('') is None
	assert cmd_token('   ') is None
	assert cmd_token('* comment') is None
	assert cmd_token('   * comment') is None


def test_cmd_token_uppercases_first_token() -> None:
	assert cmd_token('err 0.5') == 'ERR'
	assert cmd_token('  ERC   1') == 'ERC'
	assert cmd_token("STA 'x.sta'") == 'STA'


def test_force_err_erc_replaces_existing_err_and_erc() -> None:
	lines = [
		'ERR 0.5',
		'ERC 3',
		'LOC',
	]
	patched = force_err_erc(lines)

	loc = patched.index('LOC')
	assert patched[:loc].count('ERR 1.0') == 1
	assert patched[:loc].count('ERC 0') == 1
	assert 'ERR 0.5' not in patched
	assert 'ERC 3' not in patched


def test_force_err_erc_inserts_after_phs_and_before_loc_when_missing() -> None:
	lines = [
		"PHS 'x.arc'",
		'LOC',
	]
	patched = force_err_erc(lines)

	loc = patched.index('LOC')
	assert patched[0] == "PHS 'x.arc'"
	assert patched[1] == 'ERR 1.0'
	assert patched[2] == 'ERC 0'
	assert patched[:loc].count('ERR 1.0') == 1
	assert patched[:loc].count('ERC 0') == 1


def test_force_err_erc_dedupes_err_and_erc_in_effective_block() -> None:
	lines = [
		'ERR 0.2',
		"PHS 'x'",
		'ERR 0.3',
		'ERC 9',
		'ERC 8',
		'LOC',
	]
	patched = force_err_erc(lines)

	loc = patched.index('LOC')
	assert patched[:loc].count('ERR 1.0') == 1
	assert patched[:loc].count('ERC 0') == 1
	assert sum(1 for x in patched[:loc] if cmd_token(x) == 'ERR') == 1
	assert sum(1 for x in patched[:loc] if cmd_token(x) == 'ERC') == 1


def test_patch_cmd_template_patches_io_names_preserves_comments_and_forces_err_erc() -> (
	None
):
	lines = [
		'* comment line',
		'',
		"STA 'old.sta'",
		"CRH 1 'old_p.crh'",
		"CRH 2 'old_s.crh'",
		"CRH 3 'keep_this.crh'",
		"PHS 'old_phs.arc'",
		"PRT 'old.prt'",
		"SUM 'old.sum'",
		"ARC 'old.arc'",
		'LOC',
	]
	patched = patch_cmd_template(lines)

	assert patched[0] == '* comment line'
	assert patched[1] == ''

	assert "STA 'stations_synth.sta'" in patched
	assert "CRH 1 'P.crh'" in patched
	assert "CRH 2 'S.crh'" in patched
	assert "CRH 3 'keep_this.crh'" in patched
	assert "PHS 'hypoinverse_input.arc'" in patched
	assert "PRT 'hypoinverse_run.prt'" in patched
	assert "SUM 'hypoinverse_run.sum'" in patched
	assert "ARC 'hypoinverse_run_out.arc'" in patched

	loc = patched.index('LOC')
	assert patched[:loc].count('ERR 1.0') == 1
	assert patched[:loc].count('ERC 0') == 1
	assert patched.index('ERR 1.0') < loc
	assert patched.index('ERC 0') < loc

	phs_idx = patched.index("PHS 'hypoinverse_input.arc'")
	assert patched[phs_idx + 1] == 'ERR 1.0'
	assert patched[phs_idx + 2] == 'ERC 0'


def test_patch_cmd_template_for_cre_forces_err_erc_before_loc(tmp_path: Path) -> None:
	template = tmp_path / 'template.cmd'
	out_cmd = tmp_path / 'out.cmd'

	template.write_text(
		'\n'.join(
			[
				"STA 'old.sta'",
				"CRH 1 'old_p.crh'",
				"CRT 2 'old_s.crh'",
				'LOC',
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
	loc = lines.index('LOC')
	assert lines[:loc].count('ERR 1.0') == 1
	assert lines[:loc].count('ERC 0') == 1
	assert lines.index('ERR 1.0') < loc
	assert lines.index('ERC 0') < loc


def test_patch_cmd_template_paths_replaces_sta_and_preserves_model_tokens() -> None:
	lines = [
		'* keep comment',
		"STA 'old.sta'",
		"CRH 1 'old_p.crh'",
		"CRT 2 'old_s.crh'",
		"WET 1.0 0.5 0.3 0.2",
		'LOC',
	]

	patched = patch_cmd_template_paths(
		lines,
		sta_file='new.sta',
		pcrh_file='new_p.crh',
		scrh_file='new_s.crh',
	)

	assert patched[0] == '* keep comment'
	assert "STA 'new.sta'" in patched
	assert "CRH 1 'new_p.crh'" in patched
	assert "CRT 2 'new_s.crh'" in patched
	assert 'WET 1.0 0.5 0.3 0.2' in patched

	loc = patched.index('LOC')
	assert patched[:loc].count('ERR 1.0') == 1
	assert patched[:loc].count('ERC 0') == 1
