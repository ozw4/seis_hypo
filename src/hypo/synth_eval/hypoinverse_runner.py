from __future__ import annotations

import subprocess
from pathlib import Path


def patch_cmd_template(lines: list[str]) -> list[str]:
	out: list[str] = []
	for line in lines:
		s = line.strip().upper()
		if s.startswith('CRH 1'):
			out.append("CRH 1 'P.crh'")
		elif s.startswith('CRH 2'):
			out.append("CRH 2 'S.crh'")
		elif s.startswith('STA '):
			out.append("STA 'stations_synth.sta'")
		elif s.startswith('PRT '):
			out.append("PRT 'hypoinverse_run.prt'")
		elif s.startswith('SUM '):
			out.append("SUM 'hypoinverse_run.sum'")
		elif s.startswith('ARC '):
			out.append("ARC 'hypoinverse_run_out.arc'")
		elif s.startswith('PHS '):
			out.append("PHS 'hypoinverse_input.arc'")
		else:
			out.append(line)
	return out


def write_cmd_from_template(template_cmd: Path, out_cmd: Path) -> None:
	lines = template_cmd.read_text(encoding='utf-8').splitlines()
	patched = patch_cmd_template(lines)
	out_cmd.write_text('\n'.join(patched) + '\n', encoding='utf-8', newline='\n')


def run_hypoinverse(exe: Path, cmd: Path, run_dir: Path) -> None:
	with cmd.open('rb') as stdin:
		subprocess.run([str(exe)], stdin=stdin, cwd=run_dir, check=True)
