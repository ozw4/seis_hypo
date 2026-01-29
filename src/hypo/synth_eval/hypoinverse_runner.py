from __future__ import annotations

import subprocess
from pathlib import Path

from hypo.cre import format_cre_cmd_line
from hypo.hypoinverse_cmd import cmd_token, force_err_erc


def _parse_cmd_model_number(token: str) -> int:
	t = str(token).strip()
	if not t:
		raise ValueError('empty model number token')

	sign = ''
	if t[0] in ('+', '-'):  # allow explicit signs for completeness
		sign = t[0]
		t = t[1:]

	if not t.isdigit():
		raise ValueError(f'invalid model number token: {token!r}')
	return int(sign + t) if sign else int(t)


def patch_cmd_template_for_cre(
	template_cmd: Path,
	out_cmd: Path,
	*,
	sta_file: str,
	p_model: str,
	s_model: str,
	ref_elev_km: float,
	use_station_elev: bool,
) -> None:
	lines = template_cmd.read_text(encoding='utf-8').splitlines()

	out: list[str] = []
	found_sta = False
	found_m1 = False
	found_m2 = False
	found_sal = False
	last_model_idx: int | None = None

	for line in lines:
		raw = line.rstrip('\n')
		tok = cmd_token(raw)
		if tok is None:
			out.append(raw)
			continue

		s0 = raw.strip()
		parts = s0.split()

		if tok == 'STA':
			out.append(f"STA '{sta_file}'")
			found_sta = True
			continue

		# Keep the same output/input filenames as the existing patcher.
		if tok == 'PRT':
			out.append("PRT 'hypoinverse_run.prt'")
			continue
		if tok == 'SUM':
			out.append("SUM 'hypoinverse_run.sum'")
			continue
		if tok == 'ARC':
			out.append("ARC 'hypoinverse_run_out.arc'")
			continue
		if tok == 'PHS':
			out.append("PHS 'hypoinverse_input.arc'")
			continue

		if tok == 'SAL':
			out.append('SAL 1 2')
			found_sal = True
			continue

		if tok in ('CRH', 'CRT', 'CRE'):
			if len(parts) < 2:
				raise ValueError(f'invalid crust-model line: {raw!r}')

			mod = _parse_cmd_model_number(parts[1])
			if mod == 1:
				out.append(
					format_cre_cmd_line(
						1,
						str(p_model),
						float(ref_elev_km),
						use_station_elev=bool(use_station_elev),
					)
				)
				found_m1 = True
				last_model_idx = len(out) - 1
				continue
			if mod == 2:
				out.append(
					format_cre_cmd_line(
						2,
						str(s_model),
						float(ref_elev_km),
						use_station_elev=bool(use_station_elev),
					)
				)
				found_m2 = True
				last_model_idx = len(out) - 1
				continue

			# Keep unrelated model definitions untouched.
			out.append(raw)
			continue

		out.append(raw)

	if not found_sta:
		raise ValueError('template cmd is missing a STA line to replace')
	if not found_m1:
		raise ValueError('template cmd is missing a CRH/CRT/CRE 1 line to replace')
	if not found_m2:
		raise ValueError('template cmd is missing a CRH/CRT/CRE 2 line to replace')

	if not found_sal:
		# Insert after the last crust-model line (preferred), otherwise before STA.
		if last_model_idx is not None:
			out.insert(last_model_idx + 1, 'SAL 1 2')
		else:
			sta_idx = next((i for i, x in enumerate(out) if cmd_token(x) == 'STA'), None)
			if sta_idx is None:
				raise ValueError('internal error: STA line not found after patching')
			out.insert(sta_idx, 'SAL 1 2')

	out = force_err_erc(out)
	out_cmd.write_text('\n'.join(out) + '\n', encoding='utf-8', newline='\n')


def patch_cmd_template(lines: list[str]) -> list[str]:
	out: list[str] = []
	for line in lines:
		tok = cmd_token(line)
		if tok is None:
			out.append(line)
			continue

		s0 = line.strip()
		parts = s0.split()

		if tok == 'CRH' and len(parts) >= 2:
			mod = _parse_cmd_model_number(parts[1])
			if mod == 1:
				out.append("CRH 1 'P.crh'")
				continue
			if mod == 2:
				out.append("CRH 2 'S.crh'")
				continue

		if tok == 'STA':
			out.append("STA 'stations_synth.sta'")
			continue
		if tok == 'PRT':
			out.append("PRT 'hypoinverse_run.prt'")
			continue
		if tok == 'SUM':
			out.append("SUM 'hypoinverse_run.sum'")
			continue
		if tok == 'ARC':
			out.append("ARC 'hypoinverse_run_out.arc'")
			continue
		if tok == 'PHS':
			out.append("PHS 'hypoinverse_input.arc'")
			continue

		out.append(line)

	return force_err_erc(out)


def write_cmd_from_template(template_cmd: Path, out_cmd: Path) -> None:
	lines = template_cmd.read_text(encoding='utf-8').splitlines()
	patched = patch_cmd_template(lines)
	out_cmd.write_text('\n'.join(patched) + '\n', encoding='utf-8', newline='\n')


def run_hypoinverse(
	exe_path: Path,
	cmd_path: Path,
	run_dir: Path,
) -> subprocess.CompletedProcess:
	with cmd_path.open('rb') as stdin:
		return subprocess.run(
			[str(exe_path)],
			stdin=stdin,
			cwd=run_dir,
			check=True,
		)
