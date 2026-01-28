from __future__ import annotations

import subprocess
from pathlib import Path

from hypo.cre import format_cre_cmd_line


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
		s0 = raw.strip()
		if not s0:
			out.append(raw)
			continue

		u = s0.upper()
		if u.startswith('*'):
			out.append(raw)
			continue

		if u.startswith('STA '):
			out.append(f"STA '{sta_file}'")
			found_sta = True
			continue

		# Keep the same output/input filenames as the existing patcher.
		if u.startswith('PRT '):
			out.append("PRT 'hypoinverse_run.prt'")
			continue
		if u.startswith('SUM '):
			out.append("SUM 'hypoinverse_run.sum'")
			continue
		if u.startswith('ARC '):
			out.append("ARC 'hypoinverse_run_out.arc'")
			continue
		if u.startswith('PHS '):
			out.append("PHS 'hypoinverse_input.arc'")
			continue

		if u.startswith('SAL '):
			out.append('SAL 1 2')
			found_sal = True
			continue

		if u.startswith('CRH ') or u.startswith('CRT ') or u.startswith('CRE '):
			parts = s0.split()
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
			sta_idx = next(
				(
					i
					for i, x in enumerate(out)
					if x.strip().upper().startswith('STA ')
				),
				None,
			)
			if sta_idx is None:
				raise ValueError('internal error: STA line not found after patching')
			out.insert(sta_idx, 'SAL 1 2')

	out_cmd.write_text('\n'.join(out) + '\n', encoding='utf-8', newline='\n')


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
