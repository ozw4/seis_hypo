from __future__ import annotations

from pathlib import Path


def cmd_token(line: str) -> str | None:
	s = str(line).strip()
	if not s or s.startswith('*'):
		return None
	return s.split()[0].upper()


def _parse_cmd_model_number(token: str) -> int:
	t = str(token).strip()
	if not t:
		raise ValueError('empty model number token')

	sign = ''
	if t[0] in ('+', '-'):
		sign = t[0]
		t = t[1:]

	if not t.isdigit():
		raise ValueError(f'invalid model number token: {token!r}')
	return int(sign + t) if sign else int(t)


def _is_sal_12(line: str) -> bool:
	s = str(line).strip()
	if not s or s.startswith('*'):
		return False
	parts = s.split()
	if not parts or parts[0].upper() != 'SAL':
		return False
	return len(parts) >= 3 and parts[1] == '1' and parts[2] == '2'


def _first_loc_pos(lines: list[str]) -> int:
	for i, line in enumerate(lines):
		if cmd_token(line) == 'LOC':
			return i
	return len(lines)


def _compute_err_erc_insert_idx(lines: list[str], loc_pos: int) -> int:
	last_phs: int | None = None
	last_sal: int | None = None
	first_fil: int | None = None

	for i in range(loc_pos):
		tok = cmd_token(lines[i])
		if tok == 'PHS':
			last_phs = i
		if _is_sal_12(lines[i]):
			last_sal = i
		if tok == 'FIL' and first_fil is None:
			first_fil = i

	if last_phs is not None:
		return last_phs + 1
	if last_sal is not None:
		return last_sal + 1
	if first_fil is not None:
		return first_fil
	return loc_pos


def _has_cmd_before(lines: list[str], cmd: str, end: int) -> bool:
	for i in range(end):
		if cmd_token(lines[i]) == cmd:
			return True
	return False


def force_err_erc(lines: list[str]) -> list[str]:
	out: list[str] = []
	for line in lines:
		tok = cmd_token(line)
		if tok == 'ERR':
			out.append('ERR 1.0')
			continue
		if tok == 'ERC':
			out.append('ERC 0')
			continue
		out.append(line)

	loc_pos = _first_loc_pos(out)
	insert_idx = _compute_err_erc_insert_idx(out, loc_pos)

	missing_err = not _has_cmd_before(out, 'ERR', loc_pos)
	missing_erc = not _has_cmd_before(out, 'ERC', loc_pos)

	to_insert: list[str] = []
	if missing_err:
		to_insert.append('ERR 1.0')
	if missing_erc:
		to_insert.append('ERC 0')

	if to_insert:
		for j, x in enumerate(to_insert):
			out.insert(insert_idx + j, x)

	loc_pos = _first_loc_pos(out)

	def _dedupe_in_effective_block(cmd: str, expected: str) -> int:
		keep_idx: int | None = None
		for i in range(loc_pos):
			if cmd_token(out[i]) == cmd:
				keep_idx = i
				break

		if keep_idx is None:
			if loc_pos != len(out):
				raise ValueError(f'{expected} must appear before LOC')
			raise ValueError(f'{expected} is missing from cmd')

		out[keep_idx] = expected

		for i in range(loc_pos - 1, -1, -1):
			if i == keep_idx:
				continue
			if cmd_token(out[i]) == cmd:
				out.pop(i)
				if i < keep_idx:
					keep_idx -= 1

		return keep_idx

	err_idx = _dedupe_in_effective_block('ERR', 'ERR 1.0')
	erc_idx = _dedupe_in_effective_block('ERC', 'ERC 0')

	loc_pos = _first_loc_pos(out)
	if loc_pos != len(out):
		if err_idx >= loc_pos:
			raise ValueError('ERR 1.0 must appear before LOC')
		if erc_idx >= loc_pos:
			raise ValueError('ERC 0 must appear before LOC')

	return out


def patch_cmd_template_paths(
	lines: list[str],
	*,
	sta_file: str,
	pcrh_file: str,
	scrh_file: str,
) -> list[str]:
	out: list[str] = []
	found_sta = False
	found_p_model = False
	found_s_model = False

	for line in lines:
		tok = cmd_token(line)
		if tok is None:
			out.append(line)
			continue

		parts = line.strip().split()

		if tok == 'STA':
			out.append(f"STA '{sta_file}'")
			found_sta = True
			continue

		if tok in ('CRH', 'CRT', 'CRE'):
			if len(parts) < 2:
				raise ValueError(f'invalid crust-model line: {line!r}')

			model_number = _parse_cmd_model_number(parts[1])
			if model_number == 1:
				out.append(f"{tok} 1 '{pcrh_file}'")
				found_p_model = True
				continue
			if model_number == 2:
				out.append(f"{tok} 2 '{scrh_file}'")
				found_s_model = True
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

	if not found_sta:
		raise ValueError('template cmd is missing a STA line to replace')
	if not found_p_model:
		raise ValueError('template cmd is missing a CRH/CRT/CRE 1 line to replace')
	if not found_s_model:
		raise ValueError('template cmd is missing a CRH/CRT/CRE 2 line to replace')

	return force_err_erc(out)


def write_cmd_template_paths(
	template_cmd: Path,
	out_cmd: Path,
	*,
	sta_file: str,
	pcrh_file: str,
	scrh_file: str,
) -> None:
	lines = template_cmd.read_text(encoding='utf-8').splitlines()
	patched = patch_cmd_template_paths(
		lines,
		sta_file=sta_file,
		pcrh_file=pcrh_file,
		scrh_file=scrh_file,
	)
	out_cmd.write_text('\n'.join(patched) + '\n', encoding='utf-8', newline='\n')
