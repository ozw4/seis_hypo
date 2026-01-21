from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from common.json_io import read_json, write_json


def read_done_json(
	path: Path,
	*,
	on_missing: Literal['empty', 'raise'] = 'empty',
	on_error: Literal['empty', 'raise'] = 'raise',
) -> dict[str, Any]:
	p = Path(path)
	if not p.is_file():
		if on_missing == 'empty':
			return {}
		raise FileNotFoundError(p)
	try:
		return read_json(p, encoding='utf-8', errors='strict')
	except Exception:
		if on_error == 'empty':
			return {}
		raise


def write_done_json(path: Path, data: dict[str, Any]) -> None:
	p = Path(path)
	p.parent.mkdir(parents=True, exist_ok=True)
	write_json(p, data, ensure_ascii=False, indent=2)
	with p.open('a', encoding='utf-8') as f:
		f.write('\n')


def should_skip_done(
	data: dict[str, Any],
	*,
	run_tag: str,
	ok_statuses: set[str] | None,
	status_key: str = 'status',
	run_tag_key: str = 'run_tag',
) -> bool:
	if str(data.get(run_tag_key, '')) != str(run_tag):
		return False
	if ok_statuses is None:
		return True
	return str(data.get(status_key, '')) in ok_statuses
