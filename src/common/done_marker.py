from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal


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
		return json.loads(p.read_text(encoding='utf-8'))
	except Exception:
		if on_error == 'empty':
			return {}
		raise


def write_done_json(path: Path, data: dict[str, Any]) -> None:
	p = Path(path)
	p.parent.mkdir(parents=True, exist_ok=True)
	p.write_text(
		json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
	)


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
