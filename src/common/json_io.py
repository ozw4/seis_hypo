from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: Path, *, encoding: str = 'utf-8', errors: str = 'strict') -> Any:
	p = Path(path)
	text = p.read_text(encoding=encoding, errors=errors)
	return json.loads(text)


def sort_json_obj(obj: object) -> object:
	if isinstance(obj, dict):
		return {k: sort_json_obj(obj[k]) for k in sorted(obj, key=str)}
	if isinstance(obj, list):
		return [sort_json_obj(v) for v in obj]
	return obj


def write_json(
	path: Path,
	obj: Any,
	*,
	indent: int = 2,
	ensure_ascii: bool = False,
	sort_recursive: bool = False,
) -> None:
	p = Path(path)
	if sort_recursive:
		obj = sort_json_obj(obj)
	text = json.dumps(obj, ensure_ascii=ensure_ascii, indent=indent)
	p.write_text(text, encoding='utf-8')
