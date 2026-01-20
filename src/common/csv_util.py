from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def open_dict_writer(
	path: Path,
	*,
	fieldnames: list[str],
	mode: str = 'w',
	encoding: str = 'utf-8',
	newline: str = '',
	write_header: bool = True,
	extrasaction: str | None = None,
) -> tuple[object, csv.DictWriter]:
	path.parent.mkdir(parents=True, exist_ok=True)
	f = path.open(mode, newline=newline, encoding=encoding)
	kwargs: dict[str, Any] = {}
	if extrasaction is not None:
		kwargs['extrasaction'] = extrasaction
	writer = csv.DictWriter(f, fieldnames=fieldnames, **kwargs)
	if write_header:
		writer.writeheader()
	return f, writer
