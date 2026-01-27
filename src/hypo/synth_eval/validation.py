from __future__ import annotations

from pathlib import Path


def require_abs(p: Path, key: str) -> None:
	if not p.is_absolute():
		raise ValueError(f'{key} must be an absolute path: {p}')


def require_filename_only(name: str, key: str) -> None:
	if '/' in name or '\\' in name:
		raise ValueError(f'{key} must be filename only (no directory): {name}')


def require_dirname_only(name: str, key: str) -> None:
	if '/' in name or '\\' in name:
		raise ValueError(
			f'{key} must be directory name only (no path separators): {name}'
		)
	if name.strip() == '':
		raise ValueError(f'{key} must be non-empty')
