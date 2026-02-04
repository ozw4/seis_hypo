"""Common YAML config helpers (read mapping/preset and render templates)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


def read_yaml_mapping(path: Path) -> dict[str, Any]:
	"""Read YAML and require the root object to be a mapping."""
	if not path.is_file():
		raise FileNotFoundError(f'yaml not found: {path}')
	obj = yaml.safe_load(path.read_text(encoding='utf-8'))
	if not isinstance(obj, dict):
		raise ValueError(f'yaml root must be mapping: {path}')
	return obj


def read_yaml_preset_mapping(path: Path, preset: str) -> dict[str, Any]:
	"""Read YAML root mapping and return a preset mapping as a shallow copy."""
	obj = read_yaml_mapping(path)
	if preset not in obj:
		raise KeyError(f'preset not found: {preset} in {path}')
	val = obj[preset]
	if not isinstance(val, dict):
		raise ValueError(f'preset value must be mapping: preset={preset} path={path}')
	return dict(val)


def render_brace_templates(
	params: dict[str, Any], *, max_passes: int | None = None
) -> dict[str, Any]:
	"""Render `{key}` templates within string values in `params`.

	Rules (compatible with common.load_config._render_templates):
	- References are keys from the same mapping (rendered values are used).
	- Undefined key reference fails immediately with KeyError.
	- Multi-pass rendering is limited to N passes (default: len(params), min 1).
	- Literal braces can be escaped with `{{` / `}}` (format_map behavior).
	"""
	if max_passes is None:
		max_passes = max(len(params), 1)

	out = dict(params)
	for _ in range(max_passes):
		changed = False
		mapping = {k: (str(v) if v is not None else '') for k, v in out.items()}

		for k, v in list(out.items()):
			if not isinstance(v, str):
				continue
			if '{' not in v:
				continue
			try:
				v2 = v.format_map(mapping)
			except KeyError as e:
				raise KeyError(
					f"template key {e!s} is not defined (in field '{k}')"
				) from e

			if v2 != v:
				out[k] = v2
				changed = True

		if not changed:
			break

	return out


_DOLLAR_TEMPLATE_RE = re.compile(r'\$\{([^}]+)\}')


def render_dollar_template(s: str, root: dict[str, Any]) -> str:
	"""Render `${a.b}` templates by looking up values from a root mapping.

	Rules (compatible with common.config._render_template_vars):
	- Keys are dot-separated paths in the root mapping.
	- Undefined key/path raises ValueError whose message contains "template key".
	- Single-pass substitution (no recursive/multi-pass expansion).
	"""

	def _lookup(path: str) -> str:
		parts = [p for p in path.split('.') if p]
		cur: Any = root
		for p in parts:
			if not isinstance(cur, dict) or p not in cur:
				raise ValueError(f'template key {path!r} is not defined')
			cur = cur[p]
		return str(cur)

	def _replace(m: re.Match[str]) -> str:
		key = m.group(1).strip()
		return _lookup(key)

	return _DOLLAR_TEMPLATE_RE.sub(_replace, s)
