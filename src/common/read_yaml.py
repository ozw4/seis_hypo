from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml


def read_yaml_preset(path: Path, preset: str) -> dict[str, Any]:
	if not path.is_file():
		raise FileNotFoundError(f'yaml not found: {path}')
	obj = yaml.safe_load(path.read_text(encoding='utf-8'))
	if not isinstance(obj, dict):
		raise ValueError(f'yaml root must be mapping: {path}')
	if preset not in obj:
		raise KeyError(f'preset not found: {preset} in {path}')
	val = obj[preset]
	if not isinstance(val, dict):
		raise ValueError(f'preset value must be mapping: preset={preset} path={path}')
	return val


def fieldnames(cls: type) -> set[str]:
	return {f.name for f in fields(cls)}
