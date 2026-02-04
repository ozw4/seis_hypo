from dataclasses import fields
from pathlib import Path
from typing import Any

from common.yaml_config import read_yaml_preset_mapping


def read_yaml_preset(path: Path, preset: str) -> dict[str, Any]:
	return read_yaml_preset_mapping(path, preset)


def fieldnames(cls: type) -> set[str]:
	return {f.name for f in fields(cls)}
