from pathlib import Path
from typing import Any

import yaml


def load_plot_preset(
	yaml_path: str | Path,
	preset: str,
) -> dict[str, Any]:
	yaml_path = Path(yaml_path)
	if not yaml_path.is_file():
		raise FileNotFoundError(f'YAML が見つかりません: {yaml_path}')

	with yaml_path.open('r', encoding='utf-8') as f:
		cfg = yaml.safe_load(f)

	if not isinstance(cfg, dict):
		raise ValueError('YAML のトップレベルは mapping である必要があります')

	if preset not in cfg:
		raise KeyError(f'プリセット "{preset}" が YAML にありません')

	params = cfg[preset]
	if not isinstance(params, dict):
		raise ValueError(f'プリセット "{preset}" の値は mapping である必要があります')

	return params
