# file: src/common/run_snapshot.py
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import yaml


def _as_mapping(cfg_obj: Any) -> dict[str, Any]:
	# dataclass優先
	if is_dataclass(cfg_obj):
		return asdict(cfg_obj)
	# dictっぽいもの
	if isinstance(cfg_obj, dict):
		return dict(cfg_obj)
	# それ以外は属性辞書
	if hasattr(cfg_obj, '__dict__'):
		return dict(vars(cfg_obj))
	raise TypeError(f'cfg_obj is not serializable: {type(cfg_obj)}')


def _to_yaml_safe(x: Any) -> Any:
	# まず dataclass は dict に落とす
	if is_dataclass(x):
		return _to_yaml_safe(asdict(x))

	# Path は必ず str
	if isinstance(x, Path):
		return str(x)

	# dict / list / tuple を再帰
	if isinstance(x, dict):
		# YAMLキーは基本str想定（必要ならここでstr化）
		return {str(k): _to_yaml_safe(v) for k, v in x.items()}

	if isinstance(x, list):
		return [_to_yaml_safe(v) for v in x]

	if isinstance(x, tuple):
		return [_to_yaml_safe(v) for v in x]  # YAMLはlistの方が素直

	# numpy / pandas 系が紛れても落ちないように最低限だけ対処
	# .item() があるスカラーはPythonスカラーへ
	if hasattr(x, 'item') and callable(x.item):
		try:
			return x.item()
		except Exception:
			pass

	return x


def save_yaml_and_effective(
	*,
	out_dir: str | Path,
	yaml_path: str | Path,
	preset: str,
	cfg_obj: Any,
	label: str,
) -> None:
	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	yaml_path = Path(yaml_path)

	# 元yamlのコピー（存在しないなら即落とす）
	if not yaml_path.is_file():
		raise FileNotFoundError(f'yaml not found: {yaml_path}')
	(out_dir / f'config_{label}_source.yaml').write_text(
		yaml_path.read_text(encoding='utf-8'), encoding='utf-8'
	)

	effective = {
		'yaml': str(yaml_path),
		'preset': str(preset),
		'params': _to_yaml_safe(_as_mapping(cfg_obj)),
	}
	with (out_dir / f'config_{label}_effective.yaml').open('w', encoding='utf-8') as f:
		yaml.safe_dump(effective, f, sort_keys=False, allow_unicode=True)


def save_many_yaml_and_effective(
	*,
	out_dir: str | Path,
	items: list[tuple[str, str | Path, str, Any]],
) -> None:
	for label, yaml_path, preset, cfg_obj in items:
		save_yaml_and_effective(
			out_dir=out_dir,
			yaml_path=yaml_path,
			preset=preset,
			cfg_obj=cfg_obj,
			label=label,
		)
