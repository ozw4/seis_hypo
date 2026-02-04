from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

from common.yaml_config import read_yaml_preset_mapping, render_brace_templates

T = TypeVar('T')


def _ann_allows_path(ann: Any) -> bool:
	"""フィールド注釈が Path を含むか判定する。"""
	if ann is Path:
		return True
	origin = get_origin(ann)
	if origin is None:
		return False
	return any(a is Path for a in get_args(ann))


def load_config(
	cls: type[T],
	yaml_path: str | Path,
	preset: str,
) -> T:
	"""YAML の preset 定義から dataclass 設定を読み込む（テンプレ展開対応）"""
	if not is_dataclass(cls):
		raise TypeError(f'{cls} is not a dataclass')

	yaml_path = Path(yaml_path)
	if not yaml_path.is_file():
		raise FileNotFoundError(f'YAML が見つかりません: {yaml_path}')

	params = read_yaml_preset_mapping(yaml_path, preset)

	fset = {f.name for f in fields(cls)}
	unknown = [k for k in params.keys() if k not in fset]
	if unknown:
		raise ValueError(f'プリセット "{preset}" に未知のキーがあります: {unknown}')

	# ★追加: YAML内テンプレ {key} を展開
	params = render_brace_templates(params)

	type_hints = get_type_hints(cls)

	kwargs: dict[str, Any] = {}
	for f in fields(cls):
		if f.name not in params:
			continue

		val = params[f.name]
		ann = type_hints.get(f.name, f.type)

		if val is None:
			kwargs[f.name] = None
			continue

		if _ann_allows_path(ann):
			kwargs[f.name] = val if isinstance(val, Path) else Path(str(val))
		else:
			kwargs[f.name] = val

	return cls(**kwargs)  # type: ignore[arg-type]
