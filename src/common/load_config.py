from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml

T = TypeVar('T')


def _ann_allows_path(ann: Any) -> bool:
	"""フィールド注釈が Path を含むか判定する。

	対応:
	- Path
	- Path | str
	- str | Path
	- Path | None
	- Path | str | None
	"""
	if ann is Path:
		return True

	origin = get_origin(ann)
	if origin is None:
		return False

	args = get_args(ann)
	return any(a is Path for a in args)


def load_config(
	cls: type[T],
	yaml_path: str | Path,
	preset: str,
) -> T:
	"""YAML の preset 定義から dataclass 設定を読み込む。

	動作:
	- YAML が存在することを確認
	- トップレベル mapping を要求
	- preset キーの存在を要求
	- preset 値も mapping を要求
	- dataclass の未知キーがあれば即エラー
	- Path を含む注釈のフィールドは str -> Path に補正
	  ※ from __future__ import annotations 対応のため
	    get_type_hints(cls) で評価済み注釈を使用する
	"""
	if not is_dataclass(cls):
		raise TypeError(f'{cls} is not a dataclass')

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

	fset = {f.name for f in fields(cls)}
	unknown = [k for k in params.keys() if k not in fset]
	if unknown:
		raise ValueError(f'プリセット "{preset}" に未知のキーがあります: {unknown}')

	# ★ ここが肝
	type_hints = get_type_hints(cls)

	kwargs: dict[str, Any] = {}
	for f in fields(cls):
		if f.name not in params:
			continue

		val = params[f.name]
		ann = type_hints.get(f.name, f.type)

		if _ann_allows_path(ann):
			kwargs[f.name] = val if isinstance(val, Path) else Path(str(val))
		else:
			kwargs[f.name] = val

	return cls(**kwargs)  # type: ignore[arg-type]
