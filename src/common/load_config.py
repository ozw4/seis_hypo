from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml

T = TypeVar('T')


def _ann_allows_path(ann: Any) -> bool:
	"""フィールド注釈が Path を含むか判定する。"""
	if ann is Path:
		return True
	origin = get_origin(ann)
	if origin is None:
		return False
	return any(a is Path for a in get_args(ann))


def _render_templates(
	params: dict[str, Any], *, max_passes: int | None = None
) -> dict[str, Any]:
	"""params内の str 値に対して {key} 形式のテンプレ展開を行う。
	- 参照先は同じparamsのキー（展開後の値を使う）
	- 未定義キー参照は KeyError で即失敗
	- 無限ループ防止のため多段展開は最大N回（N=キー数）まで
	- リテラルの { } は {{ }} としてエスケープ可能
	"""
	if max_passes is None:
		max_passes = max(len(params), 1)

	out = dict(params)
	for _ in range(max_passes):
		changed = False

		# format_mapに渡す値は文字列化（Path等が混ざっても安全にする）
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

	# ★追加: YAML内テンプレ {key} を展開
	params = _render_templates(params)

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
