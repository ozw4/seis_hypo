from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# --- src/common/core.py に追加（write_event_json の下あたり） ---


def validate_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
	missing = [c for c in required if c not in df.columns]
	if missing:
		raise ValueError(f'{label} に必要な列がありません: {missing}')


def slice_with_pad(
	x: np.ndarray,
	start_idx: int,
	end_idx: int,
) -> np.ndarray:
	"""x: (..., nt_total)。最後の軸について [start_idx:end_idx) を0パディングで安全に抜く。"""
	nt = x.shape[-1]
	length = max(0, end_idx - start_idx)
	out_shape = x.shape[:-1] + (length,)
	y = np.zeros(out_shape, dtype=x.dtype)
	s0 = max(0, start_idx)
	e0 = min(nt, end_idx)
	if e0 > s0:
		d0 = max(0, -start_idx)
		y[..., d0 : d0 + (e0 - s0)] = x[..., s0:e0]
	return y


def write_event_json(
	*,
	event_dir: Path,
	event_id: int,
	origin_time_jst: dt.datetime,
	pre_sec: int,
	post_sec: int,
	network_code: str,
	span_min: int,
	threads: int,
	stations: list[str],
	cnt_files: list[str],
	ch_file: str,
	extra: dict[str, Any] | None = None,
) -> None:
	obj: dict[str, Any] = {
		'event_id': event_id,
		'origin_time_jst': origin_time_jst.isoformat(),
		'window': {'pre_sec': int(pre_sec), 'post_sec': int(post_sec)},
		'win32': {
			'network_code': str(network_code),
			'span_min': int(span_min),
			'threads': int(threads),
			'stations': list(stations),
			'cnt_files': list(cnt_files),
			'ch_file': str(ch_file),
		},
	}
	if extra:
		obj['extra'] = extra

	out = event_dir / 'event.json'
	with out.open('w', encoding='utf-8') as f:
		json.dump(obj, f, ensure_ascii=False, indent=2)


def load_event_json(event_dir: Path) -> dict[str, Any]:
	p = event_dir / 'event.json'
	if not p.is_file():
		raise FileNotFoundError(f'event.json not found: {p}')
	with p.open('r', encoding='utf-8') as f:
		return json.load(f)


def write_event_json_win32_groups(
	*,
	event_dir: Path,
	event_id: int,
	origin_time_jst: dt.datetime,
	pre_sec: int,
	post_sec: int,
	span_min: int,
	threads: int,
	win32_groups: list[dict[str, Any]],
	extra: dict[str, Any] | None = None,
) -> None:
	"""event.json を "groups" 形式で書く。

	win32_groups の各要素（最低限）:
	- network_code: str
	- stations: list[str]
	- cnt_files: list[str]
	- ch_file: str
	- select_used: bool
	"""
	if not win32_groups:
		raise ValueError('win32_groups is empty')

	obj: dict[str, Any] = {
		'event_id': int(event_id),
		'origin_time_jst': origin_time_jst.isoformat(),
		'window': {'pre_sec': int(pre_sec), 'post_sec': int(post_sec)},
		'win32': {
			'format': 'groups',
			'span_min': int(span_min),
			'threads': int(threads),
			'groups': win32_groups,
		},
	}
	if extra:
		obj['extra'] = extra

	out = event_dir / 'event.json'
	with out.open('w', encoding='utf-8') as f:
		json.dump(obj, f, ensure_ascii=False, indent=2)


def as_int_rate(fs: float, name: str) -> int:
	if float(fs) <= 0.0:
		raise ValueError(f'{name} must be > 0, got {fs}')
	i = int(round(float(fs)))
	if abs(float(fs) - float(i)) > 1e-6:
		raise ValueError(f'{name} must be integer-like, got {fs}')
	return int(i)
