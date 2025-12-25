# %%
from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

# ========== 設定（ここだけ直せばOK） ==========

# 1) mea（arrivetime_measurements）
MEA_FILE = '/workspace/data/arrivetime/JMA/arrivetime_measurements.csv'
MEA_COL = 'station_code'

# 2) sta（あなたが言ってる sta_set 側：例として monthly_presence と同じファイルにしてある）
STA_FILE = '/workspace/data/station/jma/station.csv'
STA_COL = 'station_code'

# 3) ch（.ch側：これも例として monthly_presence と同じにしてある。別なら差し替え）
CH_FILE = '/workspace/proc/prepare_data/jma/snapshots/monthly/monthly_presence.csv'
CH_COL = 'station'

# 出力フォルダ
OUT_DIR = './station_code_venn3'

# 正規化（揺れがあるなら True 推奨）
STRIP = True
UPPER = False

# 表示件数（長いリストは先頭だけ表示）
MAX_SHOW = 50

# =============================================


@dataclass(frozen=True)
class Venn3:
	all_abc: list[Any]
	only_a: list[Any]
	only_b: list[Any]
	only_c: list[Any]
	only_ab: list[Any]
	only_ac: list[Any]
	only_bc: list[Any]


def _read_table(path: str) -> pd.DataFrame:
	p = Path(path)
	ext = p.suffix.lower()
	if ext == '.csv':
		return pd.read_csv(path)
	if ext in ('.xlsx', '.xls'):
		return pd.read_excel(path)
	raise ValueError(f'未対応の拡張子: {ext}（.csv/.xlsx/.xls のみ）')


def _to_code_set(df: pd.DataFrame, col: str, *, strip: bool, upper: bool) -> set[str]:
	if col not in df.columns:
		raise KeyError(f'列が見つかりません: {col} / columns={list(df.columns)}')

	s = df[col].dropna().astype(str)
	if strip:
		s = s.str.strip()
	if upper:
		s = s.str.upper()

	codes = set(s.tolist())
	codes.discard('')
	return codes


def venn3(a: Iterable[Any], b: Iterable[Any], c: Iterable[Any]) -> Venn3:
	A, B, C = set(a), set(b), set(c)
	all_abc = A & B & C
	only_ab = (A & B) - C
	only_ac = (A & C) - B
	only_bc = (B & C) - A
	only_a = A - B - C
	only_b = B - A - C
	only_c = C - A - B

	return Venn3(
		all_abc=sorted(all_abc),
		only_a=sorted(only_a),
		only_b=sorted(only_b),
		only_c=sorted(only_c),
		only_ab=sorted(only_ab),
		only_ac=sorted(only_ac),
		only_bc=sorted(only_bc),
	)


def _print_list(title: str, items: list[Any], max_show: int) -> None:
	print(f'\n[{title}] ({len(items)})')
	if len(items) <= max_show:
		print(items)
		return
	print(items[:max_show])
	print(f'(and {len(items) - max_show} more)')


def _write_list(out_dir: str, name: str, items: list[Any]) -> None:
	Path(out_dir).mkdir(parents=True, exist_ok=True)
	out_path = Path(out_dir) / f'{name}.csv'
	pd.Series(items, name='station_code').to_csv(out_path, index=False)


# def main() -> None:
mea_df = _read_table(MEA_FILE)
sta_df = _read_table(STA_FILE)
ch_df = _read_table(CH_FILE)

mea_set = _to_code_set(mea_df, MEA_COL, strip=STRIP, upper=UPPER)
sta_set = _to_code_set(sta_df, STA_COL, strip=STRIP, upper=UPPER)
ch_set = _to_code_set(ch_df, CH_COL, strip=STRIP, upper=UPPER)

res = venn3(mea_set, sta_set, ch_set)

# カウント概要
print('=== counts ===')
print(f'all(mea&sta&ch): {len(res.all_abc)}')
print(f'only_mea: {len(res.only_a)}')
print(f'only_sta: {len(res.only_b)}')
print(f'only_ch: {len(res.only_c)}')
print(f'only(mea&sta): {len(res.only_ab)}')
print(f'only(mea&ch): {len(res.only_ac)}')
print(f'only(sta&ch): {len(res.only_bc)}')

# 中身表示（先頭 MAX_SHOW 件まで）
_print_list('all(mea&sta&ch)', res.all_abc, MAX_SHOW)
_print_list('only_mea', res.only_a, MAX_SHOW)
_print_list('only_sta', res.only_b, MAX_SHOW)
_print_list('only_ch', res.only_c, MAX_SHOW)
_print_list('only(mea&sta)', res.only_ab, MAX_SHOW)
_print_list('only(mea&ch)', res.only_ac, MAX_SHOW)
_print_list('only(sta&ch)', res.only_bc, MAX_SHOW)

# CSV書き出し
_write_list(OUT_DIR, 'all_mea_sta_ch', res.all_abc)
_write_list(OUT_DIR, 'only_mea', res.only_a)
_write_list(OUT_DIR, 'only_sta', res.only_b)
_write_list(OUT_DIR, 'only_ch', res.only_c)
_write_list(OUT_DIR, 'only_mea_sta', res.only_ab)
_write_list(OUT_DIR, 'only_mea_ch', res.only_ac)
_write_list(OUT_DIR, 'only_sta_ch', res.only_bc)

print(f'\nWrote CSVs to: {OUT_DIR}')


# %%
def station_key(code: str) -> tuple[str, str | None]:
	"""ざっくり「同一局」を表すためのキーを作る。
	- "NET.STA_2" / "NET.STA2" -> ("STA", "2")
	- "DP2ABU" / "E3DDR"       -> ("ABU", "2") / ("DDR", "3")
	- "ABASH2"                 -> ("ABASH", "2")
	- "AGUNI"                  -> ("AGUNI", None)
	"""
	s = str(code).strip()

	# 1) NET.STA(_)?N
	if '.' in s:
		net, sta = s.split('.', 1)
		m = re.fullmatch(r'([A-Za-z0-9]+?)(?:_?(\d+))?', sta)
		if m is None:
			return (sta, None)
		base, idx = m.group(1), m.group(2)
		return (base, idx)

	# 2) NETNSTA (例: DP2ABU, E3DDR)
	m = re.fullmatch(r'([A-Za-z]{1,3})(\d)([A-Za-z0-9]+)', s)
	if m:
		idx = m.group(2)
		base = m.group(3)
		return (base, idx)

	# 3) STAN（末尾数字が枝番）
	m = re.fullmatch(r'([A-Za-z0-9]+?)(\d+)', s)
	if m:
		base, idx = m.group(1), m.group(2)
		return (base, idx)

	return (s, None)


def build_index(codes: Iterable[str]) -> dict[tuple[str, str | None], list[str]]:
	idx: dict[tuple[str, str | None], list[str]] = defaultdict(list)
	for c in codes:
		idx[station_key(c)].append(c)
	for k in list(idx.keys()):
		idx[k] = sorted(set(idx[k]))
	return dict(idx)


def candidate_matches(
	left: list[str],
	right: list[str],
	*,
	max_print: int = 50,
	mode: str = 'base+idx',
) -> list[tuple[str, list[str]]]:
	"""Left の各コードに対して right 側で候補を探す。
	mode:
	  - "base+idx": (base, idx) 完全一致で探す
	  - "base": baseだけ一致（枝番違いも候補扱い）
	"""
	right_index = build_index(right)

	if mode == 'base+idx':
		pairs = []
		for c in left:
			key = station_key(c)
			cand = right_index.get(key, [])
			if cand:
				pairs.append((c, cand))
		pairs.sort(key=lambda x: (len(x[1]), x[0]))
		return pairs[:max_print]

	if mode == 'base':
		base_to_right: dict[str, list[str]] = defaultdict(list)
		for c in right:
			base, idx = station_key(c)
			base_to_right[base].append(c)
		for b in list(base_to_right.keys()):
			base_to_right[b] = sorted(set(base_to_right[b]))

		pairs = []
		for c in left:
			base, idx = station_key(c)
			cand = base_to_right.get(base, [])
			if cand:
				pairs.append((c, cand))
		pairs.sort(key=lambda x: (len(x[1]), x[0]))
		return pairs[:max_print]

	raise ValueError(f'unknown mode: {mode}')


def print_candidates(title: str, pairs: list[tuple[str, list[str]]]) -> None:
	print(f'\n=== {title} ===')
	print(f'pairs: {len(pairs)}')
	for src, cands in pairs:
		print(f'{src} -> {cands}')


# --- まず、あなたの結果に合わせて only 群を作る（venn3 使わず直接） ---
mea_only = sorted(mea_set - sta_set - ch_set)
sta_only = sorted(sta_set - mea_set - ch_set)
ch_only = sorted(ch_set - mea_set - sta_set)

mea_sta_only = sorted((mea_set & sta_set) - ch_set)
mea_ch_only = sorted((mea_set & ch_set) - sta_set)
sta_ch_only = sorted((sta_set & ch_set) - mea_set)

print('=== recomputed counts ===')
print('only_mea:', len(mea_only))
print('only_sta:', len(sta_only))
print('only_ch :', len(ch_only))
print('only(mea&sta):', len(mea_sta_only))
print('only(mea&ch) :', len(mea_ch_only))
print('only(sta&ch) :', len(sta_ch_only))

# --- どれだけ“命名ゆれ”で救えるか：候補マッチを出す ---
# 1) 強め一致（base+idx）
print_candidates(
	'only_mea  vs only_ch  (base+idx)',
	candidate_matches(mea_only, ch_only, max_print=80, mode='base+idx'),
)
print_candidates(
	'only_sta  vs only_ch  (base+idx)',
	candidate_matches(sta_only, ch_only, max_print=80, mode='base+idx'),
)
print_candidates(
	'only_mea&sta vs only_ch (base+idx)',
	candidate_matches(mea_sta_only, ch_only, max_print=80, mode='base+idx'),
)

# 2) ゆるめ一致（baseだけ：枝番違いも拾う）
print_candidates(
	'only_mea  vs only_ch  (base-only)',
	candidate_matches(mea_only, ch_only, max_print=80, mode='base'),
)
print_candidates(
	'only_sta  vs only_ch  (base-only)',
	candidate_matches(sta_only, ch_only, max_print=80, mode='base'),
)
# %%
