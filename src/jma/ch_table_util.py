from __future__ import annotations

import pandas as pd

from jma.prepare.inventory import (
	DEFAULT_AXIS_TAIL_CHARS,
	DEFAULT_COMP_PRIORITY,
	_axis_from_component,
	_component_rank,
)
from jma.stationcode_common import normalize_code


def normalize_ch_table_components_to_une(
	ch_df: pd.DataFrame,
	*,
	comp_priority: dict[str, list[str]] | None = None,
	axis_tail_chars: set[str] | None = None,
) -> pd.DataFrame:
	"""Channel table の component を正規化して U/N/E の3成分に揃える。

	- 出現順は使わない（component文字列から軸を推定）
	- stationごとに U/N/E を1本ずつ選ぶ（rank最小を採用）
	- 行順は station の出現順を維持しつつ、各 station 内は U,N,E の順に並べる

	返り値:
	- component 列が U/N/E になった DataFrame（reset_index済み）
	- この行順が read_win32 の出力 ndarray 行順になる前提で使う
	"""
	if 'station' not in ch_df.columns or 'component' not in ch_df.columns:
		raise ValueError('ch_df must have columns: station, component')

	if comp_priority is None:
		comp_priority = DEFAULT_COMP_PRIORITY
	if axis_tail_chars is None:
		axis_tail_chars = DEFAULT_AXIS_TAIL_CHARS

	out_rows: list[pd.Series] = []
	bad_missing: list[str] = []

	# sort=False で station の出現順を維持
	for sta_raw, g in ch_df.groupby('station', sort=False):
		sta = normalize_code(sta_raw)
		if not sta:
			continue

		best_row: dict[str, pd.Series] = {}
		best_rank: dict[str, int] = {}

		for _, r in g.iterrows():
			comp_raw = str(r['component'])
			axis = _axis_from_component(comp_raw, axis_tail_chars)
			if axis not in ('U', 'N', 'E'):
				continue
			rank = _component_rank(comp_raw, axis, comp_priority)

			prev = best_rank.get(axis)
			if prev is None or int(rank) < int(prev):
				best_rank[axis] = int(rank)
				best_row[axis] = r

		missing = [a for a in ('U', 'N', 'E') if a not in best_row]
		if missing:
			bad_missing.append(f'{sta} missing={missing}')
			continue

		for axis in ('U', 'N', 'E'):
			r2 = best_row[axis].copy()
			r2['station'] = sta
			r2['component'] = axis
			out_rows.append(r2)

	if bad_missing:
		raise ValueError(
			f'stations missing U/N/E after normalization: {bad_missing[:30]}'
		)

	if not out_rows:
		raise ValueError('no rows remained after component normalization')

	out = pd.DataFrame(out_rows)
	return out.reset_index(drop=True)
