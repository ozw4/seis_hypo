from __future__ import annotations

from typing import Any


def override_phase_weight_by_station_prefix(
	phases: list[dict[str, Any]],
	station_prefix: str,
	weight: int,
) -> list[dict[str, Any]]:
	"""station_code の prefix に一致するフェーズの weight を上書きする.

	Hypoinverse のフェーズ weight(0〜4想定)を後段で明示的に制御したい用途に使う。

	Parameters
	----------
	phases : list[dict]
		各要素は {'event_id','station_code','phase_type','weight','time'} を想定。
	station_prefix : str
		対象 station_code の prefix (例: 'D' → 'D0001' など)。
	weight : int
		上書きする Hypoinverse weight。

	Returns
	-------
	list[dict]
		weight を上書きした新しいリスト。

	"""
	if not isinstance(station_prefix, str) or not station_prefix:
		raise ValueError('station_prefix must be a non-empty str')

	w = int(weight)
	if w < 0 or w > 9:
		raise ValueError(f'weight out of range: {w}')

	out: list[dict[str, Any]] = []
	for r in phases:
		sta = str(r.get('station_code', ''))
		if sta.startswith(station_prefix):
			r2 = dict(r)
			r2['weight'] = w
			out.append(r2)
		else:
			out.append(r)
	return out
