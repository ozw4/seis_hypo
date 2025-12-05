from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from common.core import validate_columns


def map_prob_to_hypoinverse_weight(
	w_conf: float,
	rules: Sequence[tuple[float, int]] | None = None,
	*,
	default_weight: int = 4,
) -> int:
	"""確率 w_conf(0〜1) を Hypoinverse の weight(0〜4) にマップする.

	Parameters
	----------
	w_conf : float
		ピックモデルの信頼度(例: P/S probability)。
	rules : sequence of (threshold, weight), optional
		大きい threshold から順に評価されるルール。
		例: [(0.7, 0), (0.6, 1), (0.5, 2), (0.3, 3)]
		w_conf >= threshold となった最初の weight を返す。
		None の場合は上の例と同じデフォルトルールを用いる。
	default_weight : int, default 4
		どのルールにも該当しなかった場合に返す weight。

	Returns
	-------
	int
		Hypoinverse weight (0〜4想定)。

	"""
	if not np.isfinite(w_conf):
		return int(default_weight)

	if rules is None:
		rules = ((0.7, 0), (0.6, 1), (0.5, 2), (0.3, 3))

	val = float(w_conf)
	for thr, w in rules:
		if val >= float(thr):
			return int(w)
	return int(default_weight)


def extract_ml_pick_phase_records(
	picks_df: pd.DataFrame,
	*,
	weight_mapper: Callable[[float], int] | None = None,
) -> list[dict]:
	"""ピックモデルの出力 CSV から Hypoinverse 用フェーズレコードを生成する.

	想定する列
	----------
	- event_id     : int
	- station_code : str
	- Phase        : 'P' or 'S'
	- pick_time    : str or datetime64 (UTC/JST は呼び出し側の約束)
	- w_conf       : float (ピック信頼度; 例: probability)

	Parameters
	----------
	picks_df : DataFrame
		ピックモデル出力の一覧。
	weight_mapper : callable, optional
		w_conf -> weight(int) へ変換する関数。
		None の場合は map_prob_to_hypoinverse_weight を用いる。

	Returns
	-------
	list[dict]
		Hypoinverse フェーズレコードのリスト。
		各要素は {'event_id', 'station_code', 'phase_type', 'weight', 'time'}。

	"""
	validate_columns(
		picks_df,
		['event_id', 'station_code', 'Phase', 'pick_time', 'w_conf'],
		'ML picks CSV',
	)

	if weight_mapper is None:
		weight_mapper = map_prob_to_hypoinverse_weight

	records: list[dict] = []

	for _, row in picks_df.iterrows():
		eid = int(row['event_id'])

		phase_raw = row['Phase']
		phase = phase_raw.strip().upper() if isinstance(phase_raw, str) else ''
		if phase not in ('P', 'S'):
			continue

		v = row['pick_time']
		if pd.isna(v):
			continue
		t = pd.to_datetime(v)

		w_conf = float(row['w_conf'])
		weight = int(weight_mapper(w_conf))

		records.append(
			{
				'event_id': eid,
				'station_code': str(row['station_code']),
				'phase_type': phase,
				'weight': weight,
				'time': t,
			}
		)

	return records
