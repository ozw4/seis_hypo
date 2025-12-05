import numpy as np
import pandas as pd

from util import validate_columns


def map_eqt_weight(w_conf: float) -> int:
	"""EqT の確率 w_conf(0〜1) を Hypoinverse の weight(0〜4) にマップする簡易ルール。"""
	if not np.isfinite(w_conf):
		return 4  # よくわからないものは最弱

	if w_conf >= 0.7:
		return 0  # 超自信あり
	if w_conf >= 0.6:
		return 1
	if w_conf >= 0.5:
		return 2
	if w_conf >= 0.3:
		return 3
	return 4  # かなり怪しい


def extract_eqt_phase_records(
	eqt_df: pd.DataFrame,
) -> list[dict]:
	"""EqT ピック CSV から Hypoinverse 用フェーズレコードを生成する。

	想定する列
	----------
	- event_id : int
	- station_code : str
	- Phase : 'P' or 'S'
	- pick_time : str or datetime
	- w_conf : float (EqT 確率)

	戻り値
	-------
	list[dict]
	{'event_id', 'station_code', 'phase_type', 'weight', 'time'}
	"""
	validate_columns(
		eqt_df,
		[
			'event_id',
			'station_code',
			'Phase',
			'pick_time',
			'w_conf',
		],
		'EqT picks CSV',
	)

	records: list[dict] = []

	for _, row in eqt_df.iterrows():
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
		w = map_eqt_weight(w_conf)

		records.append(
			dict(
				event_id=eid,
				station_code=str(row['station_code']),
				phase_type=phase,
				weight=w,
				time=t,
			)
		)

	return records
