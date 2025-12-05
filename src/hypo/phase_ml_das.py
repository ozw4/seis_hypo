from __future__ import annotations

import pandas as pd

from common.core import validate_columns
from das.jma_matching import match_das_events_to_jma


def extract_das_phase_records(
	epic_df: pd.DataFrame,
	das_df: pd.DataFrame,
	max_dt_sec: float = 30.0,
	*,
	phase_weight_code: int = 3,
	station_prefix: str = 'D',
	channel_width: int = 4,
) -> list[dict[str, Any]]:
	"""DAS ピックから Hypoinverse 用フェーズレコードを生成する.

	戻り値は extract_phase_records と同じ形式:
		{'event_id', 'station_code', 'phase_type', 'weight', 'time'}

	Parameters
	----------
	epic_df :
		JMA arrivetime_epicenters 相当 (event_id, origin_time を含む)。
	das_df :
		DAS ピック一覧。必須カラム:
		['event_id', 'channel', 'pick_time', 'invalid']。
	max_dt_sec :
		DAS イベントと JMA イベントを時刻でマッチさせる際の許容差。
	phase_weight_code :
		Hypoinverse の pha weight に入れるコード。DAS は 3 をデフォルトにしている。
	station_prefix :
		DAS 仮想ステーション名のプレフィックス (例 'D')。
	channel_width :
		DAS 仮想ステーション番号の桁数 (例 4 → D0001, D0002, ...)。

	Returns
	-------
	list[dict]
		Hypoinverse フェーズ行に変換可能な dict のリスト。

	"""
	validate_columns(das_df, ['event_id', 'channel', 'pick_time', 'invalid'], 'das_df')

	mapping = match_das_events_to_jma(epic_df, das_df, max_dt_sec)
	mapping_matched = mapping[mapping['match_status'] == 'matched'].copy()
	if mapping_matched.empty:
		return []

	# JMA 側 event_id を jma_event_id にしてリンク表を作る
	link = mapping_matched[['das_event_id', 'event_id']].dropna().copy()
	link['event_id'] = link['event_id'].astype(int)
	link.rename(columns={'event_id': 'jma_event_id'}, inplace=True)

	# DAS 側 event_id と das_event_id を対応付ける
	das = das_df.copy()
	das = das.merge(
		link,
		left_on='event_id',  # DAS の event_id
		right_on='das_event_id',
		how='inner',
	)

	das = das[das['invalid'] == False].copy()
	das['pick_time'] = pd.to_datetime(das['pick_time'])

	records: list[dict[str, Any]] = []
	for row in das.itertuples():
		jma_event_id = int(row.jma_event_id)
		ch = int(row.channel)
		sta_code = f'{station_prefix}{ch:0{channel_width}d}'
		t = row.pick_time

		records.append(
			{
				'event_id': jma_event_id,
				'station_code': sta_code,
				'phase_type': 'P',
				'weight': phase_weight_code,
				'time': t,
			}
		)

	return records
