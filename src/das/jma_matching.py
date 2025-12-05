# src/das/jma_matching.py
from __future__ import annotations

import pandas as pd

from common.core import validate_columns


def match_das_events_to_jma(
	epic_df: pd.DataFrame,
	das_df: pd.DataFrame,
	max_dt_sec: float = 30.0,
) -> pd.DataFrame:
	"""DAS イベントと JMA イベントを「最も近い時刻」でマッチングする。

	Parameters
	----------
	epic_df :
	    JMA arrivetime_epicenters 相当の DataFrame。
	    必須カラム: ['event_id', 'origin_time']。
	das_df :
	    DAS ピックから作ったイベント情報を含む DataFrame。
	    少なくとも ['event_id', 'event_time_peak'] を含むこと。
	max_dt_sec :
	    DAS 側の代表時刻 das_time と JMA origin_time の許容最大差 [s]。

	Returns
	-------
	pd.DataFrame
	    各 DAS イベント(das_event_id) に対して、最も近い JMA イベントと
	    「DAS より前で最も近い JMA」を付与した対応表。

	    列:
	      - das_event_id              : DAS 側の event_id（連番を想定）
	      - das_time                  : DAS 側の代表時刻（event_time_peak）
	      - event_id                  : 対応する JMA event_id（無い場合は <NA>）
	      - origin_time               : 対応する JMA origin_time（無い場合は NaT）
	      - dt_sec                    : das_time - origin_time の秒差（マッチ無しは NaN）
	      - match_status              : 'matched' | 'das_only'
	      - nearest_event_id_before   : DAS ピックより早い側で最も近い JMA event_id
	      - nearest_origin_time_before: その origin_time
	      - dt_sec_before             : das_time - nearest_origin_time_before の秒差

	"""
	validate_columns(epic_df, ['event_id', 'origin_time'], 'epic_df')
	validate_columns(das_df, ['event_id', 'event_time_peak'], 'das_df')

	# --- JMA 側（時刻ソート） ---
	jma = epic_df[['event_id', 'origin_time']].copy()
	jma['origin_time'] = pd.to_datetime(jma['origin_time'])
	jma = jma.sort_values('origin_time')

	# --- DAS 側：イベント代表時刻 ---
	das_events = das_df[['event_id', 'event_time_peak']].drop_duplicates().copy()
	das_events['event_time_peak'] = pd.to_datetime(das_events['event_time_peak'])
	das_events = das_events.sort_values('event_time_peak')
	das_events.rename(
		columns={'event_id': 'das_event_id', 'event_time_peak': 'das_time'},
		inplace=True,
	)

	# 1) ±max_dt_sec 以内の最近傍で matched / das_only を判定
	mapping = pd.merge_asof(
		das_events,
		jma,
		left_on='das_time',
		right_on='origin_time',
		direction='nearest',
		tolerance=pd.Timedelta(seconds=float(max_dt_sec)),
	)

	mapping['dt_sec'] = (
		mapping['das_time'] - mapping['origin_time']
	).dt.total_seconds()
	mapping.loc[mapping['origin_time'].isna(), 'dt_sec'] = float('nan')

	mapping['match_status'] = 'matched'
	mapping.loc[mapping['event_id'].isna(), 'match_status'] = 'das_only'

	# 2) DAS ピックより早い側で最も近い JMA を別途計算
	prev = pd.merge_asof(
		das_events,
		jma,
		left_on='das_time',
		right_on='origin_time',
		direction='backward',
	)

	prev['dt_sec_before'] = (prev['das_time'] - prev['origin_time']).dt.total_seconds()
	prev.loc[prev['origin_time'].isna(), 'dt_sec_before'] = float('nan')

	prev = prev[
		[
			'das_event_id',
			'das_time',
			'event_id',
			'origin_time',
			'dt_sec_before',
		]
	].copy()
	prev.rename(
		columns={
			'event_id': 'nearest_event_id_before',
			'origin_time': 'nearest_origin_time_before',
		},
		inplace=True,
	)

	# 3) mapping に「早い側最も近いイベント情報」をマージ
	mapping = mapping.merge(
		prev,
		on=['das_event_id', 'das_time'],
		how='left',
	)

	mapping['das_event_id'] = mapping['das_event_id'].astype(int)
	mapping['event_id'] = mapping['event_id'].astype('Int64')
	mapping['nearest_event_id_before'] = mapping['nearest_event_id_before'].astype(
		'Int64'
	)

	return mapping[
		[
			'das_event_id',
			'das_time',
			'event_id',
			'origin_time',
			'dt_sec',
			'match_status',
			'nearest_event_id_before',
			'nearest_origin_time_before',
			'dt_sec_before',
		]
	]


def summarize_jma_das_events(
	epic_df: pd.DataFrame,
	das_df: pd.DataFrame,
	max_dt_sec: float = 30.0,
	das_events_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
	"""JMA と DAS のイベント対応状況を 1 行 1 イベントにまとめたサマリ表を作る。

	Parameters
	----------
	epic_df :
	    JMA arrivetime_epicenters 相当。必須カラム:
	    ['event_id', 'origin_time', 'das_score']。
	das_df :
	    DAS ピック CSV。必須カラム:
	    ['event_id', 'event_time_peak', 'score']。
	max_dt_sec :
	    DAS 側と JMA 側のマッチングで使う時間トレランス。
	das_events_df :
	    DAS イベントの RANSAC 傾き・切片を含むサマリ DataFrame（任意）。
	    必須カラム: ['event_id', 'slowness_s_per_m', 'intercept_s']。

	Returns
	-------
	pd.DataFrame
	    列:
	      - event_id
	      - origin_time
	      - das_event_id
	      - das_time
	      - dt_sec
	      - match_status  : 'matched' | 'das_only' | 'jma_only'
	      - das_score     : JMA 側 das_score
	      - score         : DAS 側 score
	      - nearest_event_id_before
	      - nearest_origin_time_before
	      - dt_sec_before
	      - slowness_s_per_m
	      - intercept_s

	"""
	validate_columns(epic_df, ['event_id', 'origin_time', 'das_score'], 'epic_df')
	validate_columns(das_df, ['event_id', 'event_time_peak', 'score'], 'das_df')

	# DAS ⇔ JMA の時刻マッチング（ここで nearest_* も付与）
	mapping = match_das_events_to_jma(epic_df, das_df, max_dt_sec)

	# JMA 側情報（origin_time, das_score）
	jma = epic_df[['event_id', 'origin_time', 'das_score']].copy()
	jma['origin_time'] = pd.to_datetime(jma['origin_time'])

	# DAS 側の event_id ごとの score を 1 行にまとめる
	das_score_df = das_df[['event_id', 'score']].copy()
	das_score_df = das_score_df.dropna(subset=['event_id'])
	das_score_df['event_id'] = das_score_df['event_id'].astype(int)
	das_score_df = das_score_df.drop_duplicates(subset=['event_id'])
	das_score_df.rename(columns={'event_id': 'das_event_id'}, inplace=True)

	# mapping（matched + das_only）に DAS score を付与
	mapping = mapping.merge(das_score_df, on='das_event_id', how='left')

	# DAS イベントサマリ（傾き・切片）を付与
	if das_events_df is not None:
		validate_columns(
			das_events_df,
			['event_id', 'slowness_s_per_m', 'intercept_s'],
			'das_events_df',
		)
		das_trend_df = das_events_df[
			['event_id', 'slowness_s_per_m', 'intercept_s']
		].copy()
		das_trend_df = das_trend_df.dropna(subset=['event_id'])
		das_trend_df['event_id'] = das_trend_df['event_id'].astype(int)
		das_trend_df = das_trend_df.drop_duplicates(subset=['event_id'])
		das_trend_df.rename(columns={'event_id': 'das_event_id'}, inplace=True)

		mapping = mapping.merge(das_trend_df, on='das_event_id', how='left')

	# 傾き・切片カラムが存在しない場合は追加して NaN で埋める
	if 'slowness_s_per_m' not in mapping.columns:
		mapping['slowness_s_per_m'] = float('nan')
	if 'intercept_s' not in mapping.columns:
		mapping['intercept_s'] = float('nan')

	# mapping 側に JMA の das_score を付与
	mapping_for_summary = mapping.merge(
		jma[['event_id', 'das_score']],
		on='event_id',
		how='left',
	)

	# matched な JMA の event_id 一覧
	matched_jma_ids = (
		mapping_for_summary.loc[
			mapping_for_summary['match_status'] == 'matched', 'event_id'
		]
		.dropna()
		.astype(int)
		.unique()
	)

	# JMA のみ（対応する DAS が無い）イベント
	jma_only = jma[~jma['event_id'].isin(matched_jma_ids)].copy()
	jma_only['das_event_id'] = pd.NA
	jma_only['das_time'] = pd.NaT
	jma_only['dt_sec'] = float('nan')
	jma_only['match_status'] = 'jma_only'
	jma_only['score'] = float('nan')
	jma_only['nearest_event_id_before'] = pd.NA
	jma_only['nearest_origin_time_before'] = pd.NaT
	jma_only['dt_sec_before'] = float('nan')
	jma_only['slowness_s_per_m'] = float('nan')
	jma_only['intercept_s'] = float('nan')

	# カラム順を揃える
	mapping_for_summary = mapping_for_summary[
		[
			'event_id',
			'origin_time',
			'das_event_id',
			'das_time',
			'dt_sec',
			'match_status',
			'das_score',  # JMA 側
			'score',  # DAS 側
			'nearest_event_id_before',
			'nearest_origin_time_before',
			'dt_sec_before',
			'slowness_s_per_m',
			'intercept_s',
		]
	]
	jma_only = jma_only[
		[
			'event_id',
			'origin_time',
			'das_event_id',
			'das_time',
			'dt_sec',
			'match_status',
			'das_score',
			'score',
			'nearest_event_id_before',
			'nearest_origin_time_before',
			'dt_sec_before',
			'slowness_s_per_m',
			'intercept_s',
		]
	]

	summary = pd.concat([mapping_for_summary, jma_only], ignore_index=True)
	summary = summary.sort_values(['origin_time', 'das_time'], na_position='last')

	return summary
