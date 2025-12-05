# %%
import pandas as pd
from make_hypoinverse_arc import validate_columns


def match_das_events_to_jma(
	epic_df: pd.DataFrame,
	das_df: pd.DataFrame,
	max_dt_sec: float = 30.0,
) -> pd.DataFrame:
	"""DAS イベントと JMA イベントを「最も近い時刻」でマッチングする。

	戻り値 DataFrame の列:
	  - das_event_id              : DAS 側の event_id（連番）
	  - das_time                  : DAS 側の代表時刻（event_time_peak）
	  - event_id                  : 対応する JMA event_id（無い場合は <NA>）
	  - origin_time               : 対応する JMA origin_time（無い場合は NaT）
	  - dt_sec                    : das_time - origin_time の秒差（マッチ無しは NaN）
	  - match_status              : 'matched' | 'das_only'
	  - nearest_event_id_before   : DAS ピックより早い側で最も近い JMA event_id
	  - nearest_origin_time_before: その origin_time
	  - dt_sec_before             : das_time - nearest_origin_time_before の秒差（早い側なので >=0）
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

	# --- 1) これまで通り「±max_dt_sec 以内」の最近傍で matched / das_only を判定 ---
	mapping = pd.merge_asof(
		das_events,
		jma,
		left_on='das_time',
		right_on='origin_time',
		direction='nearest',
		tolerance=pd.Timedelta(seconds=max_dt_sec),
	)

	mapping['dt_sec'] = (
		mapping['das_time'] - mapping['origin_time']
	).dt.total_seconds()
	mapping.loc[mapping['origin_time'].isna(), 'dt_sec'] = float('nan')

	mapping['match_status'] = 'matched'
	mapping.loc[mapping['event_id'].isna(), 'match_status'] = 'das_only'

	# --- 2) 「DAS ピックより早い側で最も近い JMA」を別途計算 ---
	prev = pd.merge_asof(
		das_events,
		jma,
		left_on='das_time',
		right_on='origin_time',
		direction='backward',  # 常に「DAS より前の中で最も近い」もの
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

	# --- 3) 1) の mapping に「早い側の最も近いイベント情報」をくっつける ---
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
	"""JMA と DAS のイベント対応状況を一覧にした DataFrame を返す。

	戻り値 DataFrame の列:
	  - event_id                  : JMA event_id（DAS のみのものは <NA>）
	  - origin_time               : JMA origin_time（DAS のみのものは NaT）
	  - das_event_id              : DAS 側の event_id（JMA のみのものは <NA>）
	  - das_time                  : DAS 側代表時刻（JMA のみのものは NaT）
	  - dt_sec                    : das_time - origin_time（マッチ無しは NaN）
	  - match_status              : 'matched' | 'das_only' | 'jma_only'
	  - das_score                 : epic_df 側の das_score
	  - score                     : DAS 側の score（matched / das_only のみ値が入る）
	  - nearest_event_id_before   : DAS ピックより早い側で最も近い JMA event_id
	  - nearest_origin_time_before: その origin_time
	  - dt_sec_before             : das_time - nearest_origin_time_before の秒差
	  - slowness_s_per_m          : DAS イベントの最終直線傾き（RANSAC）
	  - intercept_s               : DAS イベントの最終直線切片（RANSAC）
	"""
	validate_columns(epic_df, ['event_id', 'origin_time', 'das_score'], 'epic_df')
	validate_columns(das_df, ['event_id', 'event_time_peak', 'score'], 'das_df')

	# DAS ⇔ JMA の時刻マッチング（ここで nearest_* も付与される）
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
	# nearest_* 系は存在しないので NaN/NaT
	jma_only['nearest_event_id_before'] = pd.NA
	jma_only['nearest_origin_time_before'] = pd.NaT
	jma_only['dt_sec_before'] = float('nan')
	# DAS トレンド情報も無いので NaN
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


def extract_das_phase_records(
	epic_df: pd.DataFrame,
	das_df: pd.DataFrame,
	max_dt_sec: float = 30.0,
	*,
	phase_weight_code: int = 3,
	station_prefix: str = 'D',
	channel_width: int = 4,
) -> list[dict]:
	"""DAS ピックから Hypoinverse 用フェーズレコードを生成する。

	戻り値: extract_phase_records と同じ形式の list[dict]
	    {'event_id', 'station_code', 'phase_type', 'weight', 'time'}
	"""
	validate_columns(das_df, ['event_id', 'channel', 'pick_time', 'invalid'], 'das_df')

	mapping = match_das_events_to_jma(epic_df, das_df, max_dt_sec)
	mapping_matched = mapping[mapping['match_status'] == 'matched'].copy()
	if mapping_matched.empty:
		return []

	# JMA 側 event_id を明示的に jma_event_id にしてリンク表を作る
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

	records: list[dict] = []
	for row in das.itertuples():
		jma_event_id = int(row.jma_event_id)  # ここが JMA イベントID
		ch = int(row.channel)
		sta_code = f'{station_prefix}{ch:0{channel_width}d}'
		t = row.pick_time

		records.append(
			{
				'event_id': jma_event_id,
				'station_code': sta_code,
				'phase_type': 'P',
				'weight': phase_weight_code,  # DAS は pha weight = 3
				'time': t,
			}
		)

	return records


if __name__ == '__main__':
	epic_df = pd.read_csv(
		'/workspace/data/arrivetime/arrivetime_epicenters_mobara2020.csv'
	)

	if 'das_score' not in epic_df.columns:
		epic_df = epic_df.copy()
		epic_df['das_score'] = pd.NA

	das_df = pd.read_csv(
		'/home/dcuser/mobara2025/proc/proc_continuous_das/das_picks_20200215_20200301.csv'
	)
	das_events_df = pd.read_csv(
		'/home/dcuser/mobara2025/proc/proc_continuous_das/events_summary_20200215_20200301.csv'
	)

	epic_df = epic_df.copy()
	epic_df['origin_dt'] = pd.to_datetime(epic_df['origin_time'])

	target_start_date = '2020-02-15'
	target_end_date = '2020-03-01'
	start = pd.to_datetime(target_start_date)
	end = pd.to_datetime(target_end_date) + pd.Timedelta(days=1)

	# 期間内の JMA イベント
	epic_period = epic_df[
		(epic_df['origin_dt'] >= start) & (epic_df['origin_dt'] < end)
	].reset_index(drop=True)

	# 期間内の DAS ピック
	das_period = das_df.copy()
	das_period['date'] = pd.to_datetime(das_period['date']).dt.date
	das_period = das_period[
		(das_period['date'] >= start.date()) & (das_period['date'] < end.date())
	].reset_index(drop=True)

	# 期間内の DAS イベントサマリ（傾き・切片入り）
	das_events_period = das_events_df.copy()
	das_events_period['event_time'] = pd.to_datetime(das_events_period['event_time'])
	das_events_period = das_events_period[
		(das_events_period['event_time'] >= start)
		& (das_events_period['event_time'] < end)
	].reset_index(drop=True)

	summary = summarize_jma_das_events(
		epic_period,
		das_period,
		max_dt_sec=120.0,
		das_events_df=das_events_period,
	)

	# 両方にあるイベント
	matched = summary[summary['match_status'] == 'matched']
	print('=== matched (JMA + DAS) ===')
	print(
		matched[
			['event_id', 'origin_time', 'das_event_id', 'das_time', 'dt_sec']
		].head()
	)

	# JMA にしかないイベント
	jma_only = summary[summary['match_status'] == 'jma_only']
	print('=== JMA only ===')
	print(jma_only[['event_id', 'origin_time']].head())

	# DAS にしかないイベント
	das_only = summary[summary['match_status'] == 'das_only']
	print('=== DAS only ===')
	print(das_only[['das_event_id', 'das_time']].head())

	# RANSAC 傾きから速度を計算してフィルタ
	das_only = das_only.copy()
	das_only['velocity'] = 1.0 / das_only['slowness_s_per_m']
	das_only_filtered = das_only[
		(das_only['score'] > 0.7)
		& (das_only['velocity'] < 0.0)
		& (das_only['velocity'] > -10000.0)
	]
	# 必要ならここで print(das_only_filtered) など
	for score in [1, 2, 3]:
		acc = len(matched[matched.das_score == score]) / (
			len(jma_only[jma_only.das_score == score])
			+ len(matched[matched.das_score == score])
		)
		print(
			f'score={score}: matched={len(matched[matched.das_score == score])}, jma_only={len(jma_only[jma_only.das_score == score])}, acc={acc:.3f}'
		)
