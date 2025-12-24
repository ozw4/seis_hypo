# %%
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ConflictCheckResult:
	exists: bool
	conflict_keys: list[Any]
	conflict_rows: pd.DataFrame
	key_to_values: pd.Series  # index=key, value=list[str] など


def find_key_value_conflicts(
	df: pd.DataFrame,
	key_col: str,
	value_col: str,
	*,
	dropna: bool = True,
	strip_values: bool = True,
	normalize_whitespace: bool = False,
) -> ConflictCheckResult:
	"""同一 key_col に対して value_col が複数種類存在するキーを検出する。

	例: station_code が同じなのに network code が違う行があるか確認。

	Parameters
	----------
	df : pd.DataFrame
	key_col : str
	    グルーピングキー列（例: station_code）
	value_col : str
	    一意性を確認したい値列（例: network code）
	dropna : bool
	    Trueなら key/value の欠損行を除外して判定
	strip_values : bool
	    Trueなら value を文字列化して前後空白を除去（比較のブレ対策）
	normalize_whitespace : bool
	    Trueなら value 内の連続空白を1つに正規化（必要なときだけ）

	Returns
	-------
	ConflictCheckResult

	"""
	if key_col not in df.columns:
		raise KeyError(f'key_col が df に存在しません: {key_col}')
	if value_col not in df.columns:
		raise KeyError(f'value_col が df に存在しません: {value_col}')

	x = df[[key_col, value_col]].copy()

	if dropna:
		x = x.dropna(subset=[key_col, value_col])

	if strip_values or normalize_whitespace:
		v = x[value_col].astype(str)
		if strip_values:
			v = v.str.strip()
		if normalize_whitespace:
			v = v.str.replace(r'\s+', ' ', regex=True)
		x[value_col] = v

	nuniq = x.groupby(key_col)[value_col].nunique()
	conflict_keys = nuniq[nuniq > 1].index.tolist()

	if conflict_keys:
		conflict_rows = df[df[key_col].isin(conflict_keys)].sort_values(
			[key_col, value_col]
		)
		key_to_values = (
			x.groupby(key_col)[value_col]
			.apply(lambda s: sorted(set(s)))
			.loc[conflict_keys]
		)
		return ConflictCheckResult(True, conflict_keys, conflict_rows, key_to_values)

	empty_rows = df.iloc[0:0].copy()
	empty_map = pd.Series(dtype=object)
	return ConflictCheckResult(False, [], empty_rows, empty_map)


# station code from arrival file
mea_file = '/workspace/data/arrivetime/JMA/arrivetime_measurements.csv'
mea_df = pd.read_csv(mea_file, usecols=['station_code', 'station_number'])

# station code from .ch file
ch_file = '/workspace/proc/prepare_data/jma/snapshots/monthly/monthly_presence.csv'
ch_df = pd.read_csv(ch_file)

res = find_key_value_conflicts(ch_df, key_col='station', value_col='network_code')
res2 = find_key_value_conflicts(
	mea_df, key_col='station_code', value_col='station_number'
)

mea_set = set(mea_df['station_code'].dropna().astype(str).str.strip())
ch_set = set(ch_df['station'].dropna().astype(str).str.strip())

both = sorted(mea_set & ch_set)  # 両方にある
only_mea = sorted(mea_set - ch_set)  # meaにしかない
only_ch = sorted(ch_set - mea_set)  # chにしかない

print(f'both: {len(both)}')
print(f'only_mea: {len(only_mea)}')
print(f'only_ch: {len(only_ch)}')

# ---- 中身を表示（長いなら head だけにして）----
print('\n[both]')
print(both)

print('\n[only_mea]')
print(only_mea)

print('\n[only_ch]')
print(only_ch)

out_dir = '/workspace/proc/outputs'
pd.Series(both, name='station_code').to_csv(
	f'{out_dir}/station_codes_both.csv', index=False
)
pd.Series(only_mea, name='station_code').to_csv(
	f'{out_dir}/station_codes_only_mea.csv', index=False
)
pd.Series(only_ch, name='station_code').to_csv(
	f'{out_dir}/station_codes_only_ch.csv', index=False
)
