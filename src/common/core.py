import pandas as pd


def validate_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
	missing = [c for c in required if c not in df.columns]
	if missing:
		raise ValueError(f'{label} に必要な列がありません: {missing}')
