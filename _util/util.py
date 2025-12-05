import numpy as np
import pandas as pd


def mj_to_mw(mj: np.ndarray | float) -> np.ndarray | float:
	"""気象庁マグニチュード Mj をモーメントマグニチュード Mw に変換
	Mw ≈ 0.88 * Mj + 0.54
	"""
	return 0.88 * mj + 0.54


def validate_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
	missing = [c for c in required if c not in df.columns]
	if missing:
		raise ValueError(f'{label} に必要な列がありません: {missing}')
