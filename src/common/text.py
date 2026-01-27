import pandas as pd


def normalize_comment(value: object, *, unknown: str = 'Unknown') -> str:
	"""Normalize whitespace while treating NaN-like values as unknown."""
	if pd.isna(value):
		return unknown
	return ' '.join(str(value).split())
