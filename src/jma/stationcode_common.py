# src/jma/stationcode_common.py
from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

import pandas as pd


_MONTH_COL_RE = re.compile(r"^\d{4}-\d{2}$")


def normalize_code(x: object) -> str:
	if x is None or pd.isna(x):
		return ""
	s = unicodedata.normalize("NFKC", str(x))
	s = re.sub(r"\s+", "", s.strip())
	return s.upper()


def normalize_network_code(code: object) -> str:
	s = "" if code is None or pd.isna(code) else str(code).strip()
	if not s or s.lower() == "nan":
		return ""
	m = re.fullmatch(r"(\d+)([A-Za-z]+)?", s)
	if m:
		digits = m.group(1).zfill(4)
		suf = "" if m.group(2) is None else m.group(2).upper()
		return digits + suf
	return s.upper()


def month_columns(df: "pd.DataFrame") -> list[str]:
	cols = [c for c in df.columns if _MONTH_COL_RE.fullmatch(str(c))]
	cols.sort()
	return cols


def pick_preferred_network_code(codes: Sequence[object]) -> str:
	codes_norm = [normalize_network_code(c) for c in codes]
	codes_norm = [c for c in codes_norm if c]
	if not codes_norm:
		return ""

	digits = [c for c in codes_norm if c.isdigit()]
	if digits:
		return sorted((int(c), c) for c in digits)[0][1]
	return sorted(codes_norm)[0]


def pick_one_network_code(field: object) -> str:
	s = "" if field is None or pd.isna(field) else str(field).strip()
	if not s:
		return ""
	parts = [p.strip() for p in s.split(";") if p.strip()]
	if not parts:
		return ""
	return pick_preferred_network_code(parts)
