# src/jma/stationcode_presence.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from jma.stationcode_common import month_columns, normalize_code, normalize_network_code


@dataclass(frozen=True)
class PresenceDB:
	pres: pd.DataFrame
	ch_set: set[str]
	month_cols: list[str]


def load_presence_db(pres_csv: str | Path) -> PresenceDB:
	p = Path(pres_csv)
	if not p.is_file():
		raise FileNotFoundError(p)

	df = pd.read_csv(p, low_memory=False)

	req = {"network_code", "station"}
	if not req.issubset(df.columns):
		raise ValueError(f"monthly_presence missing columns: {sorted(req - set(df.columns))}")

	month_cols = month_columns(df)
	if not month_cols:
		raise ValueError("no YYYY-MM presence columns found in monthly_presence")

	df = df.copy()
	df["ch_key"] = df["station"].astype(str).map(normalize_code)
	df["network_code"] = df["network_code"].map(normalize_network_code)

	ch_set = set(df["ch_key"].unique().tolist())
	keep_cols = ["ch_key", "network_code"] + month_cols
	return PresenceDB(pres=df[keep_cols].copy(), ch_set=ch_set, month_cols=month_cols)
