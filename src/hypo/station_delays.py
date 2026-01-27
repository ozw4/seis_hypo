from __future__ import annotations

import numpy as np
import pandas as pd


def compute_delay_from_elevation_m(elevation_m: float, vel_kms: float) -> float:
	"""Elevation_m から station delay[s] を計算する。

	Hypoinverse の古い flat-earth モデル（CRH/CRE/CRT など）では station elevation を
	直接は使わないため、標高による走時差を station delay で近似したいケースがある。

	- elevation_m: station elevation [m] (positive up)
	- vel_kms: wave velocity [km/s]

	戻り値は秒。elevation_m が NaN のときは 0 を返す。
	"""
	if not np.isfinite(vel_kms) or float(vel_kms) <= 0.0:
		raise ValueError(f'vel_kms must be positive, got: {vel_kms}')

	if not np.isfinite(elevation_m):
		return 0.0

	return float(elevation_m) / 1000.0 / float(vel_kms)


def compute_pdelay_from_elevation_m(elevation_m: float, vp_kms: float) -> float:
	"""Elevation_m から P station delay[s] を計算する。

	Hypoinverse の古い flat-earth モデル（CRH/CRE/CRT など）では station elevation を
	直接は使わないため、標高による走時差を station delay で近似したいケースがある。

	- elevation_m: station elevation [m] (positive up)
	- vp_kms: P velocity [km/s]

	戻り値は秒。elevation_m が NaN のときは 0 を返す。
	"""
	return compute_delay_from_elevation_m(elevation_m, vp_kms)


def compute_sdelay_from_elevation_m(elevation_m: float, vs_kms: float) -> float:
	"""Elevation_m から S station delay[s] を計算する。"""
	return compute_delay_from_elevation_m(elevation_m, vs_kms)


def add_pdelay1_from_elevation(
	df: pd.DataFrame,
	*,
	vp_kms: float,
	elevation_col: str = 'Elevation_m',
	out_col: str = 'pdelay1',
) -> pd.DataFrame:
	"""Station DataFrame に pdelay1 列を追加して返す。

	- elevation_col が無い場合は即時に例外で失敗する。
	- 既に out_col があれば上書きする。
	"""
	if elevation_col not in df.columns:
		raise ValueError(f"station df missing column: '{elevation_col}'")

	out = df.copy()
	vals = out[elevation_col].astype(float).to_numpy()
	pdelay = np.array(
		[compute_pdelay_from_elevation_m(v, float(vp_kms)) for v in vals], dtype=float
	)
	out[out_col] = pdelay
	return out


def add_p_and_s_delays_from_elevation(
	df: pd.DataFrame,
	*,
	vp_kms: float,
	vs_kms: float,
	elevation_col: str = 'Elevation_m',
	out_col_p: str = 'pdelay1',
	out_col_s: str = 'pdelay2',
) -> pd.DataFrame:
	"""Station DataFrame に P/S の station delay を追加して返す。

	Hypoinverse station format #2 の delay フィールドは 2 つあり、
	通常は crust model #1/#2 用の P delay として使うが、SAL で P model と
	独立 S model を対応付けている場合は「model #1 の delay = P 用」「model #2 の delay = S 用」
	として使える（jma_mobara_hypoinverse と同じ運用）。
	"""
	if elevation_col not in df.columns:
		raise ValueError(f"station df missing column: '{elevation_col}'")

	out = df.copy()
	vals = out[elevation_col].astype(float).to_numpy()
	out[out_col_p] = np.array(
		[compute_pdelay_from_elevation_m(v, float(vp_kms)) for v in vals], dtype=float
	)
	out[out_col_s] = np.array(
		[compute_sdelay_from_elevation_m(v, float(vs_kms)) for v in vals], dtype=float
	)
	return out
