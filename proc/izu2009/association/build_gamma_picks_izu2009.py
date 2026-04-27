# %%
"""Build GaMMA picks CSV for the Izu 2009 WIN32 EqT picks."""

# file: proc/izu2009/association/build_gamma_picks_izu2009.py
#
# Purpose:
# - Convert proc/izu2009/pick/out/eqt_picks_win32_*.csv to one GaMMA picks CSV.
# - Reuse the generic WIN32 EqT -> GaMMA converter.
# - Use '__' between network_code and station_code because Izu station codes can
#   contain '.' (for example N.ITOH and NU.MNI1).

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _REPO_ROOT / 'src'

for _path in (_REPO_ROOT, _SRC_DIR):
	_path_str = str(_path)
	if _path_str not in sys.path:
		sys.path.insert(0, _path_str)

eqt2gamma = importlib.import_module(
	'proc.prepare_data.jma.build_gamma_picks_from_win32_eqt_csv'
)

IN_PICK_CSVS = [
	_REPO_ROOT / 'proc/izu2009/pick/out/eqt_picks_win32_0101.csv',
	_REPO_ROOT / 'proc/izu2009/pick/out/eqt_picks_win32_0203.csv',
	_REPO_ROOT / 'proc/izu2009/pick/out/eqt_picks_win32_0207.csv',
	_REPO_ROOT / 'proc/izu2009/pick/out/eqt_picks_win32_0301.csv',
]

OUT_GAMMA_PICKS_CSV = _REPO_ROOT / 'proc/izu2009/association/in/gamma_picks.csv'

STATION_ID_MODE = 'network_station'
NETWORK_STATION_SEPARATOR = '__'
INCLUDE_TRACE_COLUMNS = True


def _apply_izu2009_settings() -> None:
	eqt2gamma.IN_PICK_CSVS = IN_PICK_CSVS
	eqt2gamma.OUT_GAMMA_PICKS_CSV = OUT_GAMMA_PICKS_CSV
	eqt2gamma.STATION_ID_MODE = STATION_ID_MODE
	eqt2gamma.NETWORK_STATION_SEPARATOR = NETWORK_STATION_SEPARATOR
	eqt2gamma.INCLUDE_TRACE_COLUMNS = INCLUDE_TRACE_COLUMNS


def build_gamma_picks_izu2009() -> pd.DataFrame:
	"""Build Izu 2009 GaMMA picks DataFrame without writing it to disk."""
	_apply_izu2009_settings()
	return eqt2gamma.build_gamma_picks_from_win32_eqt_csv(IN_PICK_CSVS)


def main() -> None:
	"""Run Izu 2009 pick conversion using the constants above."""
	_apply_izu2009_settings()
	eqt2gamma.main()


if __name__ == '__main__':
	main()

# 実行例:
# python proc/izu2009/association/build_gamma_picks_izu2009.py
