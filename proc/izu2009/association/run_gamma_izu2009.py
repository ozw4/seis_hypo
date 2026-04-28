# %%
"""Run GaMMA phase association for the Izu 2009 WIN32 EqT picks."""

# file: proc/izu2009/association/run_gamma_izu2009.py
#
# Required inputs:
# - proc/izu2009/association/in/gamma_picks.csv
# - proc/izu2009/association/in/gamma_stations.csv
# - proc/izu2009/association/in/gamma_vel_jma2001.json

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _REPO_ROOT / 'src'

for _path in (_REPO_ROOT, _SRC_DIR):
	_path_str = str(_path)
	if _path_str not in sys.path:
		sys.path.insert(0, _path_str)

from gamma_workflow.run import run_gamma_from_csvs  # noqa: E402

PICKS_CSV = _REPO_ROOT / 'proc/izu2009/association/in/gamma_picks.csv'
STATIONS_CSV = _REPO_ROOT / 'proc/izu2009/association/in/gamma_stations.csv'
VEL_MODEL_JSON = _REPO_ROOT / 'proc/izu2009/association/in/gamma_vel_jma2001.json'
OUT_DIR = _REPO_ROOT / 'proc/izu2009/association/out'

METHOD = 'BGMM'
USE_DBSCAN = True
USE_AMPLITUDE = False
OVERSAMPLE_FACTOR_BGMM = 5

USE_EIKONAL_1D = True
EIKONAL_H_KM = 1.0

XY_MARGIN_KM = 10.0
Z_RANGE_KM = (0.0, 20.0)

DBSCAN_EPS_SEC: float | None = None
DBSCAN_EPS_SIGMA = 2.0
DBSCAN_EPS_MULT = 1.0
DBSCAN_MIN_SAMPLES = 3
DBSCAN_MIN_CLUSTER_SIZE = 100
DBSCAN_MAX_TIME_SPACE_RATIO = 10.0

MIN_PICKS_PER_EQ = 10
MIN_P_PICKS_PER_EQ = 3
MIN_S_PICKS_PER_EQ = 3
MAX_SIGMA11_SEC = 3.0
MAX_SIGMA22_LOG10_MS = 1.0
MAX_SIGMA12_COV = 1.0

NCPU = max(1, (os.cpu_count() or 1) - 1)


def _validate_settings() -> None:
	"""Fail fast for invalid top-of-file settings."""
	if METHOD not in {'BGMM', 'GMM'}:
		raise ValueError(f'METHOD must be BGMM or GMM: {METHOD}')
	if USE_AMPLITUDE:
		raise ValueError(
			'Izu2009 EqT picks do not include amplitudes. Set USE_AMPLITUDE=False.'
		)
	if Z_RANGE_KM[0] < 0.0 or Z_RANGE_KM[0] >= Z_RANGE_KM[1]:
		raise ValueError(f'invalid Z_RANGE_KM: {Z_RANGE_KM}')
	if XY_MARGIN_KM < 0.0:
		raise ValueError(f'XY_MARGIN_KM must be non-negative: {XY_MARGIN_KM}')
	if EIKONAL_H_KM <= 0.0:
		raise ValueError(f'EIKONAL_H_KM must be positive: {EIKONAL_H_KM}')
	if DBSCAN_EPS_SEC is not None and DBSCAN_EPS_SEC <= 0.0:
		raise ValueError(f'DBSCAN_EPS_SEC must be positive or None: {DBSCAN_EPS_SEC}')
	if MIN_PICKS_PER_EQ <= 0:
		raise ValueError(f'MIN_PICKS_PER_EQ must be positive: {MIN_PICKS_PER_EQ}')
	if MIN_P_PICKS_PER_EQ < 0:
		raise ValueError(f'MIN_P_PICKS_PER_EQ must be >= 0: {MIN_P_PICKS_PER_EQ}')
	if MIN_S_PICKS_PER_EQ < 0:
		raise ValueError(f'MIN_S_PICKS_PER_EQ must be >= 0: {MIN_S_PICKS_PER_EQ}')


def _required_input_files() -> list[Path]:
	"""Return required input files for the current settings."""
	paths = [PICKS_CSV, STATIONS_CSV]
	if USE_EIKONAL_1D:
		paths.append(VEL_MODEL_JSON)
	return paths


def _require_input_files() -> None:
	"""Fail fast when required GaMMA input files are missing."""
	missing = [path for path in _required_input_files() if not path.is_file()]
	if not missing:
		return

	missing_lines = '\n'.join(f'  - {path}' for path in missing)
	commands = '\n'.join(
		[
			'  python proc/izu2009/association/build_gamma_picks_izu2009.py',
			'  python proc/izu2009/association/build_gamma_stations_izu2009.py',
			'  python proc/izu2009/association/build_gamma_velocity_jma2001.py',
		]
	)
	raise FileNotFoundError(
		'missing GaMMA input file(s).\n'
		f'{missing_lines}\n'
		'Build inputs first from the repository root:\n'
		f'{commands}'
	)


def _print_result_summary(result: dict) -> None:
	"""Print concise GaMMA output counts and file paths."""
	events_count = result['events_count']
	config = result['config']

	if events_count == 0:
		print('GaMMA returned 0 events.')

	print(f'stations: {result["stations_count"]}')
	print(f'picks: {result["picks_count"]} (assigned: {result["assigned_count"]})')
	print(f'events: {events_count}')
	print(f'wrote: {result["events_path"]}')
	print(f'wrote: {result["picks_path"]}')
	print(f'wrote: {result["config_path"]}')

	if USE_DBSCAN:
		print(
			'dbscan:',
			f'eps0={config["dbscan_eps"]:.3f}s, '
			f'min_samples={config["dbscan_min_samples"]}, '
			f'min_cluster_size={config["dbscan_min_cluster_size"]}, '
			'max_time_space_ratio='
			f'{config["dbscan_max_time_space_ratio"]}',
		)


def main() -> None:
	"""Run GaMMA association with the Izu 2009 input tables."""
	_validate_settings()
	_require_input_files()

	result = run_gamma_from_csvs(
		picks_csv=PICKS_CSV,
		stations_csv=STATIONS_CSV,
		vel_json=VEL_MODEL_JSON,
		out_dir=OUT_DIR,
		method=METHOD,
		use_dbscan=USE_DBSCAN,
		use_amplitude=USE_AMPLITUDE,
		oversample_factor_bgmm=OVERSAMPLE_FACTOR_BGMM,
		use_eikonal_1d=USE_EIKONAL_1D,
		eikonal_h_km=EIKONAL_H_KM,
		xy_margin_km=XY_MARGIN_KM,
		z_range_km=Z_RANGE_KM,
		dbscan_eps_sec=DBSCAN_EPS_SEC,
		dbscan_eps_sigma=DBSCAN_EPS_SIGMA,
		dbscan_eps_mult=DBSCAN_EPS_MULT,
		dbscan_min_samples=DBSCAN_MIN_SAMPLES,
		dbscan_min_cluster_size=DBSCAN_MIN_CLUSTER_SIZE,
		dbscan_max_time_space_ratio=DBSCAN_MAX_TIME_SPACE_RATIO,
		ncpu=NCPU,
		min_picks_per_eq=MIN_PICKS_PER_EQ,
		min_p_picks_per_eq=MIN_P_PICKS_PER_EQ,
		min_s_picks_per_eq=MIN_S_PICKS_PER_EQ,
		max_sigma11_sec=MAX_SIGMA11_SEC,
		max_sigma22_log10_ms=MAX_SIGMA22_LOG10_MS,
		max_sigma12_cov=MAX_SIGMA12_COV,
	)
	_print_result_summary(result)


if __name__ == '__main__':
	main()
