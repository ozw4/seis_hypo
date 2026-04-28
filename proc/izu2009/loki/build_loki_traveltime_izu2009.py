# %%
"""Build the Izu2009 Loki travel-time database."""

from __future__ import annotations

import importlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2, which
from subprocess import run

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _REPO_ROOT / 'src'

for _path in (_REPO_ROOT, _SRC_DIR):
	_path_str = str(_path)
	if _path_str not in sys.path:
		sys.path.insert(0, _path_str)

core = importlib.import_module('common.core')
geo = importlib.import_module('common.geo')
json_io = importlib.import_module('common.json_io')
loki_grid = importlib.import_module('loki_tools.grid')
nll_control = importlib.import_module('nonlinloc.control')
traveltime_qc = importlib.import_module('qc.nonlinloc.traveltime_tables')

validate_columns = core.validate_columns
latlon_to_local_xy_km = geo.latlon_to_local_xy_km
read_json = json_io.read_json
write_json = json_io.write_json
GridSpec = loki_grid.GridSpec
propose_grid_from_stations = loki_grid.propose_grid_from_stations
write_loki_header = loki_grid.write_loki_header
write_nll_control_files_ps = nll_control.write_nll_control_files_ps
qc_grid2time_outputs_ps = traveltime_qc.qc_grid2time_outputs_ps

IN_STATIONS_CSV = (
	_REPO_ROOT / 'proc/izu2009/prepare_data/profile/stations47/stations_47.csv'
)
IN_LAYERS_PATH = (
	_REPO_ROOT
	/ 'proc/loki_hypo/mobara/mobara_traveltime/velocity/jma2001.layers'
)
IN_ORIGIN_JSON = _REPO_ROOT / 'proc/izu2009/association/in/origin_latlon.json'
IN_GAMMA_CONFIG_JSON = _REPO_ROOT / 'proc/izu2009/association/out/gamma_config.json'

OUT_ROOT = _REPO_ROOT / 'proc/izu2009/loki/traveltime'
OUT_DB_DIR = OUT_ROOT / 'db'
OUT_NLL_RUN_DIR = OUT_ROOT / 'nll/run'
OUT_NLL_MODEL_DIR = OUT_ROOT / 'nll/model'
OUT_VELOCITY_DIR = OUT_ROOT / 'velocity'
OUT_QC_TRAVELTIME_TABLES_DIR = OUT_ROOT / 'qc/traveltime_tables'

OUT_HEADER_PATH = OUT_DB_DIR / 'header.hdr'
OUT_LAYERS_PATH = OUT_VELOCITY_DIR / 'jma2001.layers'
OUT_CONFIG_JSON = OUT_ROOT / 'traveltime_config.json'
OUT_STATIONS_CSV = OUT_ROOT / 'stations_for_loki.csv'

REQUIRED_STATION_COLUMNS = ['network_code', 'station', 'lat', 'lon', 'elevation_m']

MODEL_LABEL = 'jma2001'

DX_KM = 1.0
DY_KM = 1.0
DZ_KM = 1.0
Z_MIN_KM = 0.0
Z_MAX_KM = 30.0
XY_MARGIN_KM = 10.0

NLL_QUANTITY = 'SLOW_LEN'
NLL_GTMODE = 'GRID3D ANGLES_NO'
NLL_DEPTH_KM_MODE = 'zero'
GT_PLFD_EPS = 1.0e-3
GT_PLFD_SWEEP = 0


def _require_file(path: Path, label: str) -> Path:
	path = Path(path)
	if not path.is_file():
		raise FileNotFoundError(f'{label} not found: {path}')
	return path


def _require_executable(name: str) -> str:
	executable = which(name)
	if executable is None:
		raise FileNotFoundError(f'executable not found in PATH: {name}')
	return str(Path(executable).resolve())


def _read_origin(path: Path) -> dict[str, float]:
	obj = read_json(_require_file(path, 'origin_latlon.json'))
	if not isinstance(obj, dict):
		raise TypeError(f'origin_latlon.json must contain a JSON object: {path}')

	missing = [key for key in ('lat0_deg', 'lon0_deg') if key not in obj]
	if missing:
		raise ValueError(f'origin_latlon.json missing keys: {missing}')

	origin = {
		'lat0_deg': float(obj['lat0_deg']),
		'lon0_deg': float(obj['lon0_deg']),
	}
	if not np.isfinite([origin['lat0_deg'], origin['lon0_deg']]).all():
		raise ValueError(f'origin_latlon.json has non-finite origin: {path}')
	return origin


def _read_gamma_config_optional(path: Path) -> tuple[dict | None, str | None]:
	if not path.is_file():
		return None, None

	obj = read_json(path)
	if not isinstance(obj, dict):
		raise TypeError(f'gamma_config.json must contain a JSON object: {path}')
	return obj, str(path.relative_to(_REPO_ROOT))


def _normalize_stations(stations_csv: Path, origin: dict[str, float]) -> pd.DataFrame:
	stations_csv = _require_file(stations_csv, 'stations_47.csv')
	stations = pd.read_csv(
		stations_csv,
		dtype={'network_code': 'string', 'station': 'string'},
	)
	validate_columns(
		stations,
		REQUIRED_STATION_COLUMNS,
		f'stations CSV: {stations_csv}',
	)
	if stations.empty:
		raise ValueError(f'stations CSV is empty: {stations_csv}')

	stations = stations.copy()
	stations['network_code'] = stations['network_code'].astype('string').str.strip()
	stations['station'] = stations['station'].astype('string').str.strip()

	for column in ['network_code', 'station']:
		values = stations[column]
		if values.isna().any() or (values == '').any():
			raise ValueError(f'{stations_csv}: empty {column} found')

	for column in ['lat', 'lon', 'elevation_m']:
		values = pd.to_numeric(stations[column], errors='raise').astype('float64')
		if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
			raise ValueError(f'{stations_csv}: non-finite {column} found')
		stations[column] = values

	duplicates = stations.loc[stations['station'].duplicated(keep=False)]
	if not duplicates.empty:
		examples = duplicates['station'].drop_duplicates().head(20).tolist()
		raise ValueError(f'duplicated station code detected: {examples}')

	x_km, y_km = latlon_to_local_xy_km(
		stations['lat'].to_numpy(dtype=float),
		stations['lon'].to_numpy(dtype=float),
		lat0_deg=origin['lat0_deg'],
		lon0_deg=origin['lon0_deg'],
	)

	return pd.DataFrame(
		{
			'station': stations['station'].astype(str),
			'network_code': stations['network_code'].astype(str),
			'lat': stations['lat'].astype(float),
			'lon': stations['lon'].astype(float),
			'elevation_m': stations['elevation_m'].astype(float),
			'x_km': x_km.astype(float),
			'y_km': y_km.astype(float),
			'z_km': 0.0,
		}
	).sort_values(['station'], kind='mergesort').reset_index(drop=True)


def _build_grid(stations: pd.DataFrame, origin: dict[str, float]) -> GridSpec:
	return propose_grid_from_stations(
		stations,
		dx_km=DX_KM,
		dy_km=DY_KM,
		dz_km=DZ_KM,
		pad_km=XY_MARGIN_KM,
		xy_half_width_km=None,
		z0_km=Z_MIN_KM,
		zmax_km=Z_MAX_KM,
		center_mode='fixed',
		lat0_deg=origin['lat0_deg'],
		lon0_deg=origin['lon0_deg'],
	)


def _copy_velocity_model(src: Path, dst: Path) -> Path:
	src = _require_file(src, 'velocity model')
	dst.parent.mkdir(parents=True, exist_ok=True)
	copy2(src, dst)
	return dst


def _remove_previous_nll_outputs() -> None:
	patterns = [
		OUT_DB_DIR / '*.time.buf',
		OUT_DB_DIR / '*.time.hdr',
		OUT_DB_DIR / '*.time.*.buf',
		OUT_DB_DIR / '*.time.*.hdr',
		OUT_DB_DIR / '*.mod.buf',
		OUT_DB_DIR / '*.mod.hdr',
		OUT_NLL_MODEL_DIR / '*.mod.buf',
		OUT_NLL_MODEL_DIR / '*.mod.hdr',
	]
	for pattern in patterns:
		for path in pattern.parent.glob(pattern.name):
			if path.is_file():
				path.unlink()


def _rel(path: Path) -> str:
	return str(Path(path).relative_to(_REPO_ROOT))


def _require_complete_grid2time_qc(
	qc_csv: Path,
	*,
	phase: str,
	station_count: int,
) -> None:
	df = pd.read_csv(qc_csv)
	validate_columns(
		df,
		['source', 'hdr_exists', 'buf_exists'],
		f'{phase} Grid2Time QC CSV: {qc_csv}',
	)
	if len(df) != station_count:
		raise ValueError(
			f'{phase}: expected {station_count} Grid2Time rows, got {len(df)}'
		)

	missing = df.loc[~(df['hdr_exists'] & df['buf_exists'])]
	if not missing.empty:
		rows = missing[['source', 'hdr_exists', 'buf_exists']].head(20)
		raise FileNotFoundError(
			f'{phase}: missing Grid2Time outputs: {rows.to_dict(orient="records")}'
		)


def _require_outputs_from_qc(
	*,
	qc_paths: dict[str, Path],
	station_count: int,
) -> list[Path]:
	if not OUT_HEADER_PATH.is_file():
		raise FileNotFoundError(f'generated header.hdr not found: {OUT_HEADER_PATH}')

	expected = 2 * int(station_count)
	dfs = [
		pd.read_csv(qc_paths['p_csv']),
		pd.read_csv(qc_paths['s_csv']),
	]

	time_buf_paths: list[Path] = []
	for df in dfs:
		validate_columns(
			df,
			['source', 'hdr_exists', 'buf_exists', 'buf_path'],
			'Grid2Time QC CSV',
		)

		missing = df.loc[~(df['hdr_exists'] & df['buf_exists'])]
		if not missing.empty:
			rows = missing[['source', 'hdr_exists', 'buf_exists']].head(20)
			raise FileNotFoundError(
				f'missing Grid2Time outputs: {rows.to_dict(orient="records")}'
			)

		time_buf_paths.extend(Path(path) for path in df['buf_path'].astype(str))

	missing_paths = [path for path in time_buf_paths if not path.is_file()]
	if missing_paths:
		head = missing_paths[:20]
		raise FileNotFoundError(f'QC-listed .time.buf files are missing: {head}')

	if len(time_buf_paths) != expected:
		raise ValueError(
			f'expected {expected} .time.buf files from QC, got {len(time_buf_paths)}'
		)

	return sorted(time_buf_paths)


def _write_config(  # noqa: PLR0913
	*,
	grid: GridSpec,
	origin: dict[str, float],
	stations_count: int,
	velocity_model_path: Path,
	gamma_config_path: str | None,
	gamma_config: dict | None,
	control_p_path: Path,
	control_s_path: Path,
	qc_paths: dict[str, Path],
	time_buf_count: int,
) -> None:
	x_min, x_max, y_min, y_max = grid.extent_xy_km()
	gamma_ranges = None
	if gamma_config is not None:
		gamma_ranges = {
			key: gamma_config[key]
			for key in ('x(km)', 'y(km)', 'z(km)')
			if key in gamma_config
		}

	cfg = {
		'generated_at_utc': datetime.now(timezone.utc).isoformat(),
		'model_label': MODEL_LABEL,
		'grid': {
			'nx': grid.nx,
			'ny': grid.ny,
			'nz': grid.nz,
			'x0_km': grid.x0_km,
			'y0_km': grid.y0_km,
			'z0_km': grid.z0_km,
			'dx_km': grid.dx_km,
			'dy_km': grid.dy_km,
			'dz_km': grid.dz_km,
			'x_min_km': x_min,
			'x_max_km': x_max,
			'y_min_km': y_min,
			'y_max_km': y_max,
			'z_min_km': Z_MIN_KM,
			'z_max_km': Z_MAX_KM,
			'xy_margin_km': XY_MARGIN_KM,
			'center_mode': grid.center_mode,
		},
		'origin': origin,
		'station_count': int(stations_count),
		'velocity_model_path': _rel(velocity_model_path),
		'gamma_config_path': gamma_config_path,
		'gamma_ranges': gamma_ranges,
		'paths': {
			'header': _rel(OUT_HEADER_PATH),
			'control_p': _rel(control_p_path),
			'control_s': _rel(control_s_path),
			'db_dir': _rel(OUT_DB_DIR),
			'nll_model_dir': _rel(OUT_NLL_MODEL_DIR),
		},
		'nll': {
			'quantity': NLL_QUANTITY,
			'gtmode': NLL_GTMODE,
			'depth_km_mode': NLL_DEPTH_KM_MODE,
			'gt_plfd_eps': GT_PLFD_EPS,
			'gt_plfd_sweep': GT_PLFD_SWEEP,
		},
		'qc': {
			'traveltime_tables_dir': _rel(OUT_QC_TRAVELTIME_TABLES_DIR),
			'tt_files_p_csv': _rel(qc_paths['p_csv']),
			'tt_files_s_csv': _rel(qc_paths['s_csv']),
			'tt_files_summary': _rel(qc_paths['summary']),
		},
		'time_buf_count': int(time_buf_count),
	}
	OUT_CONFIG_JSON.parent.mkdir(parents=True, exist_ok=True)
	write_json(OUT_CONFIG_JSON, cfg, ensure_ascii=False, indent=2)


def _run_nonlinloc(control_p_path: Path, control_s_path: Path) -> None:
	vel2grid = _require_executable('Vel2Grid')
	grid2time = _require_executable('Grid2Time')

	run([vel2grid, str(control_p_path)], check=True)  # noqa: S603
	run([grid2time, str(control_p_path)], check=True)  # noqa: S603
	run([vel2grid, str(control_s_path)], check=True)  # noqa: S603
	run([grid2time, str(control_s_path)], check=True)  # noqa: S603


def build_loki_traveltime_izu2009() -> tuple[pd.DataFrame, GridSpec, list[Path]]:
	"""Build Izu2009 Loki header, NLL controls, and travel-time buffers."""
	origin = _read_origin(IN_ORIGIN_JSON)
	gamma_config, gamma_config_path = _read_gamma_config_optional(IN_GAMMA_CONFIG_JSON)
	stations = _normalize_stations(IN_STATIONS_CSV, origin)
	grid = _build_grid(stations, origin)

	for path in (OUT_ROOT, OUT_DB_DIR, OUT_NLL_RUN_DIR, OUT_NLL_MODEL_DIR):
		path.mkdir(parents=True, exist_ok=True)

	_remove_previous_nll_outputs()

	stations.to_csv(OUT_STATIONS_CSV, index=False)
	layers_path = _copy_velocity_model(IN_LAYERS_PATH, OUT_LAYERS_PATH)
	write_loki_header(grid, stations, out_path=OUT_HEADER_PATH)

	control_p_path, control_s_path = write_nll_control_files_ps(
		grid,
		stations,
		model_label=MODEL_LABEL,
		layers_path=layers_path,
		run_dir=OUT_NLL_RUN_DIR,
		vgout_dir=OUT_NLL_MODEL_DIR,
		gtout_dir=OUT_DB_DIR,
		quantity=NLL_QUANTITY,
		gtmode=NLL_GTMODE,
		depth_km_mode=NLL_DEPTH_KM_MODE,
		gt_plfd_eps=GT_PLFD_EPS,
		gt_plfd_sweep=GT_PLFD_SWEEP,
	)

	_run_nonlinloc(Path(control_p_path), Path(control_s_path))
	qc_paths = qc_grid2time_outputs_ps(
		control_p_path,
		control_s_path,
		out_dir=OUT_QC_TRAVELTIME_TABLES_DIR,
	)
	_require_complete_grid2time_qc(
		qc_paths['p_csv'],
		phase='P',
		station_count=len(stations),
	)
	_require_complete_grid2time_qc(
		qc_paths['s_csv'],
		phase='S',
		station_count=len(stations),
	)
	time_buf_paths = _require_outputs_from_qc(
		qc_paths=qc_paths,
		station_count=len(stations),
	)
	_write_config(
		grid=grid,
		origin=origin,
		stations_count=len(stations),
		velocity_model_path=layers_path,
		gamma_config_path=gamma_config_path,
		gamma_config=gamma_config,
		control_p_path=control_p_path,
		control_s_path=control_s_path,
		qc_paths=qc_paths,
		time_buf_count=len(time_buf_paths),
	)

	print('Wrote:', OUT_STATIONS_CSV.relative_to(_REPO_ROOT))
	print('Wrote:', OUT_HEADER_PATH.relative_to(_REPO_ROOT))
	print('Wrote:', OUT_CONFIG_JSON.relative_to(_REPO_ROOT))
	print('Stations:', len(stations))
	print(f'Grid: nx={grid.nx}, ny={grid.ny}, nz={grid.nz}')
	print(f'Depth range km: {Z_MIN_KM:.1f} -> {Z_MAX_KM:.1f}')
	print('Velocity model:', OUT_LAYERS_PATH.name)
	print('Travel-time buffers:', len(time_buf_paths))

	return stations, grid, time_buf_paths


def main() -> None:
	"""Run the Izu2009 Loki travel-time builder."""
	build_loki_traveltime_izu2009()


if __name__ == '__main__':
	main()
