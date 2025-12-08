from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, radians
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loki_tools.vel1d import convert_1dvel_to_nll_layers, read_1dvel

from jma.station_reader import stations_within_radius
from loki_tools.grid import GridSpec, propose_grid_from_stations, write_loki_header
from nonlinloc.control import write_nll_control_files_ps

# ---- Defaults matching your current workflow ----
DEFAULT_CHANNEL_TABLE = Path(
	'/workspace/proc/util/hinet_util/hinet_channelstbl_20251007'
)
DEFAULT_VEL1D_SRC = Path('velocity/vjma2001')
DEFAULT_LAYERS_OUT = Path('velocity/jma2001.layers')

DEFAULT_NLL_RUN_DIR = Path('nll/run')
DEFAULT_NLL_MODEL_DIR = Path('nll/model')
DEFAULT_NLL_TIME_DIR = Path('nll/time')

DEFAULT_LOKI_HEADER_OUT = Path('db/header.hdr')
DEFAULT_QC_FIG_DIR = Path('qc')


@dataclass(frozen=True)
class QcConfig:
	# Station selection
	center_lat: float
	center_lon: float
	radius_km: float
	channel_table_path: Path = DEFAULT_CHANNEL_TABLE

	# Grid proposal (LOKI-first)
	dx_km: float = 1.0
	dy_km: float = 1.0
	dz_km: float = 1.0
	pad_km: float = 10.0
	z0_km: float = -5.0
	zmax_km: float = 80.0

	# 1D velocity -> LAYER
	vel1d_src: Path = DEFAULT_VEL1D_SRC
	layers_out: Path = DEFAULT_LAYERS_OUT

	# NonLinLoc control output
	model_label: str = 'jma2001'
	nll_run_dir: Path = DEFAULT_NLL_RUN_DIR
	nll_model_dir: Path = DEFAULT_NLL_MODEL_DIR
	nll_time_dir: Path = DEFAULT_NLL_TIME_DIR

	# LOKI header
	loki_header_out: Path = DEFAULT_LOKI_HEADER_OUT

	# QC figures
	fig_dir: Path = DEFAULT_QC_FIG_DIR


def _ensure_dir(p: Path) -> None:
	p.mkdir(parents=True, exist_ok=True)


def _latlon_to_xy_km(
	lat_deg: np.ndarray,
	lon_deg: np.ndarray,
	*,
	lat0_deg: float,
	lon0_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
	lat0_rad = radians(lat0_deg)
	km_per_deg_lat = 111.32
	km_per_deg_lon = 111.32 * cos(lat0_rad)

	x_km = (lon_deg - lon0_deg) * km_per_deg_lon
	y_km = (lat_deg - lat0_deg) * km_per_deg_lat
	return x_km, y_km


def _xy_km_to_latlon(
	x_km: float,
	y_km: float,
	*,
	lat0_deg: float,
	lon0_deg: float,
) -> tuple[float, float]:
	lat0_rad = radians(lat0_deg)
	km_per_deg_lat = 111.32
	km_per_deg_lon = 111.32 * cos(lat0_rad)

	lat = lat0_deg + (y_km / km_per_deg_lat)
	lon = lon0_deg + (x_km / km_per_deg_lon)
	return lat, lon


def _grid_corners_latlon(grid: GridSpec) -> np.ndarray:
	x0, x1, y0, y1 = grid.extent_xy_km()
	corners_xy = [
		(x0, y0),
		(x1, y0),
		(x1, y1),
		(x0, y1),
		(x0, y0),
	]
	corners_latlon = [
		_xy_km_to_latlon(x, y, lat0_deg=grid.lat0_deg, lon0_deg=grid.lon0_deg)
		for x, y in corners_xy
	]
	return np.array(corners_latlon, dtype=float)  # shape (5, 2) as (lat, lon)


def plot_qc_station_map_latlon(
	stations_df: pd.DataFrame,
	*,
	center_lat: float,
	center_lon: float,
	radius_km: float,
	grid: GridSpec,
	out_png: Path,
) -> Path:
	required = {'station', 'lat', 'lon'}
	missing = required.difference(stations_df.columns)
	if missing:
		raise ValueError(f'stations_df missing required columns: {sorted(missing)}')

	df = stations_df.copy()
	df['lat'] = df['lat'].astype(float)
	df['lon'] = df['lon'].astype(float)

	fig, ax = plt.subplots(figsize=(7, 7))

	ax.scatter(df['lon'], df['lat'], s=20)
	ax.scatter([center_lon], [center_lat], s=60, marker='x')

	# radius circle (approx, in degrees)
	# We draw a small parametric circle using local km->deg scaling around center.
	theta = np.linspace(0, 2 * pi, 361)
	lat0_rad = radians(center_lat)
	km_per_deg_lat = 111.32
	km_per_deg_lon = 111.32 * cos(lat0_rad)

	dlat = (radius_km / km_per_deg_lat) * np.sin(theta)
	dlon = (radius_km / km_per_deg_lon) * np.cos(theta)

	ax.plot(center_lon + dlon, center_lat + dlat, linewidth=1.0)

	# grid footprint
	corners = _grid_corners_latlon(grid)
	ax.plot(corners[:, 1], corners[:, 0], linewidth=1.5)

	ax.set_xlabel('Longitude (deg)')
	ax.set_ylabel('Latitude (deg)')
	ax.set_title('QC: stations, search center/radius, grid footprint')

	# tight view around stations + grid
	lon_all = np.concatenate([df['lon'].to_numpy(), corners[:, 1]])
	lat_all = np.concatenate([df['lat'].to_numpy(), corners[:, 0]])
	lon_min, lon_max = float(lon_all.min()), float(lon_all.max())
	lat_min, lat_max = float(lat_all.min()), float(lat_all.max())

	pad_lon = max(0.1, (lon_max - lon_min) * 0.15)
	pad_lat = max(0.1, (lat_max - lat_min) * 0.15)

	ax.set_xlim(lon_min - pad_lon, lon_max + pad_lon)
	ax.set_ylim(lat_min - pad_lat, lat_max + pad_lat)

	fig.tight_layout()
	fig.savefig(out_png, dpi=200)
	plt.close(fig)
	return out_png


def plot_qc_station_xy_and_grid(
	stations_df: pd.DataFrame,
	*,
	grid: GridSpec,
	out_png: Path,
) -> Path:
	required = {'station', 'lat', 'lon'}
	missing = required.difference(stations_df.columns)
	if missing:
		raise ValueError(f'stations_df missing required columns: {sorted(missing)}')

	df = stations_df.copy()
	df['lat'] = df['lat'].astype(float)
	df['lon'] = df['lon'].astype(float)

	lat_arr = df['lat'].to_numpy()
	lon_arr = df['lon'].to_numpy()

	x_km, y_km = _latlon_to_xy_km(
		lat_arr,
		lon_arr,
		lat0_deg=grid.lat0_deg,
		lon0_deg=grid.lon0_deg,
	)

	x0, x1, y0, y1 = grid.extent_xy_km()

	fig, ax = plt.subplots(figsize=(7, 7))
	ax.scatter(x_km, y_km, s=20)
	ax.scatter([0.0], [0.0], s=60, marker='x')

	# grid rectangle in local km
	rect_x = [x0, x1, x1, x0, x0]
	rect_y = [y0, y0, y1, y1, y0]
	ax.plot(rect_x, rect_y, linewidth=1.5)

	ax.set_xlabel('x East (km) relative to lat0/lon0')
	ax.set_ylabel('y North (km) relative to lat0/lon0')
	ax.set_title('QC: stations in local XY and grid coverage')

	x_all = np.concatenate([x_km, np.array([x0, x1])])
	y_all = np.concatenate([y_km, np.array([y0, y1])])

	pad_x = max(2.0, (float(x_all.max()) - float(x_all.min())) * 0.15)
	pad_y = max(2.0, (float(y_all.max()) - float(y_all.min())) * 0.15)

	ax.set_xlim(float(x_all.min()) - pad_x, float(x_all.max()) + pad_x)
	ax.set_ylim(float(y_all.min()) - pad_y, float(y_all.max()) + pad_y)

	fig.tight_layout()
	fig.savefig(out_png, dpi=200)
	plt.close(fig)
	return out_png


def plot_qc_1d_velocity(
	*,
	vel1d_src: Path,
	layers_out: Path,
	out_png: Path,
) -> Path:
	rows = read_1dvel(vel1d_src, strict=False)
	if not rows:
		raise ValueError(f'no rows parsed from {vel1d_src}')

	depth = np.array([r.depth_km for r in rows], dtype=float)
	vp = np.array([r.vp_km_s for r in rows], dtype=float)
	vs = np.array([r.vs_km_s for r in rows], dtype=float)

	fig, ax = plt.subplots(figsize=(6, 6))

	ax.plot(vp, depth)
	ax.plot(vs, depth)

	ax.invert_yaxis()
	ax.set_xlabel('Velocity (km/s)')
	ax.set_ylabel('Depth (km)')
	ax.set_title('QC: 1D velocity (Vp/Vs)')

	# quick sanity annotation
	ax.text(
		0.02,
		0.02,
		f'src: {vel1d_src}\nlayers: {layers_out}',
		transform=ax.transAxes,
		ha='left',
		va='bottom',
		fontsize=8,
	)

	fig.tight_layout()
	fig.savefig(out_png, dpi=200)
	plt.close(fig)
	return out_png


def plot_qc_depth_range(
	grid: GridSpec,
	*,
	out_png: Path,
) -> Path:
	z0 = grid.z0_km
	z1 = grid.z0_km + grid.dz_km * (grid.nz - 1)

	fig, ax = plt.subplots(figsize=(6, 2.2))
	ax.plot([z0, z1], [0, 0], linewidth=6)
	ax.scatter([z0, z1], [0, 0], s=50)

	ax.set_yticks([])
	ax.set_xlabel('Depth axis (km; sign convention as defined in your header/TRANS)')
	ax.set_title('QC: grid depth coverage (z0 to z1)')

	ax.text(
		0.02,
		0.65,
		f'z0={z0:.2f} km, z1={z1:.2f} km, dz={grid.dz_km:.2f} km, nz={grid.nz}',
		transform=ax.transAxes,
		ha='left',
		va='center',
		fontsize=9,
	)

	fig.tight_layout()
	fig.savefig(out_png, dpi=200)
	plt.close(fig)
	return out_png


def qc_prepare_loki_and_nll_inputs(cfg: QcConfig) -> dict[str, Path]:
	"""You asked to QC up to:
	    1) 半径で局抽出(rows)
	    2) station分布から grid提案(center fixed)
	    3) LOKI header書き出し
	    4) 1Dvel -> LAYER
	    5) control(P/S) 出力

	This function executes exactly those steps and produces QC figures.
	"""
	_ensure_dir(cfg.fig_dir)

	# 1) station rows
	stations_df = stations_within_radius(
		cfg.center_lat,
		cfg.center_lon,
		cfg.radius_km,
		cfg.channel_table_path,
		output='rows',
	)

	# 2) grid proposal (fixed center = search center)
	grid = propose_grid_from_stations(
		stations_df,
		dx_km=cfg.dx_km,
		dy_km=cfg.dy_km,
		dz_km=cfg.dz_km,
		pad_km=cfg.pad_km,
		z0_km=cfg.z0_km,
		zmax_km=cfg.zmax_km,
		center_mode='fixed',
		lat0_deg=cfg.center_lat,
		lon0_deg=cfg.center_lon,
	)

	# 3) LOKI header
	header_path = write_loki_header(
		grid,
		stations_df,
		out_path=cfg.loki_header_out,
	)

	# 4) 1Dvel -> LAYER
	layers_path = Path(
		convert_1dvel_to_nll_layers(
			src=cfg.vel1d_src,
			out=cfg.layers_out,
			strict=False,
		)
	)

	# 5) NonLinLoc control P/S
	control_p_path, control_s_path = write_nll_control_files_ps(
		grid,
		stations_df,
		model_label=cfg.model_label,
		layers_path=layers_path,
		run_dir=cfg.nll_run_dir,
		vgout_dir=cfg.nll_model_dir,
		gtout_dir=cfg.nll_time_dir,
		quantity='SLOW_LEN',
		gtmode='GRID3D ANGLES_NO',
		depth_km_mode='zero',
	)

	# ---- QC figures ----
	fig1 = plot_qc_station_map_latlon(
		stations_df,
		center_lat=cfg.center_lat,
		center_lon=cfg.center_lon,
		radius_km=cfg.radius_km,
		grid=grid,
		out_png=cfg.fig_dir / 'qc_stations_center_radius_grid_latlon.png',
	)

	fig2 = plot_qc_station_xy_and_grid(
		stations_df,
		grid=grid,
		out_png=cfg.fig_dir / 'qc_stations_and_grid_xy_km.png',
	)

	fig3 = plot_qc_1d_velocity(
		vel1d_src=cfg.vel1d_src,
		layers_out=layers_path,
		out_png=cfg.fig_dir / 'qc_1d_velocity_vp_vs.png',
	)

	fig4 = plot_qc_depth_range(
		grid,
		out_png=cfg.fig_dir / 'qc_grid_depth_range.png',
	)

	summary_txt = cfg.fig_dir / 'qc_summary.txt'
	summary_txt.write_text(
		'\n'.join(
			[
				'QC summary for LOKI/NonLinLoc precompute',
				f'center_lat={cfg.center_lat}, center_lon={cfg.center_lon}, radius_km={cfg.radius_km}',
				f'stations={len(stations_df)}',
				'',
				'GridSpec',
				f'nx ny nz = {grid.nx} {grid.ny} {grid.nz}',
				f'x0 y0 z0 = {grid.x0_km:.3f} {grid.y0_km:.3f} {grid.z0_km:.3f}',
				f'dx dy dz = {grid.dx_km:.3f} {grid.dy_km:.3f} {grid.dz_km:.3f}',
				f'lat0 lon0 = {grid.lat0_deg:.6f} {grid.lon0_deg:.6f}',
				'',
				f'header: {header_path}',
				f'layers: {layers_path}',
				f'control P: {control_p_path}',
				f'control S: {control_s_path}',
				'',
				'Figures',
				str(fig1),
				str(fig2),
				str(fig3),
				str(fig4),
			]
		)
		+ '\n'
	)

	return {
		'stations_csv_like_df_not_saved': Path('-'),
		'header_path': Path(header_path),
		'layers_path': layers_path,
		'control_p_path': Path(control_p_path),
		'control_s_path': Path(control_s_path),
		'fig1_latlon': fig1,
		'fig2_xy': fig2,
		'fig3_vel': fig3,
		'fig4_depth': fig4,
		'summary': summary_txt,
	}


def qc_example_run() -> dict[str, Path]:
	"""コード直書き運用の example entry.
	ここだけ編集すれば、同じQC手順を任意地域に再利用できる。
	"""
	cfg = QcConfig(
		center_lat=35.0,
		center_lon=138.0,
		radius_km=80.0,
		channel_table_path=DEFAULT_CHANNEL_TABLE,
		dx_km=1.0,
		dy_km=1.0,
		dz_km=1.0,
		pad_km=10.0,
		z0_km=-5.0,
		zmax_km=80.0,
		vel1d_src=DEFAULT_VEL1D_SRC,
		layers_out=DEFAULT_LAYERS_OUT,
		model_label='jma2001',
		nll_run_dir=DEFAULT_NLL_RUN_DIR,
		nll_model_dir=DEFAULT_NLL_MODEL_DIR,
		nll_time_dir=DEFAULT_NLL_TIME_DIR,
		loki_header_out=DEFAULT_LOKI_HEADER_OUT,
		fig_dir=DEFAULT_QC_FIG_DIR,
	)

	return qc_prepare_loki_and_nll_inputs(cfg)
