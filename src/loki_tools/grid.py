from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite
from pathlib import Path
from typing import Literal

import pandas as pd

from common.core import validate_columns
from common.geo import latlon_to_local_xy_km
from common.stations import normalize_station_rows

DEFAULT_LOKI_HEADER_OUT = Path('db/header.hdr')


@dataclass(frozen=True)
class GridSpec:
	"""1つに統一したグリッド仕様。

	- LOKI header.hdr に必要な最小情報
	- グリッド提案時のメタ情報（pad/zmax/center_mode）も同居

	単位:
		x0,y0,z0,dx,dy,dz は km
		lat/lon は deg
	"""

	nx: int
	ny: int
	nz: int

	x0_km: float
	y0_km: float
	z0_km: float

	dx_km: float
	dy_km: float
	dz_km: float

	lat0_deg: float
	lon0_deg: float

	pad_km: float = 0.0
	zmax_km: float = 0.0
	center_mode: str = 'fixed'

	def extent_xy_km(self) -> tuple[float, float, float, float]:
		x1 = self.x0_km + self.dx_km * (self.nx - 1)
		y1 = self.y0_km + self.dy_km * (self.ny - 1)
		return self.x0_km, x1, self.y0_km, y1


def _validate_grid_basic(
	nx: int, ny: int, nz: int, dx: float, dy: float, dz: float
) -> None:
	if nx <= 0 or ny <= 0 or nz <= 0:
		raise ValueError('nx, ny, nz must be positive')
	if dx <= 0 or dy <= 0 or dz <= 0:
		raise ValueError('dx_km, dy_km, dz_km must be positive')


def propose_grid_from_stations(
	stations_df: pd.DataFrame,
	*,
	dx_km: float = 1.0,
	dy_km: float = 1.0,
	dz_km: float = 1.0,
	pad_km: float = 10.0,
	xy_half_width_km: float | None = None,
	z0_km: float = -5.0,
	zmax_km: float = 80.0,
	center_mode: Literal['fixed', 'mean', 'median'] = 'fixed',
	lat0_deg: float | None = None,
	lon0_deg: float | None = None,
) -> GridSpec:
	"""Station 分布から LOKI用グリッドを提案する。

	xy_half_width_km:
		None のときは stations の min/max に pad_km を足して XY 範囲を決める。
		指定したときは XY を [-xy_half_width_km, +xy_half_width_km] の対称ボックスに固定する。
		このとき pad_km は XY には適用しない（ぴったり指定を優先）。

	center_mode:
		- "fixed"  : ここで渡した lat0/lon0 をグリッド中心に採用
		- "mean"   : stations の平均を中心
		- "median" : stations の中央値を中心

	stations_df:
		必須列: station, lat, lon
		任意列: elevation_m

	返す GridSpec は LOKI header と同じ定義を持つ。
	"""
	if stations_df.empty:
		raise ValueError('stations_df is empty')

	validate_columns(stations_df, ['station', 'lat', 'lon'], 'station_df')

	lat_vals = stations_df['lat'].to_numpy(dtype=float)
	lon_vals = stations_df['lon'].to_numpy(dtype=float)

	if center_mode == 'fixed':
		if lat0_deg is None or lon0_deg is None:
			raise ValueError(
				"lat0_deg/lon0_deg must be provided when center_mode='fixed'"
			)
		lat0 = float(lat0_deg)
		lon0 = float(lon0_deg)
	elif center_mode == 'mean':
		lat0 = float(lat_vals.mean())
		lon0 = float(lon_vals.mean())
	elif center_mode == 'median':
		lat0 = float(pd.Series(lat_vals).median())
		lon0 = float(pd.Series(lon_vals).median())
	else:
		raise ValueError(f'unsupported center_mode: {center_mode}')

	x_km, y_km = latlon_to_local_xy_km(
		lat_deg=lat_vals,
		lon_deg=lon_vals,
		lat0_deg=lat0,
		lon0_deg=lon0,
	)

	if xy_half_width_km is None:
		xmin = float(x_km.min()) - pad_km
		xmax = float(x_km.max()) + pad_km
		ymin = float(y_km.min()) - pad_km
		ymax = float(y_km.max()) + pad_km
	else:
		half = float(xy_half_width_km)
		if not isfinite(half) or half <= 0.0:
			raise ValueError('xy_half_width_km must be a finite positive number')

		abs_x = pd.Series(x_km).abs()
		abs_y = pd.Series(y_km).abs()
		viol = (abs_x > half) | (abs_y > half)
		if bool(viol.any()):
			sta = stations_df['station'].astype(str).reset_index(drop=True)
			bad = sta[viol].tolist()
			max_abs_x = float(abs_x.max())
			max_abs_y = float(abs_y.max())
			head = ','.join(bad[:10])
			raise ValueError(
				'xy_half_width_km is too small to cover stations: '
				f'max_abs_x={max_abs_x:.3f} km, max_abs_y={max_abs_y:.3f} km, half={half:.3f} km, '
				f'violations={len(bad)}, stations={head}'
			)

		xmin = -half
		xmax = half
		ymin = -half
		ymax = half

	if xmax <= xmin or ymax <= ymin:
		raise ValueError('invalid station extent after padding')

	nx = int(ceil((xmax - xmin) / dx_km)) + 1
	ny = int(ceil((ymax - ymin) / dy_km)) + 1

	if zmax_km <= z0_km:
		raise ValueError('zmax_km must be greater than z0_km')

	nz = int(ceil((zmax_km - z0_km) / dz_km)) + 1

	_validate_grid_basic(nx, ny, nz, dx_km, dy_km, dz_km)

	return GridSpec(
		nx=nx,
		ny=ny,
		nz=nz,
		x0_km=xmin,
		y0_km=ymin,
		z0_km=z0_km,
		dx_km=dx_km,
		dy_km=dy_km,
		dz_km=dz_km,
		lat0_deg=lat0,
		lon0_deg=lon0,
		pad_km=pad_km,
		zmax_km=zmax_km,
		center_mode=center_mode,
	)


def format_loki_header_lines(
	grid: GridSpec,
	stations_df: pd.DataFrame,
) -> list[str]:
	"""GridSpec + station 情報から LOKI header.hdr の行を生成。

	stations_df:
		必須列: station, lat, lon
		任意列: elevation_m

	注:
		component 重複がある場合は
		station 単位に代表行へ集約した DataFrame を渡すのが安全。
	"""
	_validate_grid_basic(grid.nx, grid.ny, grid.nz, grid.dx_km, grid.dy_km, grid.dz_km)
	df = normalize_station_rows(stations_df, require_elevation=False)

	lines: list[str] = []
	lines.append(f'{grid.nx} {grid.ny} {grid.nz}')
	lines.append(f'{grid.x0_km:.3f} {grid.y0_km:.3f} {grid.z0_km:.3f}')
	lines.append(f'{grid.dx_km:.3f} {grid.dy_km:.3f} {grid.dz_km:.3f}')
	lines.append(f'{grid.lat0_deg:.6f} {grid.lon0_deg:.6f}')

	for sta, lat, lon, elev_m in df[
		['station', 'lat', 'lon', 'elevation_m']
	].itertuples(index=False, name=None):
		lines.append(f'{sta} {float(lat):.6f} {float(lon):.6f} {float(elev_m):.2f}')

	return lines


def format_loki_header_text(
	grid: GridSpec,
	stations_df: pd.DataFrame,
) -> str:
	return '\n'.join(format_loki_header_lines(grid, stations_df)) + '\n'


def write_loki_header(
	grid: GridSpec,
	stations_df: pd.DataFrame,
	out_path: str | Path = DEFAULT_LOKI_HEADER_OUT,
) -> Path:
	out_path = Path(out_path)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(format_loki_header_text(grid, stations_df))
	return out_path
