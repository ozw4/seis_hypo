from __future__ import annotations

from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

from common.core import load_event_json
from common.geo import latlon_to_local_xy_km
from loki_tools.loki_parse import (
	parse_loki_event_dir,
	parse_loki_grid_spec,
	parse_loki_header,
)
from viz.core.fig_io import save_figure


def _load_jma_hypocenter_latlon_deg(event_dir: Path) -> tuple[float, float]:
	"""event.json から (lat, lon) を取り出す。

	トップレベル or extra どちらでも受ける。
	"""
	ev = load_event_json(event_dir)

	lat = ev.get('latitude_deg', None)
	lon = ev.get('longitude_deg', None)
	if lat is None or lon is None:
		extra = ev.get('extra', None)
		if isinstance(extra, dict):
			lat = lat if lat is not None else extra.get('latitude_deg', None)
			lon = lon if lon is not None else extra.get('longitude_deg', None)

	if lat is None or lon is None:
		raise ValueError(
			f'event.json missing JMA lat/lon: {event_dir / "event.json"} '
			'(need latitude_deg/longitude_deg; may be under extra)'
		)

	return float(lat), float(lon)


def _coherence_xy_from_corrmatrix(
	corr: np.ndarray,
	*,
	nx: int,
	ny: int,
	nz: int,
) -> np.ndarray:
	"""corrmatrix_trial_*.npy から XY (ny, nx) の coherence を作る。

	対応フォーマット:
	- 2D: (ny, nx) or (nx, ny)
	- 3D: (nz, ny, nx), (ny, nx, nz), (nx, ny, nz) など軸順の揺れを許容
	      → depth 方向に max して最終的に (ny, nx) を返す
	"""
	if corr.ndim == 2:
		if corr.shape == (ny, nx):
			return corr.astype(np.float32, copy=False)
		if corr.shape == (nx, ny):
			return corr.T.astype(np.float32, copy=False)
		raise ValueError(
			f'corrmatrix 2D shape mismatch: got={corr.shape} expected one of {(ny, nx)} or {(nx, ny)}'
		)

	if corr.ndim != 3:
		raise ValueError(f'corrmatrix must be 2D or 3D, got ndim={corr.ndim}')

	# Canonical and common variants
	if corr.shape == (nz, ny, nx):
		return np.max(corr, axis=0).astype(np.float32, copy=False)  # -> (ny, nx)

	if corr.shape == (ny, nx, nz):
		return np.max(corr, axis=2).astype(np.float32, copy=False)  # -> (ny, nx)

	# Your observed case: (nx, ny, nz)
	if corr.shape == (nx, ny, nz):
		return np.max(corr, axis=2).T.astype(
			np.float32, copy=False
		)  # (nx, ny)->T->(ny, nx)

	# Additional permutations (safe to support)
	if corr.shape == (ny, nz, nx):
		return np.max(corr, axis=1).astype(np.float32, copy=False)  # -> (ny, nx)

	if corr.shape == (nx, nz, ny):
		return np.max(corr, axis=1).T.astype(
			np.float32, copy=False
		)  # (nx, ny)->T->(ny, nx)

	if corr.shape == (nz, nx, ny):
		return np.max(corr, axis=0).T.astype(
			np.float32, copy=False
		)  # (nx, ny)->T->(ny, nx)

	raise ValueError(
		'corrmatrix 3D shape mismatch: '
		f'got={corr.shape} expected a permutation of (nx, ny, nz)={(nx, ny, nz)}'
	)


def plot_loki_event_coherence_xy_overlay(
	*,
	event_dir: str | Path,
	loki_output_dir: str | Path,
	header_path: str | Path,
	out_png: str | Path | None = None,
	trial: int = 0,
	dpi: int = 200,
	show_station_labels: bool = True,
	station_label_max: int = 60,
	station_label_fontsize: int = 7,
	station_label_offset_km: float = 0.4,
) -> Path | None:
	"""各イベントについて、XY 上に Coherence + (LOKI/JMA/Stations) を重ねて保存する。

	- coherence: corrmatrix_trial_*.npy を depth 方向に max して XY に射影
	- LOKI: *.loc の x_km/y_km
	- JMA: event.json の lat/lon を header の lat0/lon0 で local XY に変換
	- Stations: header.hdr の station lat/lon を同様に local XY に変換

	返り値:
		書き出した PNG の Path。corrmatrix が無い場合は None。
	"""
	event_dir = Path(event_dir)
	loki_output_dir = Path(loki_output_dir)
	header_path = Path(header_path)

	if not event_dir.is_dir():
		raise FileNotFoundError(f'event_dir not found: {event_dir}')
	if not loki_output_dir.is_dir():
		raise FileNotFoundError(f'loki_output_dir not found: {loki_output_dir}')
	if not header_path.is_file():
		raise FileNotFoundError(f'header not found: {header_path}')

	event_id = event_dir.name
	ev_out_dir = loki_output_dir / event_id
	if not ev_out_dir.is_dir():
		raise FileNotFoundError(f'loki event output dir not found: {ev_out_dir}')

	res = parse_loki_event_dir(ev_out_dir)
	if not res.corrmatrix_paths:
		return None

	preferred = ev_out_dir / f'corrmatrix_trial_{trial}.npy'
	corr_path = preferred if preferred.is_file() else res.corrmatrix_paths[0]

	grid = parse_loki_grid_spec(header_path)
	header = parse_loki_header(header_path)
	stations_df = header.stations_df
	name_col = None
	for c in ('station', 'sta', 'name'):
		if c in stations_df.columns:
			name_col = c
			break
	corr = np.load(corr_path)
	coh_xy = _coherence_xy_from_corrmatrix(corr.T, nx=grid.nx, ny=grid.ny, nz=grid.nz)

	# extent for imshow (km)
	x0 = float(grid.x0_km)
	y0 = float(grid.y0_km)
	x1 = x0 + float(grid.dx_km) * float(grid.nx - 1)
	y1 = y0 + float(grid.dy_km) * float(grid.ny - 1)

	# stations (local XY)
	sx, sy = latlon_to_local_xy_km(
		stations_df['lat'].to_numpy(float),
		stations_df['lon'].to_numpy(float),
		lat0_deg=grid.lat0_deg,
		lon0_deg=grid.lon0_deg,
	)

	# JMA hypocenter (local XY)
	jma_lat, jma_lon = _load_jma_hypocenter_latlon_deg(event_dir)
	jx, jy = latlon_to_local_xy_km(
		np.asarray([jma_lat], dtype=float),
		np.asarray([jma_lon], dtype=float),
		lat0_deg=grid.lat0_deg,
		lon0_deg=grid.lon0_deg,
	)

	# LOKI hypocenter (XY in km)
	if len(res.loc_rows) != 1:
		raise ValueError(
			f'.loc must have exactly 1 row for now: {res.loc_path} rows={len(res.loc_rows)}'
		)
	lr = res.loc_rows[0]
	lx = float(lr.x_km)
	ly = float(lr.y_km)

	fig, ax = plt.subplots(figsize=(9, 8))
	im = ax.imshow(
		coh_xy,
		origin='lower',
		extent=(x0, x1, y0, y1),
		aspect='equal',
	)
	cb = fig.colorbar(im, ax=ax)
	cb.set_label('Coherence (max over depth)')

	ax.scatter(
		sx,
		sy,
		s=26.0,
		marker='^',
		facecolors='white',
		edgecolors='black',
		linewidths=0.8,
		label='Stations',
		zorder=3,
	)
	ax.scatter(
		[lx],
		[ly],
		s=140.0,
		marker='X',
		c='red',
		edgecolors='black',
		linewidths=0.8,
		label='LOKI',
		zorder=5,
	)
	ax.scatter(
		[float(jx[0])],
		[float(jy[0])],
		s=220.0,
		marker='*',
		c='deepskyblue',
		edgecolors='black',
		linewidths=0.8,
		label='JMA',
		zorder=6,
	)

	# Station名ラベル（多すぎると潰れるので最大数で間引く）
	if show_station_labels:
		if name_col is None:
			raise ValueError(
				f'stations_df missing station name column; have={stations_df.columns.tolist()}'
			)
		names = stations_df[name_col].astype(str).to_numpy()
		nsta = len(names)
		step = max(1, int(np.ceil(nsta / max(1, int(station_label_max)))))
		for i in range(0, nsta, step):
			txt = ax.text(
				float(sx[i]) + float(station_label_offset_km),
				float(sy[i]) + float(station_label_offset_km),
				names[i],
				fontsize=int(station_label_fontsize),
				color='white',
				zorder=7,
			)
			# 白文字に黒フチを付けて、背景に負けないようにする
			txt.set_path_effects(
				[pe.Stroke(linewidth=1.6, foreground='black'), pe.Normal()]
			)

	ax.set_xlabel('X (km, east +)')
	ax.set_ylabel('Y (km, north +)')
	ax.set_title(f'event={event_id} coherence + hypocenters + stations')
	ax.legend(loc='upper right')

	if out_png is None:
		out_png = ev_out_dir / f'coherence_xy_overlay_trial{trial}.png'
	else:
		out_png = Path(out_png)
		if out_png.is_dir():
			out_png = out_png / f'coherence_xy_overlay_trial{trial}.png'

	return save_figure(fig, out_png, dpi=int(dpi))
