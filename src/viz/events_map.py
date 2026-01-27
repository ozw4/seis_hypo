# pasted.txt の plot_events_map_and_sections を「リンク線」対応に拡張した完全版
# - extras_lld: (lon,lat,depth) の点群を XY/XZ/YZ に重ねる
# - links_lld: 対応点ペアを線で結ぶ（XY/XZ/YZ すべて）
#
# 元の関数（extras_xyのみ）を置き換えてOK :contentReference[oaicite:0]{index=0}

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from shapely.geometry import Point

from viz.core.fig_io import save_figure


def plot_events_map_and_sections(
	df: pd.DataFrame,
	*,
	out_png: str | Path | None = None,
	mag_col: str | None = 'mag1',
	size_col: str | None = None,
	depth_col: str = 'depth_km',
	origin_time_col: str = 'origin_time',
	lat_col: str = 'latitude_deg',
	lon_col: str = 'longitude_deg',
	min_mag: float | None = None,
	max_mag: float | None = None,
	start_time: str | pd.Timestamp | None = None,
	end_time: str | pd.Timestamp | None = None,
	lon_range: tuple[float, float] | None = None,
	lat_range: tuple[float, float] | None = None,
	depth_range: tuple[float, float] | None = None,
	fontsize: int = 8,
	markersize: int = 8,
	extras_xy: list[dict] | None = None,
	extras_lld: list[dict] | None = None,
	links_lld: list[dict] | None = None,
	prefecture_shp: str
	| Path = '/workspace/data/N03-20240101_GML/N03-20240101_prefecture.shp',
) -> None:
	prefecture_shp = Path(prefecture_shp)

	required_cols = {lat_col, lon_col, depth_col, origin_time_col}
	if not required_cols.issubset(df.columns):
		missing = required_cols.difference(df.columns)
		raise ValueError(f'DataFrame に必要な列がありません: {missing}')

	df = df.copy()
	# ---- 時間フィルタ ----
	df[origin_time_col] = pd.to_datetime(df[origin_time_col])
	if start_time is not None:
		t0 = pd.to_datetime(start_time)
		df = df[df[origin_time_col] >= t0]
	if end_time is not None:
		t1 = pd.to_datetime(end_time)
		df = df[df[origin_time_col] <= t1]

	# ---- マグニチュードフィルタ ----
	if mag_col is not None and mag_col in df.columns:
		if min_mag is not None:
			df = df[df[mag_col] >= min_mag]
		if max_mag is not None:
			df = df[df[mag_col] <= max_mag]
	elif mag_col is not None and mag_col not in df.columns:
		raise ValueError(f'mag_col "{mag_col}" not found in DataFrame columns.')

	# ---- 必須列の NaN 除去 ----
	df = df.dropna(subset=[lat_col, lon_col, depth_col])
	if df.empty:
		raise RuntimeError('プロット可能なイベントがありません。')
	df = df.reset_index(drop=True)

	def _sizes_from_values(vals: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
		size_min = max(0.1, float(markersize) * 0.1)
		size_max = float(markersize) * 9.0

		out = np.empty_like(vals, dtype=float)
		mask = np.isfinite(vals)
		if vmax > vmin:
			v = np.clip(vals[mask], vmin, vmax)
			n = (v - vmin) / (vmax - vmin)
			out[mask] = size_min + n * (size_max - size_min)
		else:
			out[mask] = 0.5 * (size_min + size_max)
		out[~mask] = 0.5 * (size_min + size_max)
		return out

	color_col = mag_col
	size_col_eff = size_col if size_col is not None else mag_col

	has_color = (
		color_col is not None
		and color_col in df.columns
		and df[color_col].notna().any()
	)
	has_size = (
		size_col_eff is not None
		and size_col_eff in df.columns
		and df[size_col_eff].notna().any()
	)

	cmap = None
	norm = None
	sizes: np.ndarray | None = None

	color_vmin = -2.0
	color_vmax = 7.0
	vals_color: np.ndarray | None = None
	if has_color and color_col is not None:
		vals_color = df[color_col].astype(float).to_numpy()
		if color_col == 'cmax':
			color_vmin = df[color_col].min()
			color_vmax = df[color_col].max()
		cmap = plt.get_cmap('winter')
		norm = colors.Normalize(vmin=float(color_vmin), vmax=float(color_vmax))

	if has_size and size_col_eff is not None:
		vals_size = df[size_col_eff].astype(float).to_numpy()
		size_vmin = -2.0
		size_vmax = 7.0
		if size_col_eff == 'cmax':
			size_vmin = 0.0
			size_vmax = 1.0
		sizes = _sizes_from_values(vals_size, float(size_vmin), float(size_vmax))

	# ---- GeoDataFrame (XY 用) ----
	geometry = [
		Point(lon, lat) for lon, lat in zip(df[lon_col], df[lat_col], strict=True)
	]
	gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

	if len(lon_range) != 2:
		raise ValueError(
			'lon_range は (min_lon, max_lon) の長さ2タプルである必要があります。'
		)
	minx = float(lon_range[0])
	maxx = float(lon_range[1])
	if not np.isfinite(minx) or not np.isfinite(maxx):
		raise ValueError('lon_range の値は有限実数である必要があります。')
	if minx >= maxx:
		raise ValueError('lon_range は min_lon < max_lon を満たす必要があります。')

	if len(lat_range) != 2:
		raise ValueError(
			'lat_range は (min_lat, max_lat) の長さ2タプルである必要があります。'
		)
	miny = float(lat_range[0])
	maxy = float(lat_range[1])
	if not np.isfinite(miny) or not np.isfinite(maxy):
		raise ValueError('lat_range の値は有限実数である必要があります。')
	if miny >= maxy:
		raise ValueError('lat_range は min_lat < max_lat を満たす必要があります。')

	if depth_range is None:
		minz, maxz = 0.0, 400.0
	else:
		if len(depth_range) != 2:
			raise ValueError(
				'depth_range は (min_depth, max_depth) の長さ2タプルである必要があります。'
			)
		minz = float(depth_range[0])
		maxz = float(depth_range[1])
		if not np.isfinite(minz) or not np.isfinite(maxz):
			raise ValueError('depth_range の値は有限実数である必要があります。')
		if minz >= maxz:
			raise ValueError(
				'depth_range は min_depth < max_depth を満たす必要があります。'
			)

	# ---- extras_xy の整理(XY 用)----
	extras_xy_items: list[tuple[np.ndarray, np.ndarray, dict]] = []
	if extras_xy:
		for item in extras_xy:
			xy = item.get('xy', [])
			if not xy:
				continue
			lons = np.array([float(p[0]) for p in xy], float)
			lats = np.array([float(p[1]) for p in xy], float)
			style = {
				'label': item.get('label', 'Extra'),
				'marker': item.get('marker', 'o'),
				'color': item.get('color', 'red'),
				'size': item.get('size', 20.0),
				'annotate': bool(item.get('annotate', False)),
				'names': item.get('names'),
			}
			extras_xy_items.append((lons, lats, style))

	# ---- extras_lld の整理(XY/XZ/YZ 用)----
	extras_lld_items: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict]] = []
	if extras_lld:
		for item in extras_lld:
			lld = item.get('lld', [])
			if not lld:
				continue
			lons = np.array([float(p[0]) for p in lld], float)
			lats = np.array([float(p[1]) for p in lld], float)
			deps = np.array([float(p[2]) for p in lld], float)
			style = {
				'label': item.get('label', 'Extra'),
				'marker': item.get('marker', 'X'),
				'color': item.get('color', 'black'),
				'size': item.get('size', 40.0),
				'mag': item.get('mag'),
				'annotate': bool(item.get('annotate', False)),
				'names': item.get('names'),
			}
			extras_lld_items.append((lons, lats, deps, style))

	extras_handles: list = []
	extras_labels: list[str] = []

	# ---- Figure / Axes ----
	plt.rcParams['font.family'] = 'Arial'
	plt.rcParams.update({'font.size': fontsize, 'axes.linewidth': 0.5})

	fig = plt.figure(figsize=(10, 10))
	gs = fig.add_gridspec(
		2,
		2,
		width_ratios=(3.0, 1.5),
		height_ratios=(3.0, 1.5),
		wspace=0.1,
		hspace=0.1,
	)

	ax_xy = fig.add_subplot(gs[0, 0])
	ax_yz = fig.add_subplot(gs[0, 1])
	ax_xz = fig.add_subplot(gs[1, 0])
	ax_empty = fig.add_subplot(gs[1, 1])
	ax_empty.axis('off')

	# ---- XY(地図)----
	pref = gpd.read_file(prefecture_shp)
	if pref.crs is None or pref.crs.to_string().upper() != 'EPSG:4326':
		pref = pref.to_crs('EPSG:4326')

	pref.plot(
		ax=ax_xy,
		facecolor='whitesmoke',
		edgecolor='gray',
		linewidth=0.6,
		zorder=1,
		label='Pref.',
	)

	if has_color and cmap is not None and norm is not None and color_col is not None:
		gdf_mag = gdf[gdf[color_col].notna()].copy()
		size_xy = sizes[gdf_mag.index.to_numpy()] if sizes is not None else markersize
		gdf_mag.plot(
			ax=ax_xy,
			column=color_col,
			cmap=cmap,
			norm=norm,
			markersize=size_xy,
			marker='o',
			edgecolor='k',
			linewidth=0.1,
			alpha=0.5,
			zorder=2,
		)
	else:
		gdf.plot(
			ax=ax_xy,
			color='crimson',
			markersize=markersize,
			marker='o',
			edgecolor='k',
			linewidth=0.1,
			alpha=0.5,
			zorder=2,
			label='Event',
		)

	# ---- XY extras_xy ----
	for lons, lats, st in extras_xy_items:
		h_extra = ax_xy.scatter(
			lons,
			lats,
			marker=st['marker'],
			s=float(st['size']),
			c=st['color'],
			edgecolors='k',
			linewidths=0.4,
			alpha=0.9,
			zorder=3,
			label=st['label'],
		)
		extras_handles.append(h_extra)
		extras_labels.append(st['label'])

		if st['annotate']:
			names = st['names']
			if names is None:
				names = [f'{st["label"]}_{i + 1}' for i in range(len(lons))]
			for lon_v, lat_v, name in zip(lons, lats, names, strict=False):
				ax_xy.text(
					lon_v,
					lat_v,
					str(name),
					ha='left',
					va='bottom',
					fontsize=fontsize,
					color=st['color'],
					bbox=dict(facecolor='white', edgecolor='none', pad=0.4),
					zorder=4,
				)

	# ---- XY extras_lld（点）----
	for lons, lats, deps, st in extras_lld_items:
		s_val = None
		mag_vals = st.get('mag')
		if mag_vals is not None:
			m = np.asarray(mag_vals, dtype=float)
			if m.size != lons.size:
				raise ValueError('extras_lld mag length mismatch')
			s_val = _sizes_from_values(m, -2.0, 7.0)
		h_extra = ax_xy.scatter(
			lons,
			lats,
			marker=st['marker'],
			s=float(st['size']) if s_val is None else s_val,
			c=st['color'],
			edgecolors=None,
			linewidths=0.4,
			alpha=0.5,
			zorder=3,
			label=st['label'],
		)
		extras_handles.append(h_extra)
		extras_labels.append(st['label'])

		if st['annotate']:
			names = st['names']
			if names is None:
				names = [f'{st["label"]}_{i + 1}' for i in range(len(lons))]
			for lon_v, lat_v, name in zip(lons, lats, names, strict=False):
				ax_xy.text(
					lon_v,
					lat_v,
					str(name),
					ha='left',
					va='bottom',
					fontsize=fontsize,
					color=st['color'],
					bbox=dict(facecolor='white', edgecolor='none', pad=0.4),
					zorder=4,
				)

	ax_xy.set_ylabel('Latitude', fontsize=fontsize + 2)
	ax_xy.set_xlim(minx, maxx)
	ax_xy.set_ylim(miny, maxy)
	ax_xy.set_aspect('auto')

	# ---- XZ ----
	if has_color and cmap is not None and norm is not None and color_col is not None:
		ax_xz.scatter(
			df[lon_col],
			df[depth_col],
			c=df[color_col],
			s=markersize if sizes is None else sizes,
			cmap=cmap,
			norm=norm,
			edgecolors='k',
			linewidths=0.1,
			alpha=0.5,
		)
	else:
		ax_xz.scatter(
			df[lon_col],
			df[depth_col],
			c='crimson',
			s=markersize,
			edgecolors='k',
			linewidths=0.1,
			alpha=0.5,
		)

	for lons, lats, deps, st in extras_lld_items:
		s_val = None
		mag_vals = st.get('mag')
		if mag_vals is not None:
			m = np.asarray(mag_vals, dtype=float)
			if m.size != lons.size:
				raise ValueError('extras_lld mag length mismatch')
			s_val = _sizes_from_values(m, -2.0, 7.0)
		ax_xz.scatter(
			lons,
			deps,
			marker=st['marker'],
			s=float(st['size']) if s_val is None else s_val,
			c=st['color'],
			edgecolors=None,
			linewidths=0.4,
			alpha=0.5,
			zorder=3,
		)

	ax_xz.set_xlabel('Longitude', fontsize=fontsize + 2)
	ax_xz.set_ylabel('Depth (km)', fontsize=fontsize + 2)
	ax_xz.set_ylim(minz, maxz)
	ax_xz.invert_yaxis()

	# ---- YZ ----
	if has_color and cmap is not None and norm is not None and color_col is not None:
		ax_yz.scatter(
			df[depth_col],
			df[lat_col],
			c=df[color_col],
			s=markersize if sizes is None else sizes,
			cmap=cmap,
			norm=norm,
			edgecolors='k',
			linewidths=0.1,
			alpha=0.5,
		)
	else:
		ax_yz.scatter(
			df[depth_col],
			df[lat_col],
			c='crimson',
			s=markersize,
			edgecolors='k',
			linewidths=0.1,
			alpha=0.5,
		)

	for lons, lats, deps, st in extras_lld_items:
		s_val = None
		mag_vals = st.get('mag')
		if mag_vals is not None:
			m = np.asarray(mag_vals, dtype=float)
			if m.size != deps.size:
				raise ValueError('extras_lld mag length mismatch')
			s_val = _sizes_from_values(m, -2.0, 7.0)
		ax_yz.scatter(
			deps,
			lats,
			marker=st['marker'],
			s=float(st['size']) if s_val is None else s_val,
			c=st['color'],
			edgecolors=None,
			linewidths=0.4,
			alpha=0.5,
			zorder=3,
		)

	ax_yz.set_xlabel('Depth (km)', fontsize=fontsize + 2)
	ax_yz.set_xlim(minz, maxz)

	# ---- 軸範囲を揃える ----
	lon_min, lon_max = ax_xy.get_xlim()
	lat_min, lat_max = ax_xy.get_ylim()
	ax_xz.set_xlim(lon_min, lon_max)
	ax_yz.set_ylim(lat_min, lat_max)

	_xy_xlim = ax_xy.get_xlim()
	_xy_ylim = ax_xy.get_ylim()
	_xz_xlim = ax_xz.get_xlim()
	_xz_ylim = ax_xz.get_ylim()
	_yz_xlim = ax_yz.get_xlim()
	_yz_ylim = ax_yz.get_ylim()

	ax_xy.set_autoscale_on(False)
	ax_xz.set_autoscale_on(False)
	ax_yz.set_autoscale_on(False)

	if links_lld:
		for item in links_lld:
			pairs = item.get('pairs', [])
			if not pairs:
				continue

			color = item.get('color', 'black')
			lw = float(item.get('linewidth', 0.6))
			alpha = float(item.get('alpha', 0.35))
			label = item.get('label', None)

			first = True
			for (lon1, lat1, dep1), (lon2, lat2, dep2) in pairs:
				lbl = label if (label is not None and first) else None
				first = False

				ax_xy.plot(
					[float(lon1), float(lon2)],
					[float(lat1), float(lat2)],
					color=color,
					linewidth=lw,
					alpha=alpha,
					label=lbl,
					zorder=2.6,
				)
				ax_xz.plot(
					[float(lon1), float(lon2)],
					[float(dep1), float(dep2)],
					color=color,
					linewidth=lw,
					alpha=alpha,
					zorder=2.6,
				)
				ax_yz.plot(
					[float(dep1), float(dep2)],
					[float(lat1), float(lat2)],
					color=color,
					linewidth=lw,
					alpha=alpha,
					zorder=2.6,
				)

	# ★追加: 表示範囲を必ず元に戻す（これで“点が動く”見え方が消える）
	ax_xy.set_xlim(_xy_xlim)
	ax_xy.set_ylim(_xy_ylim)
	ax_xz.set_xlim(_xz_xlim)
	ax_xz.set_ylim(_xz_ylim)
	ax_yz.set_xlim(_yz_xlim)
	ax_yz.set_ylim(_yz_ylim)

	# ---- カラーバー + サイズ凡例 + extras/links 凡例 ----
	if has_color and cmap is not None and norm is not None:
		sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
		sm.set_array([])
		cbar = fig.colorbar(
			sm,
			ax=[ax_xy, ax_xz, ax_yz],
			shrink=0.7,
			orientation='horizontal',
			pad=0.07,
			fraction=0.02,
		)
		cbar.set_label(str(color_col))

	if sizes is not None and size_col_eff is not None:
		size_title = (
			'Magnitude (size)'
			if 'mag' in str(size_col_eff)
			else f'{size_col_eff} (size)'
		)

		mag_samples = (
			np.array([0.0, 3.0, 5.0, 7.0])
			if 'mag' in str(size_col_eff)
			else np.array([0.0, 0.3, 0.6, 1.0])
		)
		mag_vmin = -2.0 if 'mag' in str(size_col_eff) else 0.0
		mag_vmax = 7.0 if 'mag' in str(size_col_eff) else 1.0

		size_min = max(0.1, float(markersize) * 0.1)
		size_max = float(markersize) * 9.0

		m_clip_leg = np.clip(mag_samples, mag_vmin, mag_vmax)
		norm_mag_leg = (
			(m_clip_leg - mag_vmin) / (mag_vmax - mag_vmin)
			if mag_vmax > mag_vmin
			else np.zeros_like(m_clip_leg)
		)
		size_leg = size_min + norm_mag_leg * (size_max - size_min)

		handles = []
		for s_ in size_leg:
			h = ax_xy.scatter(
				[], [], s=s_, edgecolors='k', facecolors='none', linewidths=0.8
			)
			handles.append(h)
		labels = [f'M {m:g}' for m in mag_samples]

		all_handles = handles
		all_labels = labels
		if extras_handles:
			all_handles = all_handles + extras_handles
			all_labels = all_labels + extras_labels

		ax_xy.legend(
			all_handles,
			all_labels,
			title=size_title,
			loc='lower right',
			scatterpoints=1,
			fontsize=fontsize - 1,
			title_fontsize=fontsize - 1,
			framealpha=0.5,
		)
	elif extras_handles:
		ax_xy.legend(
			extras_handles,
			extras_labels,
			loc='lower right',
			scatterpoints=1,
			fontsize=fontsize - 1,
			framealpha=0.5,
		)

	# ---- タイトル ----
	bbox_xy = ax_xy.get_position()
	title_y = bbox_xy.y1 + 0.03
	if start_time is not None or end_time is not None:
		if start_time is not None and end_time is not None:
			title_time = f'{pd.to_datetime(start_time)} – {pd.to_datetime(end_time)}'
		elif start_time is not None:
			title_time = f'from {pd.to_datetime(start_time)}'
		else:
			title_time = f'until {pd.to_datetime(end_time)}'
		fig.suptitle(
			f'Earthquake Events {title_time}', fontsize=fontsize + 2, y=title_y
		)
	else:
		fig.suptitle('Earthquake Events', fontsize=fontsize + 2, y=title_y)

	fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.96], pad=0.2)

	if out_png is not None:
		out_png = Path(out_png)
		save_figure(fig, out_png, dpi=300, bbox_inches='tight', close=False)
	else:
		plt.show()

	plt.close(fig)
