# %%
# 依存: read_hinet_channel_table() を同一モジュール or import 済みとして使います
from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib import colors
from scipy.signal import detrend as sp_detrend
from scipy.signal.windows import tukey
from shapely.geometry import Point
from station_reader import read_hinet_channel_table


def plot_events_map_and_sections(
	df: pd.DataFrame,
	*,
	prefecture_shp: str | Path,
	out_png: str | Path | None = None,
	mag_col: str | None = 'mag1',
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

	# ---- mag -> 色 & サイズ（絶対スケール）----
	has_mag = (
		mag_col is not None and mag_col in df.columns and df[mag_col].notna().any()
	)
	cmap = None
	norm = None
	sizes: np.ndarray | None = None

	if has_mag:
		mag_series = df[mag_col].astype(float)
		mag_valid = mag_series.dropna().to_numpy()
		if mag_valid.size > 0:
			size_min = max(0.1, float(markersize) * 0.1)
			size_max = float(markersize) * 9.0

			mags = mag_series.to_numpy()
			sizes = np.empty_like(mags, dtype=float)
			mask_valid = np.isfinite(mags)

			mag_vmin = -2.0
			mag_vmax = 7.0

			if mag_vmax > mag_vmin:
				m_clip = np.clip(mags[mask_valid], mag_vmin, mag_vmax)
				norm_mag = (m_clip - mag_vmin) / (mag_vmax - mag_vmin)
				sizes[mask_valid] = size_min + norm_mag * (size_max - size_min)
			else:
				sizes[mask_valid] = 0.5 * (size_min + size_max)

			sizes[~mask_valid] = 0.5 * (size_min + size_max)

			cmap = plt.get_cmap('jet')
			norm = colors.Normalize(vmin=mag_vmin, vmax=mag_vmax)
		else:
			has_mag = False

	# ---- GeoDataFrame (XY 用) ----
	geometry = [
		Point(lon, lat) for lon, lat in zip(df[lon_col], df[lat_col], strict=True)
	]
	gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

	# ---- 表示範囲 ----
	if lon_range is None:
		minx, maxx = 118.0, 155.0
	else:
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

	if lat_range is None:
		miny, maxy = 22.0, 48.0
	else:
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

	# ---- extras_xy の整理（XY 用）----
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
				'marker': item.get('marker', 'X'),
				'color': item.get('color', 'black'),
				'size': item.get('size', 40.0),
				'annotate': bool(item.get('annotate', False)),
				'names': item.get('names'),
			}
			extras_xy_items.append((lons, lats, style))

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

	# ---- XY（地図）----
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

	if has_mag and cmap is not None and norm is not None and sizes is not None:
		gdf_mag = gdf[gdf[mag_col].notna()].copy()
		size_xy = sizes[gdf_mag.index.to_numpy()]
		gdf_mag.plot(
			ax=ax_xy,
			column=mag_col,
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

	# ---- XY extras ----
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

	ax_xy.set_ylabel('Latitude', fontsize=fontsize + 2)
	ax_xy.set_xlim(minx, maxx)
	ax_xy.set_ylim(miny, maxy)
	ax_xy.set_aspect('auto')

	# ---- XZ ----
	if has_mag and cmap is not None and norm is not None and sizes is not None:
		ax_xz.scatter(
			df[lon_col],
			df[depth_col],
			c=df[mag_col],
			s=sizes,
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
	ax_xz.set_xlabel('Longitude', fontsize=fontsize + 2)
	ax_xz.set_ylabel('Depth (km)', fontsize=fontsize + 2)
	ax_xz.set_ylim(minz, maxz)
	ax_xz.invert_yaxis()

	# ---- YZ ----
	if has_mag and cmap is not None and norm is not None and sizes is not None:
		ax_yz.scatter(
			df[depth_col],
			df[lat_col],
			c=df[mag_col],
			s=sizes,
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
	ax_yz.set_xlabel('Depth (km)', fontsize=fontsize + 2)
	ax_yz.set_xlim(minz, maxz)

	# ---- 軸範囲を揃える ----
	lon_min, lon_max = ax_xy.get_xlim()
	lat_min, lat_max = ax_xy.get_ylim()
	ax_xz.set_xlim(lon_min, lon_max)
	ax_yz.set_ylim(lat_min, lat_max)

	# ---- カラーバー + サイズ凡例 + extras 凡例 ----
	if has_mag and cmap is not None and norm is not None and sizes is not None:
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
		cbar.set_label('magnitude')

		mag_samples = np.array([0.0, 3.0, 5.0, 7.0])
		m_clip_leg = np.clip(mag_samples, mag_vmin, mag_vmax)
		norm_mag_leg = (m_clip_leg - mag_vmin) / (mag_vmax - mag_vmin)
		size_leg = size_min + norm_mag_leg * (size_max - size_min)

		handles = []
		for s_ in size_leg:
			h = ax_xy.scatter(
				[],
				[],
				s=s_,
				edgecolors='k',
				facecolors='none',
				linewidths=0.8,
			)
			handles.append(h)
		labels = [f'M {m:g}' for m in mag_samples]

		all_handles = handles
		all_labels = labels
		if extras_handles:
			all_handles = handles + extras_handles
			all_labels = labels + extras_labels

		ax_xy.legend(
			all_handles,
			all_labels,
			title='Magnitude (size)',
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
			f'Earthquake Events {title_time}',
			fontsize=fontsize + 2,
			y=title_y,
		)
	else:
		fig.suptitle(
			'Earthquake Events',
			fontsize=fontsize + 2,
			y=title_y,
		)

	fig.tight_layout(
		rect=[0.02, 0.02, 0.98, 0.96],
		pad=0.2,
	)
	if out_png is not None:
		fig.savefig(out_png, dpi=300, bbox_inches='tight')


def plot_stations_from_hinet_table(
	station_names: list[str],
	*,
	prefecture_shp: str | Path,  # 例: "N03-20240101_GML/N03-20240101_prefecture.shp"
	out_png: str | Path = 'Figure_Stations.png',
	marker: str = '^',
	markersize: int = 24,
	fontsize: int = 8,
	hinet_table_path: str
	| Path = '/workspace/proc/util/hinet_util/hinet_channelstbl_20251007',
	# 追加: 任意ポイント群（カテゴリ単位で凡例を出す）
	extras: list[dict]
	| None = None,  # 例: [{"label":"Well A","xy":[(138.1,36.2)],"marker":"o","color":"royalblue","size":30,"annotate":True}]
	# 追加: 駅ラベルのオフセット
	label_dlat: float = 0.03,  # 緯度方向オフセット（度）
) -> None:
	"""Hi-net チャネル表から station_names を抽出して日本地図上に描画。
	extras で任意の点座標をカテゴリ別に凡例付きで重ね描きできる。

	Parameters
	----------
	station_names : List[str]
		プロットする局名（チャネル表の 'station' 列値）
	prefecture_shp : Path
		都道府県ポリゴンのシェープファイルパス
	out_png : Path
		出力ファイル名（PNG）
	marker, markersize, fontsize : 見た目
	hinet_table_path : Path
		read_hinet_channel_table() に渡すチャネル表
	extras : List[dict] | None
		任意ポイントのリスト。各 dict は以下のキーを推奨:
		  - "label": 凡例ラベル（必須）
		  - "xy": Iterable[tuple[lon, lat]]（必須；WGS84度）
		  - "marker": matplotlib marker（既定 "o"）
		  - "color":  色名（既定 "tab:blue"）
		  - "size":   散布サイズ（既定 30）
		  - "annotate": bool（点名注記；既定 False）
		  - "names":   Iterable[str]（annotate=True のときに各点に付す名前）
	label_dlat : float
		局ラベルの緯度方向オフセット（度）

	"""
	df = read_hinet_channel_table(hinet_table_path)
	df = df[df['station'].isin(station_names)].copy()
	if df.empty:
		raise RuntimeError('指定ステーションがチャンネル表に見つかりません。')

	# --- GeoDataFrame 化（WGS84） ---
	df['geometry'] = [
		Point(lon, lat) for lon, lat in zip(df['lon'], df['lat'], strict=False)
	]
	gdf = gpd.GeoDataFrame(
		df.drop_duplicates(subset=['station']), geometry='geometry', crs='EPSG:4326'
	)

	# --- 表示範囲（ステーション + extras を考慮） ---
	xs = list(gdf.geometry.x)
	ys = list(gdf.geometry.y)

	extras_gdfs: list[gpd.GeoDataFrame] = []
	if extras:
		for item in extras:
			xy: Iterable[tuple[float, float]] = item.get('xy', [])
			if not xy:
				continue
			ex_df = pd.DataFrame(xy, columns=['lon', 'lat'])
			ex_df['geometry'] = [Point(lon, lat) for lon, lat in xy]
			ex_gdf = gpd.GeoDataFrame(ex_df, geometry='geometry', crs='EPSG:4326')
			ex_gdf.attrs['style'] = {
				'label': item.get('label', 'Extra'),
				'marker': item.get('marker', 'o'),
				'color': item.get('color', 'tab:blue'),
				'size': item.get('size', 30),
				'annotate': bool(item.get('annotate', False)),
				'names': item.get('names'),  # None or list[str]
			}
			extras_gdfs.append(ex_gdf)
			xs.extend(ex_gdf.geometry.x.to_list())
			ys.extend(ex_gdf.geometry.y.to_list())

	if not xs or not ys:
		raise RuntimeError('プロット対象がありません。')

	minx, maxx = min(xs), max(xs)
	miny, maxy = min(ys), max(ys)
	pad_x = max(0.5, (maxx - minx) * 0.15)
	pad_y = max(0.5, (maxy - miny) * 0.15)

	# --- 描画 ---
	plt.rcParams['font.family'] = 'Arial'
	plt.rcParams.update({'font.size': fontsize, 'axes.linewidth': 0.5})

	fig, ax = plt.subplots(figsize=(6, 6))

	pref = gpd.read_file(prefecture_shp)
	if pref.crs is None or pref.crs.to_string().upper() != 'EPSG:4326':
		pref = pref.to_crs('EPSG:4326')

	pref.plot(
		ax=ax,
		facecolor='whitesmoke',
		edgecolor='gray',
		linewidth=0.6,
		zorder=1,
		label='Pref.',
	)

	# extras（カテゴリ毎に凡例）
	for ex_gdf in extras_gdfs:
		st = ex_gdf.attrs['style']
		ex_gdf.plot(
			ax=ax,
			color=st['color'],
			marker=st['marker'],
			markersize=st['size'],
			zorder=3,
			label=st['label'],
		)
		if st['annotate']:
			names = st['names']
			if names is None:
				# 名前未指定なら 1..N を付番
				names = [f'{st["label"]}_{i + 1}' for i in range(len(ex_gdf))]
			for (x0, y0), name in zip(
				ex_gdf[['lon', 'lat']].to_numpy(), names, strict=False
			):
				ax.text(
					x0,
					y0 + label_dlat,
					str(name),
					ha='center',
					va='bottom',
					fontsize=fontsize,
					color='black',
					bbox=dict(facecolor='white', edgecolor='none', pad=0.6),
					zorder=5,
				)

	# ステーション
	gdf.plot(
		ax=ax,
		color='crimson',
		marker=marker,
		markersize=markersize,
		zorder=4,
		label='Station',
	)
	texts = []
	for x0, y0, name in zip(
		gdf.geometry.x, gdf.geometry.y, gdf['station'], strict=False
	):
		t = ax.text(
			x0,
			y0 - label_dlat,
			name,
			ha='center',
			va='top',
			fontsize=fontsize,
			bbox=dict(facecolor='white', edgecolor='none', pad=0.6),
			zorder=3,
		)
		texts.append(t)

	xs_obs = []  # 障害物 points: ステーション点 + extras 点（全部）
	ys_obs = []

	# ステーション点を障害物に加える
	xs_obs.extend(gdf.geometry.x.to_list())
	ys_obs.extend(gdf.geometry.y.to_list())

	# extras の点も障害物に加える（凡例カテゴリごと）
	for ex_gdf in extras_gdfs:
		xs_obs.extend(ex_gdf.geometry.x.to_list())
		ys_obs.extend(ex_gdf.geometry.y.to_list())

	ax.set_xlabel('Longitude')
	ax.set_ylabel('Latitude')
	ax.set_xlim(minx - pad_x, maxx + pad_x)
	ax.set_ylim(miny - pad_y, maxy + pad_y)
	plt.tight_layout()
	ax.legend(loc='lower right', fontsize=fontsize)

	adjust_text(
		texts,
		x=xs_obs,
		y=ys_obs,
		ax=ax,
		expand_text=(1.05, 1.2),
		expand_points=(1.05, 1.2),
		# arrowprops=dict(arrowstyle='-', lw=0.5, color='0.3', alpha=0.7),
	)

	fig.savefig(out_png, dpi=200)
	print(f'Saved: {out_png}')


def _latlon_to_xy(lat, lon):
	lat = np.asarray(lat, float)
	lon = np.asarray(lon, float)
	R = 6371000.0
	lat0 = np.deg2rad(np.nanmean(lat))
	x = R * np.deg2rad(lon - np.nanmean(lon)) * np.cos(lat0)
	y = R * np.deg2rad(lat - np.nanmean(lat))
	return x, y  # x: East, y: North


# ---- 並び順 ----
def compute_station_order(
	station_df: pd.DataFrame, mode: str = 'pca', azimuth_deg: float | None = None
) -> np.ndarray:
	lat = station_df['lat'].to_numpy(float)
	lon = station_df['lon'].to_numpy(float)
	x, y = _latlon_to_xy(lat, lon)

	if mode == 'lat':
		order = np.argsort(y)
	elif mode == 'lon':
		order = np.argsort(x)
	elif mode == 'azimuth':
		if azimuth_deg is None:
			raise ValueError(
				"mode='azimuth' では azimuth_deg を指定してください（0°=北, 90°=東）"
			)
		th = np.deg2rad(azimuth_deg)
		ux, uy = np.sin(th), np.cos(th)
		s = x * ux + y * uy
		order = np.argsort(s)
	elif mode == 'pca':
		XY = np.column_stack([x - x.mean(), y - y.mean()])
		_, _, Vt = np.linalg.svd(XY, full_matrices=False)
		v = Vt[0]
		s = XY @ v
		order = np.argsort(s)
	else:
		raise ValueError(f'unknown mode: {mode}')
	return order


# ---- ギャザー描画（塗りつぶしwiggle）----
def plot_gather(
	data: np.ndarray,
	station_df: pd.DataFrame | None = None,
	scaling: str = 'zscore',
	amp: float = 4.0,
	title: str | None = None,
	p_idx: np.ndarray | None = None,
	s_idx: np.ndarray | None = None,
	order_mode: str = 'pca',
	azimuth_deg: float | None = None,
	ax: plt.Axes | None = None,
	decim: int = 1,
	detrend: str | None = None,  # 'constant' | 'linear' | None
	taper_frac: float = 0.02,  # 端部ターパー（片側比）
	y_time: str = 'samples',  # ← 'samples' | 'absolute' | 'relative'
	fs: float | None = None,  # ← y_time≠'samples'なら必須(Hz)
	t_start: dt.datetime | None = None,  # ← 窓の開始(絶対時刻, JST想定)
	event_time: dt.datetime | None = None,  # ← 相対表示で0にしたい時刻
):
	assert data.ndim == 2
	n_ch, n_t = data.shape

	# --- 並び替え ---
	if station_df is not None and {'station', 'lat', 'lon'}.issubset(
		station_df.columns
	):
		order = compute_station_order(
			station_df.iloc[:n_ch], mode=order_mode, azimuth_deg=azimuth_deg
		)
		data = data[order]
		station_df = station_df.iloc[order].reset_index(drop=True)
		if p_idx is not None:
			p_idx = np.asarray(p_idx)[order]
		if s_idx is not None:
			s_idx = np.asarray(s_idx)[order]

	n_ch, n_t = data.shape  # 念のため更新
	x = data.astype(float, copy=False)

	# --- Detrend → Taper ---
	if detrend in ('constant', 'linear'):
		x = sp_detrend(x, axis=1, type=detrend)
	elif detrend is not None:
		raise ValueError("detrend must be 'constant'|'linear'|None")

	if taper_frac > 0.0:
		w = tukey(n_t, alpha=2 * taper_frac)
		x *= w

	# --- 正規化 ---
	if scaling == 'zscore':
		m = x.mean(axis=1, keepdims=True)
		s = x.std(axis=1, keepdims=True) + 1e-12
		x = (x - m) / s
	elif scaling == 'max':
		m = np.max(np.abs(x), axis=1, keepdims=True) + 1e-12
		x = x / m
	elif scaling == 'none':
		pass
	else:
		raise ValueError(f'unknown scaling: {scaling}')

	# --- 間引き（描画用） ---
	if decim > 1:
		x = x[:, ::decim]
	n_t_dec = x.shape[1]
	y = np.arange(n_t_dec) * decim  # 縦軸は元のサンプル番号で揃える

	# --- 横配置（ampは横振れ幅スケール） ---
	x_for_layout = (
		x
		if scaling == 'none'
		else x / (np.max(np.abs(x), axis=1, keepdims=True) + 1e-12)
	)
	centers = np.arange(n_ch)[:, None]
	xs = centers + amp * x_for_layout

	# --- 描画：塗りつぶしwiggle（正側だけ塗る） ---
	if ax is None:
		fig, ax = plt.subplots(figsize=(max(8.0, 0.12 * n_ch), 8))

	for i in range(n_ch):
		xi = xs[i]
		base = float(i)
		ax.plot(xi, y, lw=0.5, c='k', zorder=2)  # 輪郭線
		ax.fill_betweenx(
			y, base, xi, where=(xi >= base), linewidth=0, alpha=0.7, zorder=1, color='k'
		)

	# 軸と範囲
	ax.set_xlim(float(xs.min()) - 0.1, float(xs.max()) + 0.1)
	ax.set_ylim(y[-1], y[0] if len(y) > 0 else 0)
	ax.set_xlabel('station')
	if y_time == 'samples':
		ax.set_ylabel('sample')
	else:
		if fs is None or t_start is None:
			raise ValueError("y_time≠'samples' の場合は fs と t_start が必要です")
		total_s = (n_t_dec - 1) * (decim / fs)
		# 目盛り間隔（だいたい5〜8本出るように）
		cand = np.array([0.5, 1, 2, 5, 10, 20, 30], float)
		step_s = cand[np.argmin(np.abs(total_s / cand - 6))]
		t_grid = np.arange(0.0, total_s + 1e-9, step_s)
		yticks = (t_grid * fs / decim).astype(int)
		if y_time == 'absolute':
			labels = [
				(t_start + dt.timedelta(seconds=float(ts))).strftime('%H:%M:%S.%f')[:-3]
				for ts in t_grid
			]
			ax.set_ylabel('time (JST)')
		elif y_time == 'relative':
			if event_time is None:
				raise ValueError("y_time='relative' には event_time が必要です")
			offset = (event_time - t_start).total_seconds()
			labels = [f'{(ts - offset):+0.1f}s' for ts in t_grid]
			ax.set_ylabel('time from event')
		else:
			raise ValueError("y_time は 'samples' | 'absolute' | 'relative'")
		ax.set_yticks(yticks)
		ax.set_yticklabels(labels)
	ax.set_xticks(np.arange(n_ch))
	if station_df is not None and 'station' in station_df.columns:
		ax.set_xticklabels(station_df['station'].to_numpy()[:n_ch], rotation=90)
	else:
		ax.set_xticklabels([str(i) for i in range(n_ch)])
	if title:
		ax.set_title(title)

	# Picks
	if p_idx is not None:
		m = np.isfinite(p_idx)
		ax.scatter(
			np.arange(n_ch)[m],
			np.asarray(p_idx)[m],
			s=20,
			marker='_',
			c='b',
			linewidths=1,
		)
	if s_idx is not None:
		m = np.isfinite(s_idx)
		ax.scatter(
			np.arange(n_ch)[m],
			np.asarray(s_idx)[m],
			s=20,
			marker='_',
			c='r',
			linewidths=1,
		)

	plt.tight_layout()
	return ax


if __name__ == '__main__':
	from station_reader import stations_within_radius

	well_coord = (35.511111, 140.1925)  # (lat, lon)
	station_list = stations_within_radius(
		lat=well_coord[0], lon=well_coord[1], radius_km=50.0
	)
	extras = [
		{
			'label': 'mobara site',
			'xy': [(well_coord[1], well_coord[0])],
			'marker': 'o',
			'color': 'royalblue',
			'size': 30,
			'annotate': False,
		}
	]
	plot_stations_from_hinet_table(
		station_names=station_list,
		prefecture_shp='N03-20240101_GML/N03-20240101_prefecture.shp',
		out_png='Figure_Stations.png',
		extras=extras,
	)

# %%
