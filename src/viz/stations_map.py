# %%
# 依存: read_hinet_channel_table() を同一モジュール or import 済みとして使います
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from shapely.geometry import Point

from jma.station_reader import read_hinet_channel_table


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
	# 追加: 任意ポイント群(カテゴリ単位で凡例を出す)
	extras: list[dict]
	| None = None,  # 例: [{"label":"Well A","xy":[(138.1,36.2)],"marker":"o","color":"royalblue","size":30,"annotate":True}]
	# 追加: 駅ラベルのオフセット
	label_dlat: float = 0.03,  # 緯度方向オフセット(度)
) -> None:
	"""Hi-net チャネル表から station_names を抽出して日本地図上に描画.
	extras で任意の点座標をカテゴリ別に凡例付きで重ね描きできる。

	Parameters
	----------
	station_names : List[str]
		プロットする局名(チャネル表の 'station' 列値)
	prefecture_shp : Path
		都道府県ポリゴンのシェープファイルパス
	out_png : Path
		出力ファイル名(PNG)
	marker, markersize, fontsize : 見た目
	hinet_table_path : Path
		read_hinet_channel_table() に渡すチャネル表
	extras : List[dict] | None
		任意ポイントのリスト。各 dict は以下のキーを推奨:
		- "label": 凡例ラベル(必須)
		- "xy": Iterable[tuple[lon, lat]](必須:WGS84度)
		- "marker": matplotlib marker(既定 "o")
		- "color":  色名(既定 "tab:blue")
		- "size":   散布サイズ(既定 30)
		- "annotate": bool(点名注記:既定 False)
		- "names":   Iterable[str](annotate=True のときに各点に付す名前)
	label_dlat : float
		局ラベルの緯度方向オフセット(度)

	"""
	df = read_hinet_channel_table(hinet_table_path)
	df = df[df['station'].isin(station_names)].copy()
	if df.empty:
		raise RuntimeError('指定ステーションがチャンネル表に見つかりません。')

	# --- GeoDataFrame 化(WGS84) ---
	df['geometry'] = [
		Point(lon, lat) for lon, lat in zip(df['lon'], df['lat'], strict=False)
	]
	gdf = gpd.GeoDataFrame(
		df.drop_duplicates(subset=['station']), geometry='geometry', crs='EPSG:4326'
	)

	# --- 表示範囲(ステーション + extras を考慮) ---
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

	# extras(カテゴリ毎に凡例)
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

	xs_obs = []  # 障害物 points: ステーション点 + extras 点(全部)
	ys_obs = []

	# ステーション点を障害物に加える
	xs_obs.extend(gdf.geometry.x.to_list())
	ys_obs.extend(gdf.geometry.y.to_list())

	# extras の点も障害物に加える(凡例カテゴリごと)
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
