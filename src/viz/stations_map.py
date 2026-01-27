# %%
# 依存: read_hinet_channel_table() を同一モジュール or import 済みとして使います
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text
from shapely.geometry import Point

from viz.core.fig_io import save_figure

from jma.station_affiliation import _comment_to_affiliation_en
from jma.station_reader import read_hinet_channel_table


def plot_stations_by_affiliation_from_station_csv(
	station_csv: str | Path,
	*,
	prefecture_shp: str | Path,
	out_png: str | Path = 'Figure_Stations_by_affiliation.png',
	station_codes: Iterable[str] | None = None,
	marker: str = '^',
	markersize: int = 24,
	fontsize: int = 8,
	extras: list[dict] | None = None,
	label_dlat: float = 0.03,
	show_station_labels: bool = True,
	cmap_name: str = 'tab20',
	affiliation_colors: dict[str, str] | None = None,
	affiliation_filter: Iterable[str] | None = None,
) -> None:
	"""station.csv から観測点を読み込み、所属ごとに色を変えてプロットする。
	地方区分はすべて "JMA" に統一。
	表示範囲は日本（経度 128–151, 緯度 30–46）の範囲に限定する。

	Parameters
	----------
	station_csv : Path
		station.csv のパス（列: station_code, Latitude_deg, Longitude_deg, Comment）。
	prefecture_shp : Path
		都道府県ポリゴンのシェープファイルパス（EPSG:4326 または変換可能な CRS）。
	out_png : Path
		出力 PNG ファイル名。
	station_codes : Iterable[str] | None
		プロット対象とする station_code。None の場合は全局。
	marker, markersize, fontsize :
		Matplotlib の見た目設定。
	extras : list[dict] | None
		任意ポイント（井戸など）のリスト。
	label_dlat : float
		局ラベルの緯度方向オフセット（度）。
	show_station_labels : bool
		True なら station_code を地図上に表示。
	cmap_name : str
		Colormap 名（例: 'tab20', 'tab10', 'Set2', 'Dark2', 'Paired' など）。
	affiliation_colors : dict[str, str] | None
		所属ごとの色を固定するマップ（{'NIED': 'tab:orange', ...}）。
		指定された場合はこちらを優先し、cmap_name はベース色としてのみ使用される。
	affiliation_filter : Iterable[str] | None
		この affiliation_en のみプロットしたいときに指定（例: ['NIED', 'University']）。
		None の場合は全ての affiliation_en を描画する。

	"""
	station_csv = Path(station_csv)
	prefecture_shp = Path(prefecture_shp)
	out_png = Path(out_png)

	if not station_csv.is_file():
		raise FileNotFoundError(f'station_csv not found: {station_csv}')
	if not prefecture_shp.is_file():
		raise FileNotFoundError(f'prefecture_shp not found: {prefecture_shp}')

	df = pd.read_csv(station_csv)

	required_cols = {'station_code', 'Latitude_deg', 'Longitude_deg', 'Comment'}
	missing = required_cols.difference(df.columns)
	if missing:
		raise ValueError(f'station.csv missing required columns: {sorted(missing)}')

	if station_codes is not None:
		station_codes_set = set(station_codes)
		df = df[df['station_code'].isin(station_codes_set)].copy()

	if df.empty:
		raise RuntimeError('No stations to plot (check station_codes / CSV contents).')

	df['lat'] = df['Latitude_deg'].astype(float)
	df['lon'] = df['Longitude_deg'].astype(float)
	df['affiliation_en'] = df['Comment'].map(_comment_to_affiliation_en)

	# ---- affiliation でフィルタ ----
	if affiliation_filter is not None:
		aff_set = set(affiliation_filter)
		df = df[df['affiliation_en'].isin(aff_set)].copy()
		if df.empty:
			raise RuntimeError('No stations to plot for specified affiliations.')

	df['geometry'] = [
		Point(lon, lat) for lon, lat in zip(df['lon'], df['lat'], strict=False)
	]
	gdf = gpd.GeoDataFrame(
		df.drop_duplicates(subset=['station_code']),
		geometry='geometry',
		crs='EPSG:4326',
	)

	extras_gdfs: list[gpd.GeoDataFrame] = []
	if extras:
		for item in extras:
			xy = item.get('xy', [])
			if not xy:
				continue
			ex_df = pd.DataFrame(xy, columns=['lon', 'lat'])
			ex_df['geometry'] = [
				Point(lon, lat)
				for lon, lat in zip(ex_df['lon'], ex_df['lat'], strict=False)
			]
			ex_gdf = gpd.GeoDataFrame(ex_df, geometry='geometry', crs='EPSG:4326')
			ex_gdf.attrs['style'] = {
				'label': item.get('label', 'Extra'),
				'marker': item.get('marker', 'o'),
				'color': item.get('color', 'tab:blue'),
				'size': item.get('size', 30),
				'annotate': bool(item.get('annotate', False)),
				'names': item.get('names'),
			}
			extras_gdfs.append(ex_gdf)

	plt.rcParams['font.family'] = 'Arial'
	plt.rcParams.update({'font.size': fontsize, 'axes.linewidth': 0.5})

	fig, ax = plt.subplots(figsize=(10, 10))

	pref = gpd.read_file(prefecture_shp)
	if pref.crs is None or pref.crs.to_string().upper() != 'EPSG:4326':
		pref = pref.to_crs('EPSG:4326')

	pref.plot(
		ax=ax,
		facecolor='whitesmoke',
		edgecolor='gray',
		linewidth=0.6,
		zorder=1,
		label='Prefecture',
	)

	for ex_gdf in extras_gdfs:
		st = ex_gdf.attrs['style']
		ex_gdf.plot(
			ax=ax,
			color=st['color'],
			marker=st['marker'],
			markersize=st['size'],
			zorder=5,
			label=st['label'],
		)
		if st['annotate']:
			names = st['names']
			if names is None:
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
					zorder=6,
				)

	# ---- 所属ごとの色を決定 ----
	affiliations = sorted(gdf['affiliation_en'].unique())

	# ベース色: colormap から決める
	cmap = plt.get_cmap(cmap_name)
	n_colors = getattr(cmap, 'N', 256)
	base_color_map = {
		aff: cmap(i / max(n_colors - 1, 1)) for i, aff in enumerate(affiliations)
	}

	# affiliation_colors で指定されたものだけ上書き
	if affiliation_colors is None:
		color_map = base_color_map
	else:
		color_map: dict[str, object] = {}
		for aff in affiliations:
			# 手動指定があればそれを使い、なければ colormap 由来の色を使う
			if aff in affiliation_colors:
				color_map[aff] = affiliation_colors[aff]
			else:
				color_map[aff] = base_color_map[aff]

	for aff in affiliations:
		sub = gdf[gdf['affiliation_en'] == aff]
		if sub.empty:
			continue
		sub.plot(
			ax=ax,
			color=color_map[aff],
			marker=marker,
			markersize=markersize,
			zorder=4,
			label=aff,
		)

	texts = []
	if show_station_labels:
		for x0, y0, code in zip(
			gdf.geometry.x, gdf.geometry.y, gdf['station_code'], strict=False
		):
			t = ax.text(
				x0,
				y0 - label_dlat,
				str(code),
				ha='center',
				va='top',
				fontsize=fontsize,
				bbox=dict(facecolor='white', edgecolor='none', pad=0.6),
				zorder=3,
			)
			texts.append(t)

	xs_obs = gdf.geometry.x.to_list()
	ys_obs = gdf.geometry.y.to_list()
	for ex_gdf in extras_gdfs:
		xs_obs.extend(ex_gdf.geometry.x.to_list())
		ys_obs.extend(ex_gdf.geometry.y.to_list())

	ax.set_xlabel('Longitude')
	ax.set_ylabel('Latitude')

	# 日本全体が入るように、固定範囲を使用（IRIS など遠方局に引っ張られない）
	minx, miny, maxx, maxy = 128, 30, 151, 46
	ax.set_xlim(minx, maxx)
	ax.set_ylim(miny, maxy)

	handles, labels = ax.get_legend_handles_labels()
	uniq: dict[str, object] = {}
	for h, l in zip(handles, labels, strict=False):
		if l not in uniq:
			uniq[l] = h
	ax.legend(
		uniq.values(),
		uniq.keys(),
		loc='lower right',
		fontsize=fontsize,
		title='Affiliation',
		title_fontsize=fontsize,
	)

	plt.tight_layout()

	if texts:
		adjust_text(
			texts,
			x=xs_obs,
			y=ys_obs,
			ax=ax,
			expand_text=(1.05, 1.2),
			expand_points=(1.05, 1.2),
		)

	out_png = save_figure(fig, out_png, dpi=200)
	print(f'Saved: {out_png}')


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

	out_png = save_figure(fig, out_png, dpi=200)
	print(f'Saved: {out_png}')


def _normalize_comment_local(s: object) -> str:
	if pd.isna(s):
		return 'Unknown'
	return ' '.join(str(s).split())


def _sanitize_filename(s: str) -> str:
	bad = '\\/:*?"<>|'
	for ch in bad:
		s = s.replace(ch, '_')
	s = s.replace('\n', ' ').replace('\r', ' ')
	s = ' '.join(s.split())
	return s if s else 'Unknown'


def plot_stations_by_original_affiliation_from_station_csv(
	station_csv: str | Path,
	*,
	prefecture_shp: str | Path,
	out_dir: str | Path = 'fig_affiliations',
	out_png_template: str = 'Figure_Stations_affiliation_{affiliation}.png',
	station_codes: Iterable[str] | None = None,
	affiliation_comments: Iterable[str] | None = None,
	exclude_jma: bool = False,
	marker: str = '^',
	markersize: int = 24,
	fontsize: int = 8,
	extras: list[dict] | None = None,
	label_dlat: float = 0.03,
	show_station_labels: bool = True,
	fixed_extent: bool = True,
) -> None:
	station_csv = Path(station_csv)
	prefecture_shp = Path(prefecture_shp)
	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	if not station_csv.is_file():
		raise FileNotFoundError(f'station_csv not found: {station_csv}')
	if not prefecture_shp.is_file():
		raise FileNotFoundError(f'prefecture_shp not found: {prefecture_shp}')

	df = pd.read_csv(station_csv)

	required_cols = {'station_code', 'Latitude_deg', 'Longitude_deg', 'Comment'}
	missing = required_cols.difference(df.columns)
	if missing:
		raise ValueError(f'station.csv missing required columns: {sorted(missing)}')

	if station_codes is not None:
		st_set = set(station_codes)
		df = df[df['station_code'].isin(st_set)].copy()

	if df.empty:
		raise RuntimeError('No stations to plot (check station_codes / CSV contents).')

	df['lat'] = df['Latitude_deg'].astype(float)
	df['lon'] = df['Longitude_deg'].astype(float)
	df['affiliation_raw'] = df['Comment'].map(_normalize_comment_local)

	# JMA（地方区分）と JMA Intensity を除外（出力はあくまで affiliation_raw で分割）
	if exclude_jma:
		aff_cat = df['Comment'].map(_comment_to_affiliation_en)
		df = df[~aff_cat.isin({'JMA', 'JMA Intensity'})].copy()
		if df.empty:
			raise RuntimeError('No stations to plot after excluding JMA.')

	if affiliation_comments is not None:
		want = {_normalize_comment_local(s) for s in affiliation_comments}
		df = df[df['affiliation_raw'].isin(want)].copy()
		if df.empty:
			raise RuntimeError(
				'No stations to plot for specified affiliation_comments.'
			)

	df['geometry'] = [
		Point(lon, lat) for lon, lat in zip(df['lon'], df['lat'], strict=False)
	]
	gdf_all = gpd.GeoDataFrame(
		df.drop_duplicates(subset=['station_code']),
		geometry='geometry',
		crs='EPSG:4326',
	)

	pref = gpd.read_file(prefecture_shp)
	if pref.crs is None or pref.crs.to_string().upper() != 'EPSG:4326':
		pref = pref.to_crs('EPSG:4326')

	extras_gdfs: list[gpd.GeoDataFrame] = []
	if extras:
		for item in extras:
			xy = item.get('xy', [])
			if not xy:
				continue
			ex_df = pd.DataFrame(xy, columns=['lon', 'lat'])
			ex_df['geometry'] = [
				Point(lon, lat)
				for lon, lat in zip(ex_df['lon'], ex_df['lat'], strict=False)
			]
			ex_gdf = gpd.GeoDataFrame(ex_df, geometry='geometry', crs='EPSG:4326')
			ex_gdf.attrs['style'] = {
				'label': item.get('label', 'Extra'),
				'marker': item.get('marker', 'o'),
				'color': item.get('color', 'tab:blue'),
				'size': item.get('size', 30),
				'annotate': bool(item.get('annotate', False)),
				'names': item.get('names'),
			}
			extras_gdfs.append(ex_gdf)

	plt.rcParams['font.family'] = 'Arial'
	plt.rcParams.update({'font.size': fontsize, 'axes.linewidth': 0.5})

	affiliations = sorted(gdf_all['affiliation_raw'].unique())

	for aff in affiliations:
		sub = gdf_all[gdf_all['affiliation_raw'] == aff]
		if sub.empty:
			continue

		fig, ax = plt.subplots(figsize=(10, 10))

		pref.plot(
			ax=ax,
			facecolor='whitesmoke',
			edgecolor='gray',
			linewidth=0.6,
			zorder=1,
			label='Prefecture',
		)

		for ex_gdf in extras_gdfs:
			st = ex_gdf.attrs['style']
			ex_gdf.plot(
				ax=ax,
				color=st['color'],
				marker=st['marker'],
				markersize=st['size'],
				zorder=5,
				label=st['label'],
			)
			if st['annotate']:
				names = st['names']
				if names is None:
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
						zorder=6,
					)

		sub.plot(
			ax=ax,
			color='tab:red',
			marker=marker,
			markersize=markersize,
			zorder=4,
			label='Station',
		)

		texts = []
		if show_station_labels:
			for x0, y0, code in zip(
				sub.geometry.x, sub.geometry.y, sub['station_code'], strict=False
			):
				texts.append(
					ax.text(
						x0,
						y0 - label_dlat,
						str(code),
						ha='center',
						va='top',
						fontsize=fontsize,
						bbox=dict(facecolor='white', edgecolor='none', pad=0.6),
						zorder=3,
					)
				)

		ax.set_xlabel('Longitude')
		ax.set_ylabel('Latitude')
		ax.set_title(aff)

		if fixed_extent:
			ax.set_xlim(128, 151)
			ax.set_ylim(30, 46)
		else:
			xs = sub.geometry.x.to_list()
			ys = sub.geometry.y.to_list()
			for ex_gdf in extras_gdfs:
				xs.extend(ex_gdf.geometry.x.to_list())
				ys.extend(ex_gdf.geometry.y.to_list())
			minx, maxx = min(xs), max(xs)
			miny, maxy = min(ys), max(ys)
			pad_x = max(0.5, (maxx - minx) * 0.15)
			pad_y = max(0.5, (maxy - miny) * 0.15)
			ax.set_xlim(minx - pad_x, maxx + pad_x)
			ax.set_ylim(miny - pad_y, maxy + pad_y)

		handles, labels = ax.get_legend_handles_labels()
		uniq: dict[str, object] = {}
		for h, l in zip(handles, labels, strict=False):
			if l not in uniq:
				uniq[l] = h
		ax.legend(uniq.values(), uniq.keys(), loc='lower right', fontsize=fontsize)

		plt.tight_layout()

		if texts:
			xs_obs = sub.geometry.x.to_list()
			ys_obs = sub.geometry.y.to_list()
			for ex_gdf in extras_gdfs:
				xs_obs.extend(ex_gdf.geometry.x.to_list())
				ys_obs.extend(ex_gdf.geometry.y.to_list())
			adjust_text(
				texts,
				x=xs_obs,
				y=ys_obs,
				ax=ax,
				expand_text=(1.05, 1.2),
				expand_points=(1.05, 1.2),
			)

		fname = out_png_template.format(affiliation=_sanitize_filename(aff))
		out_png = out_dir / fname
		out_png = save_figure(fig, out_png, dpi=200)
		print(f'Saved: {out_png}')
