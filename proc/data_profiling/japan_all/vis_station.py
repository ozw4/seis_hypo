# %%
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text
from shapely.geometry import Point


def _normalize_comment(s: str) -> str:
	# 余分な空白を 1 個にそろえる
	return ' '.join(str(s).split())


# Comment（元の所属文字列） → 英語 Legend ラベル
# ・地方区分: すべて "JMA"
# ・大学: すべて "University"
# ・東京都＋青森県＋静岡県: すべて "Prefectural Gov."
_COMMENT_LABEL_MAP_RAW: dict[str, str] = {
	'Hokkaido District  北海道地方': 'JMA',
	'Tohoku District  東北地方': 'JMA',
	'Kanto and Chubu Districts  関東・中部地方': 'JMA',
	'Kinki,Chugoku and Shikoku Districts  近畿・中国・四国地方': 'JMA',
	'Kyushu District  九州地方': 'JMA',
	'Ryukyu Is. District  琉球列島地方': 'JMA',
	'Geographical Survey Institute  国土地理院': 'GSI',
	# ---- ここから大学 → すべて "University" ----
	'Hokkaido University  北海道大学': 'University',
	'Hirosaki University  弘前大学': 'University',
	'Tohoku University  東北大学': 'University',
	'University of Tokyo  東京大学': 'University',
	'Nagoya University  名古屋大学': 'University',
	'Kyoto University  京都大学': 'University',
	'Kochi University  高知大学': 'University',
	'Kyushu University  九州大学': 'University',
	'Kagoshima University  鹿児島大学': 'University',
	# ---- 研究機関等 ----
	'National Research Institute for Earth Science and Disaster Prevention  国立研究開発法人防災科学技術研究所': 'NIED',
	'Japan Marine Science and Technology Center  国立研究開発法人海洋研究開発機構': 'JAMSTEC',
	'National Institute of Advanced Industrial Science and Technology  国立研究開発法人産業技術総合研究所': 'AIST',
	# ---- 都道府県・自治体 → Prefectural Gov. ----
	'Tokyo Metropolitan Government  東京都': 'Prefectural Gov.',
	'Aomori Prefecture  青森県': 'Prefectural Gov.',
	'Shizuoka Prefecture  静岡県': 'Prefectural Gov.',
	'Hot Springs Research Institute of Kanagawa Prefecture  神奈川県温泉地学研究所': 'Kanagawa HSRI',
	# ---- 観測網・その他 ----
	'Seismic Intensity Observation in Local Meteorological Observatory  気象官署の計測震度計': 'JMA Intensity',
	'F-net stations of National Research Institute for Earth Science and Disaster Prevention  国立研究開発法人防災科学技術研究所・広帯域地震観測網': 'NIED F-net',
	'Incorporated Research Institute for Seismology 米国大学間地震学研究連合(IRIS)': 'IRIS',
	'Association for the Development of Earthquake Prediction 公益財団法人地震予知総合研究振興会': 'ADEP',
	'Newly added stations (preliminary version) 新規追加観測点（暫定版リスト）': 'New (prelim.)',
}


_COMMENT_LABEL_MAP: dict[str, str] = {
	_normalize_comment(k): v for k, v in _COMMENT_LABEL_MAP_RAW.items()
}


def _comment_to_affiliation_en(comment: str | float) -> str:
	key = _normalize_comment(comment)
	label = _COMMENT_LABEL_MAP.get(key)
	if label is None:
		return 'Other'
	return label


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

	fig.savefig(out_png, dpi=200)
	print(f'Saved: {out_png}')


if __name__ == '__main__':
	plot_stations_by_affiliation_from_station_csv(
		station_csv='/workspace/data/station/station.csv',
		prefecture_shp='/workspace/util/N03-20240101_GML/N03-20240101_prefecture.shp',
		out_png='img/Figure_Stations_all.png',
		station_codes=None,  # サブセットにしたいときは ["ABASH2", ...] を渡す
		marker='^',
		markersize=10,
		fontsize=8,
		label_dlat=0.03,
		cmap_name='tab20',
		affiliation_colors={
			'NIED': 'tab:gray',
			'NIED F-net': 'tab:orange',
			'JMA': 'tab:blue',
		},
		show_station_labels=False,  # 全局描画時は False 推奨
	)

	plot_stations_by_affiliation_from_station_csv(
		station_csv='/workspace/data/station/station.csv',
		prefecture_shp='/workspace/util/N03-20240101_GML/N03-20240101_prefecture.shp',
		out_png='img/Figure_Stations_NIED.png',
		station_codes=None,  # サブセットにしたいときは ["ABASH2", ...] を渡す
		marker='^',
		markersize=10,
		fontsize=8,
		label_dlat=0.03,
		cmap_name='tab20',
		affiliation_colors={'NIED': 'tab:gray', 'NIED F-net': 'tab:orange'},
		affiliation_filter=['NIED', 'NIED F-net'],
		show_station_labels=False,  # 全局描画時は False 推奨
	)

	plot_stations_by_affiliation_from_station_csv(
		station_csv='/workspace/data/station/station.csv',
		prefecture_shp='/workspace/util/N03-20240101_GML/N03-20240101_prefecture.shp',
		out_png='img/Figure_Stations_JMA.png',
		station_codes=None,  # サブセットにしたいときは ["ABASH2", ...] を渡す
		marker='^',
		markersize=10,
		fontsize=8,
		label_dlat=0.03,
		cmap_name='tab20',
		affiliation_colors={'JMA': 'tab:blue'},
		affiliation_filter=['JMA', 'JMA Intensity'],
		show_station_labels=False,  # 全局描画時は False 推奨
	)
