# %%
import subprocess
from pathlib import Path

import pandas as pd
from join_hypoinverse_prt_jma import build_joined_jma_hypo_csv
from make_hypoinverse_arc import write_hypoinverse_arc
from profile_hypoinvese_prt import plot_event_quality
from vis import plot_events_map_and_sections

from common.load_config import load_plot_preset

sta_file = Path('/workspace/data/station/stations_hypoinverse.sta')
pcrh_file = Path('/workspace/data/velocity/jma_crh/JMA2001A_P.crh')
scrh_file = Path('/workspace/data/velocity/jma_crh/JMA2001A_S.crh')
exe_file = Path('/workspace/external_source/hyp1.40/hypoinverse.exe')
cmd_file = Path('/workspace/proc/hypocenter_determination/template/jma2001a.cmd')

epicenter_csv = Path('/workspace/data/arrivetime/arrivetime_epicenters_1day.csv')
measurement_csv = Path('/workspace/data/arrivetime/arrivetime_measurements_1day.csv')

# 作業フォルダ（Hypoinverse の PHS / PRT / ARC などはここにまとめる）
run_dir = Path('./test_1day_jma_fix_depth')
run_dir.mkdir(parents=True, exist_ok=True)

# この pipeline.py 自身のスナップショットを作業フォルダに保存（再現用）
if '__file__' in globals():
	script_path = Path(__file__).resolve()
	snapshot_path = run_dir / ('bak_' + str(script_path.name))
	snapshot_path.write_text(script_path.read_text(encoding='utf-8'), encoding='utf-8')

# 出力系パス
arc_file = run_dir / 'hypoinverse_input.arc'
prt_path = run_dir / 'hypoinverse_run.prt'
out_join_csv = run_dir / 'hypoinverse_events_jma_join.csv'

img_dir = run_dir / 'img'
img_dir.mkdir(parents=True, exist_ok=True)

prefecture_shp = Path('/workspace/util/N03-20240101_GML/N03-20240101_prefecture.shp')
out_location_png = img_dir / 'Hypoinv_event_location.png'
out_jma_location_png = img_dir / 'jma_event_location.png'
plot_setting = 'japan_default'

params = load_plot_preset('/workspace/data/config/plot_config.yaml', plot_setting)

lon_min, lon_max = params['lon_range']
lat_min, lat_max = params['lat_range']
depth_min, depth_max = params['depth_range']
well_coord = params.get('well_coord')
min_mag = params.get('min_mag')
max_mag = params.get('max_mag')

extras = []
if well_coord is not None:
	extras.append(
		{
			'label': 'mobara site',
			'xy': [(well_coord[1], well_coord[0])],  # (lon, lat)
			'marker': 'o',
			'color': 'royalblue',
			'size': 30,
			'annotate': False,
		}
	)

### event_initial value ###
use_jma_flag = True
fix_depth = True
# if use_jma_flag is True, below parameters are ignored
default_depth_km = 10.0
p_centroid_top_n = 5
origin_time_offset_sec = 3.0
############################

df_epic = pd.read_csv(epicenter_csv)
df_meas = pd.read_csv(measurement_csv)

# ARC を「作業フォルダ」に出力
write_hypoinverse_arc(
	df_epic,
	df_meas,
	'/workspace/data/station/station.csv',
	str(arc_file),
	# ここで初期深さなどの挙動を切り替え可能
	default_depth_km=default_depth_km,
	use_jma_flag=use_jma_flag,
	p_centroid_top_n=p_centroid_top_n,
	origin_time_offset_sec=origin_time_offset_sec,
	fix_depth=True,
)

# Hypoinverse 実行
# - jma2001a.cmd 側では PHS/PRT/SUM/ARC を「相対パス」で書いておく想定
# - cwd=run_dir にすることで、これらは run_dir 内のファイルとして扱われる
with cmd_file.open('rb') as stdin:
	result = subprocess.run(
		[str(exe_file)],
		stdin=stdin,
		cwd=run_dir,
		capture_output=True,
		text=True,
		check=True,  # 異常終了なら CalledProcessError を投げる
	)

print(result.stdout)
print('returncode:', result.returncode)

# JMA カタログと Hypoinverse 出力の結合 CSV を作成
df = build_joined_jma_hypo_csv(
	df_epic,
	df_meas,
	prt_path,
	out_join_csv,
)
print(df.head())

# 品質プロファイル
plot_event_quality(
	df,
	out_dir=img_dir,
	lat_col='lat_deg_jma',
	lon_col='lon_deg_jma',
	depth_col='depth_km_jma',
)


if well_coord is not None:
	extras = [
		{
			'label': 'mobara site',
			'xy': [(well_coord[1], well_coord[0])],
			'marker': '*',
			'color': 'royalblue',
			'size': 30,
			'annotate': False,
		}
	]
else:
	extras = None

df_join = pd.read_csv(out_join_csv)


# 震源分布（マップ＋断面）
plot_events_map_and_sections(
	df=df_join,
	prefecture_shp=str(prefecture_shp),
	out_png=str(out_location_png),
	mag_col=None,
	origin_time_col='origin_time_hyp',
	lat_col='lat_deg_hyp',
	lon_col='lon_deg_hyp',
	depth_col='depth_km_hyp',
	markersize=30,  # 中間マグニチュードの基準サイズ
	lon_range=(lon_min, lon_max),
	lat_range=(lat_min, lat_max),
	depth_range=(depth_min, depth_max),
	extras_xy=extras,
)

plot_events_map_and_sections(
	df=df_join,
	prefecture_shp=str(prefecture_shp),
	out_png=str(out_jma_location_png),
	mag_col='mag1_jma',
	origin_time_col='origin_time_jma',
	lat_col='lat_deg_jma',
	lon_col='lon_deg_jma',
	depth_col='depth_km_jma',
	markersize=10,  # 中間マグニチュードの基準サイズ
	lon_range=(lon_min, lon_max),
	lat_range=(lat_min, lat_max),
	depth_range=(depth_min, depth_max),
	extras_xy=extras,
)
# %$
