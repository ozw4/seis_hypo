# %%
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from build_epic_from_eqt_picks import build_epic_from_eqt_picks
from extract_phase_records import extract_eqt_phase_records
from filter_das_picks import filter_and_decimate_das_picks
from join_hypoinverse_prt_jma import build_joined_jma_hypo_csv
from make_hypoinverse_arc import (
	write_hypoinverse_arc_from_phases,
)
from match_mobaradas_events_to_jma import extract_das_phase_records
from profile_hypoinvese_prt import plot_event_quality
from vis import plot_events_map_and_sections

from common.load_config import load_plot_preset

sta_file = Path('/workspace/data/station/stations_hypoinverse_with_das.sta')
pcrh_file = Path('/workspace/data/velocity/jma_crh/JMA2001A_P.crh')
scrh_file = Path('/workspace/data/velocity/jma_crh/JMA2001A_S.crh')
exe_file = Path('/workspace/external_source/hyp1.40/hypoinverse.exe')
cmd_file = Path(
	'/workspace/proc/hypocenter_determination/template/jma2001a_with_das.cmd'
)
measurement_csv = Path(
	'/home/dcuser/mobara2025/proc/proc_continuous/outputs/hinet_gather_events_mobaranet1/eqt_picks_hinet_eqt_for_hypoinverse.csv'
)
das_measurment_csv = Path(
	'/home/dcuser/mobara2025/proc/proc_continuous_das/das_picks_20200215_20200301.csv'
)
das_epicenter_csv = Path(
	'/home/dcuser/mobara2025/proc/proc_continuous_das/events_summary_20200215_20200301.csv'
)

pick_npz = Path(
	'/home/dcuser/mobara2025/proc/proc_continuous/outputs/hinet_gather_events_mobaranet1/eqt_input_hinet_mobaranet1_60s_3ch.npz'
)

data = np.load(pick_npz, allow_pickle=True)

# npz に入れておいたステーション情報を取り出す
sta_code = data['sta_code']  # (n_sta,)
sta_lat = data['sta_lat']  # (n_sta,)
sta_lon = data['sta_lon']  # (n_sta,)

# bytes で入っている場合は str にそろえる
if np.issubdtype(sta_code.dtype, np.bytes_):
	sta_code = sta_code.astype(str)

station_df = pd.DataFrame(
	dict(
		station=sta_code,
		lat=sta_lat,
		lon=sta_lon,
	)
)

# ============================================================
# フィルタ条件（日時 / das_score）
#   - target_start / target_end : JMA origin_time / DAS event_time の範囲
#   - min_das_score             : epicenter_csv 側の das_score の下限
#                                 None にすればスコアフィルタなし
# ============================================================
target_start = pd.Timestamp('2020-02-15 00:00:00')
target_end = pd.Timestamp('2020-03-02 00:00:00')  # target_end は「未満」比較

# 作業フォルダ（Hypoinverse の PHS / PRT / ARC などはここにまとめる）
run_dir = Path('./test_mobara2020_jma_fpevent')
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
plot_setting = 'mobara_default'

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
use_jma_flag = False
fix_depth = False
# if use_jma_flag is True, below parameters are ignored
default_depth_km = 10.0
p_centroid_top_n = 5
origin_time_offset_sec = 3.0
############################

# =========================
# データ読み込み
# =========================

eqt_df = pd.read_csv(measurement_csv)
df_epic = build_epic_from_eqt_picks(eqt_df, station_df)


df_das_epic = pd.read_csv(das_epicenter_csv)
df_das_meas = pd.read_csv(das_measurment_csv)

df_epic = df_epic.copy()
df_epic['origin_dt'] = pd.to_datetime(df_epic['origin_time'])

mask_time_epic = (df_epic['origin_dt'] >= target_start) & (
	df_epic['origin_dt'] < target_end
)

mask_epic = mask_time_epic
df_epic = df_epic[mask_epic].reset_index(drop=True)


# =========================
# DAS 側: 日時でフィルタ
#   - events_summary_* : event_time
#   - das_picks_*      : date（文字列）と event_time_peak がある想定
# =========================
df_das_epic = df_das_epic.copy()
df_das_epic['event_time'] = pd.to_datetime(df_das_epic['event_time'])
mask_time_das_event = (df_das_epic['event_time'] >= target_start) & (
	df_das_epic['event_time'] < target_end
)
df_das_epic = df_das_epic[mask_time_das_event].reset_index(drop=True)

df_das_meas = df_das_meas.copy()
df_das_meas['event_time_peak'] = pd.to_datetime(df_das_meas['event_time_peak'])
mask_time_das_pick = (df_das_meas['event_time_peak'] >= target_start) & (
	df_das_meas['event_time_peak'] < target_end
)
df_das_meas = df_das_meas[mask_time_das_pick].reset_index(drop=True)

# =========================
# DAS ピックのフィルタ＋間引き
# =========================
df_das_meas_filtered = filter_and_decimate_das_picks(
	df_das_epic,
	df_das_meas,
	dt_sec=0.01,
	fiber_spacing_m=1.0,
	channel_start=200,  # use_ch_range.start と合わせる
	win_half_samples=500,  # idx-500:idx+500 に合わせる
	residual_thresh_s=0.05,  # ±0.05s 以内を「整合的」とみなす
	spacing_m=25.0,  # 100m ごとに 1 チャネルに間引き
)

# =========================
# Hypoinverse 用フェーズ作成
# =========================
phases_hinet = extract_eqt_phase_records(eqt_df)
phases_das = extract_das_phase_records(df_epic, df_das_meas_filtered, max_dt_sec=10.0)
phases_all = phases_hinet + phases_das

write_hypoinverse_arc_from_phases(
	df_epic,
	phases_all,
	'/workspace/data/station/station_with_das.csv',
	str(arc_file),
	default_depth_km=default_depth_km,
	use_jma_flag=use_jma_flag,
	p_centroid_top_n=p_centroid_top_n,
	origin_time_offset_sec=origin_time_offset_sec,
	fix_depth=fix_depth,
)

# =========================
# Hypoinverse 実行
# =========================
with cmd_file.open('rb') as stdin:
	result = subprocess.run(
		[str(exe_file)],
		stdin=stdin,
		cwd=run_dir,
		capture_output=True,
		text=True,
		check=True,
	)

print(result.stdout)
print('returncode:', result.returncode)

# =========================
# JMA カタログと Hypoinverse 出力の結合 CSV を作成
# =========================
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
	hist_ranges={
		'RMS': (0.0, 1.5),
	},
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

# 震源分布（Hypoinverse 解：マップ＋断面）
plot_events_map_and_sections(
	df=df_join,
	prefecture_shp=str(prefecture_shp),
	out_png=str(out_location_png),
	mag_col=None,
	origin_time_col='origin_time_hyp',
	lat_col='lat_deg_hyp',
	lon_col='lon_deg_hyp',
	depth_col='depth_km_hyp',
	markersize=30,
	lon_range=(lon_min, lon_max),
	lat_range=(lat_min, lat_max),
	depth_range=(depth_min, depth_max),
	extras_xy=extras,
)

# 震源分布（JMA 解：マップ＋断面）
plot_events_map_and_sections(
	df=df_join,
	prefecture_shp=str(prefecture_shp),
	out_png=str(out_jma_location_png),
	mag_col='mag1_jma',
	origin_time_col='origin_time_jma',
	lat_col='lat_deg_jma',
	lon_col='lon_deg_jma',
	depth_col='depth_km_jma',
	markersize=10,
	lon_range=(lon_min, lon_max),
	lat_range=(lat_min, lat_max),
	depth_range=(depth_min, depth_max),
	extras_xy=extras,
)
# %%
