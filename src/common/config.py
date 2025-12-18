from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

# ---- Defaults (あなたの現状に合わせて適宜調整) ----
DEFAULT_CHANNEL_TABLE = Path(
	'/workspace/proc/util/hinet_util/hinet_channelstbl_20251007'
)

# 出力は output_dir / <DEFAULT_*> に展開される前提で「相対パス固定」
DEFAULT_VEL1D_SRC = Path('velocity/vjma2001')
DEFAULT_LAYERS_OUT = Path('velocity/jma2001.layers')

DEFAULT_NLL_RUN_DIR = Path('nll/run')
DEFAULT_NLL_MODEL_DIR = Path('nll/model')
DEFAULT_LOKI_HEADER_OUT = Path('db/header.hdr')  # 固定

DEFAULT_QC_FIG_DIR = Path('qc')


@dataclass(frozen=True)
class PrepareEventsConfig:
	# カタログとイベント出力
	catalog_csv: Path
	base_input_dir: Path = Path('proc/inputs/events')

	# Hi-net
	network_code: str = '0101'
	fs: float = 100.0
	pre_sec: int = 20
	post_sec: int = 120
	hinet_threads: int = 8

	# 観測点選択（局配置）
	station_site_lat: float = 0.0
	station_site_lon: float = 0.0
	station_radius_km: float = 50.0

	# イベント選択（期間・マグ・震央距離）
	start_time: str | None = None
	end_time: str | None = None
	min_mag: float | None = None
	max_mag: float | None = None
	event_radius_km: float | None = None

	# 上限
	max_events: int | None = None

	# 任意メタ
	config_name: str | None = None


@dataclass(frozen=True)
class TravelTimeBaseConfig:
	"""走時前処理に共通の設定。
	QCも本番パイプラインもここを継承して使う。
	"""

	# ★YAMLで指定するのはこれだけ（出力のルート）
	output_dir: Path

	# station / grid
	center_lat: float
	center_lon: float
	radius_km: float
	channel_table_path: Path

	dx_km: float
	dy_km: float
	dz_km: float
	pad_km: float
	z0_km: float
	zmax_km: float
	center_mode: str

	# velocity input（入力なので絶対/任意パスでOK）
	vel1d_src: Path
	strict_1dvel: bool = False

	# NLL/LOKI
	model_label: str = 'jma2001'
	quantity: str = 'SLOW_LEN'
	gtmode: str = 'GRID3D ANGLES_NO'
	depth_km_mode: str = 'zero'


@dataclass(frozen=True)
class LokiWaveformStackingInputs:
	tshortp_min: float = 0.2
	tshortp_max: float = 0.2
	tshorts_min: float = 0.3
	tshorts_max: float = 0.3
	slrat: float = 5
	npr: int = 2
	ntrial: int = 1
	derivative: bool = True
	vfunc: str = 'erg'
	hfunc: str = 'pca'
	model: str = 'jma2001'
	epsilon: float = 0.001
	base_sampling_rate_hz: int = 100

	# --- preprocess before LOKI ---
	pre_enable: bool = True
	pre_detrend: str | None = 'linear'
	pre_fstop_lo: float = 0.5
	pre_fpass_lo: float = 1.0
	pre_fpass_hi: float = 23.0
	pre_fstop_hi: float = 25.0
	pre_gpass: float = 1.0
	pre_gstop: float = 40.0

	# --- robust (MAD) scaling ---
	pre_mad_scale: bool = True
	pre_mad_eps: float = 1e-6
	pre_mad_c: float = 1.4826


@dataclass(frozen=True)
class LokiWaveformStackingPipelineConfig:
	# download_win_for_event が作ったイベントディレクトリ群（event.json + WIN32が入っている場所）
	base_input_dir: Path
	base_traveltime_dir: Path
	# LOKIは data_path 配下の leaf dir をイベントとして列挙するので、ここに空dirを作る
	loki_data_path: Path

	# LOKIの出力先
	loki_output_path: Path

	# 走時表
	loki_db_path: Path
	loki_hdr_filename: str

	# inputs yaml（preset式）
	inputs_yaml: Path
	inputs_preset: str = 'default'

	# LOKI params
	comp: tuple[str, ...] = ('U', 'N', 'E')  # ★決め打ち
	precision: str = 'single'
	search: str = 'classic'
	extension: str = '*'

	# イベント選別（必要なら）
	event_glob: str = '[0-9]*'
	max_events: int | None = None


@dataclass(frozen=True)
class QcConfig(TravelTimeBaseConfig):
	"""1) 半径で局抽出(rows)
	2) grid提案
	3) LOKI header
	4) 1Dvel -> LAYER
	5) control(P/S)

	までのQC用設定。
	"""

	# YAML には書かなくてOKにする
	fig_dir: Path | None = None


def resolve_qc_config_fig_dir(cfg: QcConfig, preset_name: str) -> QcConfig:
	"""QC の fig_dir を自動決定する。

	ルール:
		<run_dir の親>/qc/<preset 名>

	例:
		nll_run_dir = "nll/run"
		-> fig_dir = "nll/qc/mobara"
	"""
	if cfg.fig_dir is not None:
		return cfg

	run_parent = Path(cfg.nll_run_dir).parent
	fig_dir = run_parent / 'qc' / preset_name
	return replace(cfg, fig_dir=fig_dir)
