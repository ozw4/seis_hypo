from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

# ---- Defaults (あなたの現状に合わせて適宜調整) ----
DEFAULT_CHANNEL_TABLE = Path(
	'/workspace/proc/util/hinet_util/hinet_channelstbl_20251007'
)

DEFAULT_VEL1D_SRC = Path('velocity/vjma2001')
DEFAULT_LAYERS_OUT = Path('velocity/jma2001.layers')

DEFAULT_NLL_RUN_DIR = Path('nll/run')
DEFAULT_NLL_MODEL_DIR = Path('nll/model')
DEFAULT_NLL_TIME_DIR = Path('nll/time')

DEFAULT_LOKI_HEADER_OUT = Path('db/header.hdr')
DEFAULT_QC_FIG_DIR = Path('qc')


@dataclass(frozen=True)
class TravelTimeBaseConfig:
	"""走時前処理に共通の設定。
	QCも本番パイプラインもここを継承して使う。
	"""

	# --- Station selection ---
	center_lat: float
	center_lon: float
	radius_km: float
	channel_table_path: Path = DEFAULT_CHANNEL_TABLE

	# --- Grid proposal ---
	dx_km: float = 1.0
	dy_km: float = 1.0
	dz_km: float = 1.0
	pad_km: float = 10.0
	z0_km: float = -5.0
	zmax_km: float = 80.0

	# ここを Base に上げておくと、
	# 同じ traveltime_config.yaml を QC と本番で共用しやすい
	center_mode: Literal['fixed', 'mean', 'median'] = 'fixed'

	# --- 1D velocity -> NLL LAYER ---
	vel1d_src: Path = DEFAULT_VEL1D_SRC
	layers_out: Path = DEFAULT_LAYERS_OUT
	strict_1dvel: bool = False

	# --- NonLinLoc outputs ---
	model_label: str = 'jma2001'
	nll_run_dir: Path = DEFAULT_NLL_RUN_DIR
	nll_model_dir: Path = DEFAULT_NLL_MODEL_DIR
	nll_time_dir: Path = DEFAULT_NLL_TIME_DIR

	# --- NLL calc params (QCでも文字列生成までは使うので共通に置いてOK) ---
	quantity: str = 'SLOW_LEN'
	gtmode: str = 'GRID3D ANGLES_NO'
	depth_km_mode: Literal['zero', 'from_elevation'] = 'zero'

	# --- LOKI header output ---
	loki_header_out: Path = DEFAULT_LOKI_HEADER_OUT


@dataclass(frozen=True)
class TravelTimePipelineConfig(TravelTimeBaseConfig):
	"""本番のVel2Grid/Grid2Timeまで回すための設定。"""

	# center_mode は Base に移動したので追加フィールド不要


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
