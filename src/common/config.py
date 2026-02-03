from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml

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
	xy_half_width_km: float
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
	origin_time_start: str | None = None
	origin_time_end: str | None = None
	mag_min: float | None = None
	mag_max: float | None = None
	drop_if_mag_missing: bool = True


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
		(<output_dir>/DEFAULT_NLL_RUN_DIR の親)/qc/<preset 名>

	例:
		output_dir = "/tmp/out"
		DEFAULT_NLL_RUN_DIR = "nll/run"
		-> fig_dir = "/tmp/out/nll/qc/mobara"
	"""
	if cfg.fig_dir is not None:
		return cfg

	run_parent = (Path(cfg.output_dir) / DEFAULT_NLL_RUN_DIR).parent
	fig_dir = run_parent / 'qc' / preset_name
	return replace(cfg, fig_dir=fig_dir)


@dataclass(frozen=True, slots=True)
class EqTInputs:
	"""EqTransformer inference parameters.

	Note:
	- eqt_weights: SeisBench pretrained name (e.g. 'original') OR local weights path.
	- channel_prefix: output trace channel prefix, last char becomes 'P'/'S' (e.g. 'HH' -> 'HHP'/'HHS')

	"""

	eqt_weights: str = 'original'
	eqt_in_samples: int = 6000
	eqt_overlap: int = 3000
	eqt_batch_size: int = 64
	eqt_channel_prefix: str = 'HH'


@dataclass(frozen=True)
class JmaDtPickErrorRunConfig:
	run_id: str
	out_dir: Path
	overwrite: bool = False
	notes: str = ''


@dataclass(frozen=True)
class JmaDtPickErrorInputsConfig:
	event_root: Path
	epicenters_csv: Path
	measurements_csv: Path
	mapping_report_csv: Path
	near0_csv: Path
	monthly_presence_csv: Path
	mag1_types_allowed: list[str]
	distance: str
	phase_defs: dict[str, list[str]]
	stations_allowlist: list[str] | None = None
	event_id_allowlist: list[int] | None = None


@dataclass(frozen=True)
class JmaDtPickErrorBandpassConfig:
	fstop_lo: float
	fpass_lo: float
	fpass_hi: float
	fstop_hi: float
	gpass: float
	gstop: float


@dataclass(frozen=True)
class JmaDtPickErrorPreprocessConfig:
	preprocess_preset: str
	fs_target_hz: float
	detrend: str | None
	bandpass: JmaDtPickErrorBandpassConfig


@dataclass(frozen=True)
class JmaDtPickErrorStaltaConfig:
	transform: str
	sta_sec: float
	lta_sec: float


@dataclass(frozen=True)
class JmaDtPickErrorPickerConfig:
	picker_name: str
	picker_preset: str
	phase: str
	component: str
	stalta: JmaDtPickErrorStaltaConfig | None = None


@dataclass(frozen=True)
class JmaDtPickErrorPickExtractConfig:
	search_pre_sec: float
	search_post_sec: float
	clip_search_window: bool
	choose: str
	tie_break: str
	thr: float
	min_sep_sec: float
	search_i1_inclusive: bool


@dataclass(frozen=True)
class JmaDtPickErrorEvalConfig:
	tol_sec: list[float]
	keep_missing_rows: bool


@dataclass(frozen=True)
class JmaDtPickErrorOutputConfig:
	dt_table_csv: Path
	skips_csv: Path
	save_config_snapshot: bool


@dataclass(frozen=True)
class JmaDtPickErrorExperiment:
	name: str | None = None
	run: dict[str, Any] | None = None
	inputs: dict[str, Any] | None = None
	preprocess: dict[str, Any] | None = None
	picker: dict[str, Any] | None = None
	pick_extract: dict[str, Any] | None = None
	eval: dict[str, Any] | None = None
	output: dict[str, Any] | None = None


@dataclass(frozen=True)
class JmaDtPickErrorConfigV1:
	version: int
	run: JmaDtPickErrorRunConfig
	inputs: JmaDtPickErrorInputsConfig
	preprocess: JmaDtPickErrorPreprocessConfig
	picker: JmaDtPickErrorPickerConfig
	pick_extract: JmaDtPickErrorPickExtractConfig
	eval: JmaDtPickErrorEvalConfig
	output: JmaDtPickErrorOutputConfig
	experiments: list[JmaDtPickErrorExperiment]


_DT_PICK_ERROR_ALLOWED_KEYS = {
	'version',
	'run',
	'inputs',
	'preprocess',
	'picker',
	'pick_extract',
	'eval',
	'output',
	'experiments',
}
_DT_PICK_ERROR_EXPERIMENT_KEYS = {
	'name',
	'run',
	'inputs',
	'preprocess',
	'picker',
	'pick_extract',
	'eval',
	'output',
}
_TEMPLATE_RE = re.compile(r'\$\{([^}]+)\}')


def load_dt_pick_error_config_v1(
	yaml_path: str | Path,
) -> JmaDtPickErrorConfigV1:
	yaml_path = Path(yaml_path)
	if not yaml_path.is_file():
		raise FileNotFoundError(f'YAML が見つかりません: {yaml_path}')

	with yaml_path.open('r', encoding='utf-8') as f:
		cfg = yaml.safe_load(f)

	if not isinstance(cfg, dict):
		raise ValueError('YAML のトップレベルは mapping である必要があります')

	unknown = sorted(set(cfg.keys()) - _DT_PICK_ERROR_ALLOWED_KEYS)
	if unknown:
		raise ValueError(f'未知のトップレベルキーがあります: {unknown}')

	if 'version' not in cfg:
		raise ValueError('version が必要です')
	if int(cfg['version']) != 1:
		raise ValueError(f'unsupported version: {cfg["version"]!r}')

	for key in [
		'run',
		'inputs',
		'preprocess',
		'picker',
		'pick_extract',
		'eval',
		'output',
	]:
		if key not in cfg:
			raise ValueError(f'{key} が必要です')

	exp_list = cfg.get('experiments')
	if exp_list is None:
		exp_list = []

	if not isinstance(exp_list, list):
		raise ValueError('experiments は list である必要があります')

	run_cfg = _parse_dt_pick_error_run(cfg['run'])
	inputs_cfg = _parse_dt_pick_error_inputs(cfg['inputs'])
	pre_cfg = _parse_dt_pick_error_preprocess(cfg['preprocess'])
	picker_cfg = _parse_dt_pick_error_picker(cfg['picker'])
	pick_extract_cfg = _parse_dt_pick_error_pick_extract(cfg['pick_extract'])
	eval_cfg = _parse_dt_pick_error_eval(cfg['eval'])
	output_cfg = _parse_dt_pick_error_output(cfg['output'])
	exps = _parse_dt_pick_error_experiments(exp_list)

	return JmaDtPickErrorConfigV1(
		version=1,
		run=run_cfg,
		inputs=inputs_cfg,
		preprocess=pre_cfg,
		picker=picker_cfg,
		pick_extract=pick_extract_cfg,
		eval=eval_cfg,
		output=output_cfg,
		experiments=exps,
	)


def expand_dt_pick_error_experiments(
	cfg: JmaDtPickErrorConfigV1,
) -> list[JmaDtPickErrorConfigV1]:
	base_dict = _dt_pick_error_config_to_dict(cfg)
	base_dict['experiments'] = []

	out: list[JmaDtPickErrorConfigV1] = []
	base_rendered = _apply_run_out_dir_template(copy.deepcopy(base_dict))
	out.append(_build_dt_pick_error_config_from_dict(base_rendered))

	for exp in cfg.experiments:
		merged = copy.deepcopy(base_dict)
		_apply_exp_override(merged, 'run', exp.run)
		_apply_exp_override(merged, 'inputs', exp.inputs)
		_apply_exp_override(merged, 'preprocess', exp.preprocess)
		_apply_exp_override(merged, 'picker', exp.picker)
		_apply_exp_override(merged, 'pick_extract', exp.pick_extract)
		_apply_exp_override(merged, 'eval', exp.eval)
		_apply_exp_override(merged, 'output', exp.output)

		if exp.name:
			run_id = str(merged['run']['run_id'])
			merged['run']['run_id'] = f'{run_id}__{exp.name}'

		merged['experiments'] = []
		merged = _apply_run_out_dir_template(merged)
		out.append(_build_dt_pick_error_config_from_dict(merged))

	return out


def _dt_pick_error_config_to_dict(cfg: JmaDtPickErrorConfigV1) -> dict[str, Any]:
	return asdict(cfg)


def _apply_exp_override(
	merged: dict[str, Any],
	section: str,
	override: dict[str, Any] | None,
) -> None:
	if override is None:
		return
	if not isinstance(override, dict):
		raise ValueError(f'experiment.{section} must be mapping')
	merged[section] = _deep_merge_dict(merged[section], override)


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
	out = dict(base)
	for k, v in override.items():
		if isinstance(v, dict) and isinstance(out.get(k), dict):
			out[k] = _deep_merge_dict(out[k], v)
		else:
			out[k] = v
	return out


def _build_dt_pick_error_config_from_dict(
	cfg: dict[str, Any],
) -> JmaDtPickErrorConfigV1:
	return JmaDtPickErrorConfigV1(
		version=int(cfg['version']),
		run=_parse_dt_pick_error_run(cfg['run']),
		inputs=_parse_dt_pick_error_inputs(cfg['inputs']),
		preprocess=_parse_dt_pick_error_preprocess(cfg['preprocess']),
		picker=_parse_dt_pick_error_picker(cfg['picker']),
		pick_extract=_parse_dt_pick_error_pick_extract(cfg['pick_extract']),
		eval=_parse_dt_pick_error_eval(cfg['eval']),
		output=_parse_dt_pick_error_output(cfg['output']),
		experiments=[],
	)


def _apply_run_out_dir_template(cfg: dict[str, Any]) -> dict[str, Any]:
	run = cfg.get('run')
	if not isinstance(run, dict):
		raise ValueError('run must be mapping')
	out_dir = run.get('out_dir')
	if out_dir is None:
		raise ValueError('run.out_dir is required')
	out_dir_s = str(out_dir)
	if '${' in out_dir_s:
		out_dir_s = _render_template_vars(out_dir_s, cfg)
	run['out_dir'] = out_dir_s
	return cfg


def _render_template_vars(s: str, root: dict[str, Any]) -> str:
	def _lookup(path: str) -> str:
		parts = [p for p in path.split('.') if p]
		cur: Any = root
		for p in parts:
			if not isinstance(cur, dict) or p not in cur:
				raise KeyError(f'template key {path!r} is not defined')
			cur = cur[p]
		return str(cur)

	def _replace(m: re.Match[str]) -> str:
		key = m.group(1).strip()
		return _lookup(key)

	return _TEMPLATE_RE.sub(_replace, s)


def _expect_mapping(obj: Any, *, name: str) -> dict[str, Any]:
	if not isinstance(obj, dict):
		raise ValueError(f'{name} は mapping である必要があります')
	return obj


def _require_keys(obj: dict[str, Any], keys: set[str], *, name: str) -> None:
	miss = sorted(set(keys) - set(obj.keys()))
	if miss:
		raise ValueError(f'{name} に必要なキーがありません: {miss}')


def _deny_unknown_keys(obj: dict[str, Any], keys: set[str], *, name: str) -> None:
	unknown = sorted(set(obj.keys()) - set(keys))
	if unknown:
		raise ValueError(f'{name} に未知のキーがあります: {unknown}')


def _as_path(v: Any, *, name: str) -> Path:
	if isinstance(v, Path):
		return v
	if isinstance(v, str):
		return Path(v)
	raise ValueError(f'{name} must be str or Path')


def _as_str_list(v: Any, *, name: str) -> list[str]:
	if not isinstance(v, list):
		raise ValueError(f'{name} must be list')
	out = []
	for item in v:
		out.append(str(item))
	return out


def _parse_dt_pick_error_run(obj: Any) -> JmaDtPickErrorRunConfig:
	o = _expect_mapping(obj, name='run')
	_require_keys(o, {'run_id', 'out_dir'}, name='run')
	_deny_unknown_keys(o, {'run_id', 'out_dir', 'overwrite', 'notes'}, name='run')
	return JmaDtPickErrorRunConfig(
		run_id=str(o['run_id']),
		out_dir=_as_path(o['out_dir'], name='run.out_dir'),
		overwrite=bool(o.get('overwrite', False)),
		notes=str(o.get('notes', '')),
	)


def _parse_dt_pick_error_inputs(obj: Any) -> JmaDtPickErrorInputsConfig:
	o = _expect_mapping(obj, name='inputs')
	_require_keys(
		o,
		{
			'event_root',
			'epicenters_csv',
			'measurements_csv',
			'mapping_report_csv',
			'near0_csv',
			'monthly_presence_csv',
			'mag1_types_allowed',
			'distance',
			'phase_defs',
		},
		name='inputs',
	)
	_deny_unknown_keys(
		o,
		{
			'event_root',
			'epicenters_csv',
			'measurements_csv',
			'mapping_report_csv',
			'near0_csv',
			'monthly_presence_csv',
			'mag1_types_allowed',
			'distance',
			'phase_defs',
			'stations_allowlist',
			'event_id_allowlist',
		},
		name='inputs',
	)
	phase_defs = _expect_mapping(o['phase_defs'], name='inputs.phase_defs')
	phase_defs_out: dict[str, list[str]] = {}
	for k, v in phase_defs.items():
		phase_defs_out[str(k)] = _as_str_list(v, name=f'inputs.phase_defs.{k}')

	stations_allowlist = o.get('stations_allowlist', None)
	if stations_allowlist is not None:
		stations_allowlist = _as_str_list(
			stations_allowlist, name='inputs.stations_allowlist'
		)

	event_id_allowlist = o.get('event_id_allowlist', None)
	if event_id_allowlist is not None:
		if not isinstance(event_id_allowlist, list):
			raise ValueError('inputs.event_id_allowlist must be list')
		event_id_allowlist = [int(x) for x in event_id_allowlist]

	return JmaDtPickErrorInputsConfig(
		event_root=_as_path(o['event_root'], name='inputs.event_root'),
		epicenters_csv=_as_path(o['epicenters_csv'], name='inputs.epicenters_csv'),
		measurements_csv=_as_path(
			o['measurements_csv'], name='inputs.measurements_csv'
		),
		mapping_report_csv=_as_path(
			o['mapping_report_csv'], name='inputs.mapping_report_csv'
		),
		near0_csv=_as_path(o['near0_csv'], name='inputs.near0_csv'),
		monthly_presence_csv=_as_path(
			o['monthly_presence_csv'], name='inputs.monthly_presence_csv'
		),
		mag1_types_allowed=_as_str_list(
			o['mag1_types_allowed'], name='inputs.mag1_types_allowed'
		),
		distance=str(o['distance']),
		phase_defs=phase_defs_out,
		stations_allowlist=stations_allowlist,
		event_id_allowlist=event_id_allowlist,
	)


def _parse_dt_pick_error_bandpass(obj: Any) -> JmaDtPickErrorBandpassConfig:
	o = _expect_mapping(obj, name='preprocess.bandpass')
	_require_keys(
		o,
		{'fstop_lo', 'fpass_lo', 'fpass_hi', 'fstop_hi', 'gpass', 'gstop'},
		name='preprocess.bandpass',
	)
	_deny_unknown_keys(
		o,
		{'fstop_lo', 'fpass_lo', 'fpass_hi', 'fstop_hi', 'gpass', 'gstop'},
		name='preprocess.bandpass',
	)
	return JmaDtPickErrorBandpassConfig(
		fstop_lo=float(o['fstop_lo']),
		fpass_lo=float(o['fpass_lo']),
		fpass_hi=float(o['fpass_hi']),
		fstop_hi=float(o['fstop_hi']),
		gpass=float(o['gpass']),
		gstop=float(o['gstop']),
	)


def _parse_dt_pick_error_preprocess(
	obj: Any,
) -> JmaDtPickErrorPreprocessConfig:
	o = _expect_mapping(obj, name='preprocess')
	_require_keys(
		o,
		{'preprocess_preset', 'fs_target_hz', 'detrend', 'bandpass'},
		name='preprocess',
	)
	_deny_unknown_keys(
		o,
		{'preprocess_preset', 'fs_target_hz', 'detrend', 'bandpass'},
		name='preprocess',
	)
	return JmaDtPickErrorPreprocessConfig(
		preprocess_preset=str(o['preprocess_preset']),
		fs_target_hz=float(o['fs_target_hz']),
		detrend=None if o['detrend'] is None else str(o['detrend']),
		bandpass=_parse_dt_pick_error_bandpass(o['bandpass']),
	)


def _parse_dt_pick_error_stalta(
	obj: Any,
) -> JmaDtPickErrorStaltaConfig:
	o = _expect_mapping(obj, name='picker.stalta')
	_require_keys(o, {'transform', 'sta_sec', 'lta_sec'}, name='picker.stalta')
	_deny_unknown_keys(o, {'transform', 'sta_sec', 'lta_sec'}, name='picker.stalta')
	return JmaDtPickErrorStaltaConfig(
		transform=str(o['transform']),
		sta_sec=float(o['sta_sec']),
		lta_sec=float(o['lta_sec']),
	)


def _parse_dt_pick_error_picker(obj: Any) -> JmaDtPickErrorPickerConfig:
	o = _expect_mapping(obj, name='picker')
	_require_keys(
		o, {'picker_name', 'picker_preset', 'phase', 'component'}, name='picker'
	)
	_deny_unknown_keys(
		o,
		{
			'picker_name',
			'picker_preset',
			'phase',
			'component',
			'stalta',
		},
		name='picker',
	)
	stalta = o.get('stalta')
	stalta_cfg = None
	if stalta is not None:
		stalta_cfg = _parse_dt_pick_error_stalta(stalta)

	picker_name = str(o['picker_name'])
	if picker_name == 'stalta' and stalta_cfg is None:
		raise ValueError('picker.stalta is required when picker_name=stalta')

	return JmaDtPickErrorPickerConfig(
		picker_name=picker_name,
		picker_preset=str(o['picker_preset']),
		phase=str(o['phase']),
		component=str(o['component']),
		stalta=stalta_cfg,
	)


def _parse_dt_pick_error_pick_extract(
	obj: Any,
) -> JmaDtPickErrorPickExtractConfig:
	o = _expect_mapping(obj, name='pick_extract')
	_require_keys(
		o,
		{
			'search_pre_sec',
			'search_post_sec',
			'clip_search_window',
			'choose',
			'tie_break',
			'thr',
			'min_sep_sec',
			'search_i1_inclusive',
		},
		name='pick_extract',
	)
	_deny_unknown_keys(
		o,
		{
			'search_pre_sec',
			'search_post_sec',
			'clip_search_window',
			'choose',
			'tie_break',
			'thr',
			'min_sep_sec',
			'search_i1_inclusive',
		},
		name='pick_extract',
	)
	return JmaDtPickErrorPickExtractConfig(
		search_pre_sec=float(o['search_pre_sec']),
		search_post_sec=float(o['search_post_sec']),
		clip_search_window=bool(o['clip_search_window']),
		choose=str(o['choose']),
		tie_break=str(o['tie_break']),
		thr=float(o['thr']),
		min_sep_sec=float(o['min_sep_sec']),
		search_i1_inclusive=bool(o['search_i1_inclusive']),
	)


def _parse_dt_pick_error_eval(obj: Any) -> JmaDtPickErrorEvalConfig:
	o = _expect_mapping(obj, name='eval')
	_require_keys(o, {'tol_sec', 'keep_missing_rows'}, name='eval')
	_deny_unknown_keys(o, {'tol_sec', 'keep_missing_rows'}, name='eval')
	if not isinstance(o['tol_sec'], list):
		raise ValueError('eval.tol_sec must be list')
	tol_sec = [float(x) for x in o['tol_sec']]
	return JmaDtPickErrorEvalConfig(
		tol_sec=tol_sec,
		keep_missing_rows=bool(o['keep_missing_rows']),
	)


def _parse_dt_pick_error_output(obj: Any) -> JmaDtPickErrorOutputConfig:
	o = _expect_mapping(obj, name='output')
	_require_keys(
		o, {'dt_table_csv', 'skips_csv', 'save_config_snapshot'}, name='output'
	)
	_deny_unknown_keys(
		o,
		{'dt_table_csv', 'skips_csv', 'save_config_snapshot'},
		name='output',
	)
	return JmaDtPickErrorOutputConfig(
		dt_table_csv=_as_path(o['dt_table_csv'], name='output.dt_table_csv'),
		skips_csv=_as_path(o['skips_csv'], name='output.skips_csv'),
		save_config_snapshot=bool(o['save_config_snapshot']),
	)


def _parse_dt_pick_error_experiments(
	exp_list: list[Any],
) -> list[JmaDtPickErrorExperiment]:
	out: list[JmaDtPickErrorExperiment] = []
	for i, item in enumerate(exp_list):
		if not isinstance(item, dict):
			raise ValueError(f'experiments[{i}] must be mapping')
		unknown = sorted(set(item.keys()) - _DT_PICK_ERROR_EXPERIMENT_KEYS)
		if unknown:
			raise ValueError(f'experiments[{i}] has unknown keys: {unknown}')
		for key in _DT_PICK_ERROR_EXPERIMENT_KEYS - {'name'}:
			if (
				key in item
				and item[key] is not None
				and not isinstance(item[key], dict)
			):
				raise ValueError(f'experiments[{i}].{key} must be mapping')
		out.append(
			JmaDtPickErrorExperiment(
				name=None if item.get('name') is None else str(item.get('name')),
				run=item.get('run'),
				inputs=item.get('inputs'),
				preprocess=item.get('preprocess'),
				picker=item.get('picker'),
				pick_extract=item.get('pick_extract'),
				eval=item.get('eval'),
				output=item.get('output'),
			)
		)
	return out
