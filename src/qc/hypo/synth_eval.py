from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from hypo.synth_eval.builders import _event_subsample_mask_from_xyz
from hypo.synth_eval.pipeline import _validate_event_subsample_config
from hypo.synth_eval.validation import (
	require_abs,
	require_dirname_only,
	require_filename_only,
)
from hypo.uncertainty_ellipsoid import ELLIPSE_COLS
from qc.hypo.heatmap import (
	HeatmapConfig,
	HeatmapOutputConfig,
	HeatmapScaleConfig,
	HeatmapSlicesConfig,
	run_heatmap_qc,
)
from viz.hypo.synth_eval import (
	save_dxdy_scatter,
	save_hist,
	save_true_pred_xyz_3view,
	save_true_pred_xyz_3view_with_uncertainty,
)


@dataclass(frozen=True)
class Config:
	dataset_dir: str
	outputs_dir: str
	receiver_geometry: str
	uncertainty_plot: UncertaintyPlotConfig
	heatmap: HeatmapConfig


@dataclass(frozen=True)
class UncertaintySlicePlaneConfig:
	enabled: bool
	coord_m: float
	half_thickness_m: float


@dataclass(frozen=True)
class UncertaintySliceConfig:
	xy: UncertaintySlicePlaneConfig
	xz: UncertaintySlicePlaneConfig
	yz: UncertaintySlicePlaneConfig

	def to_plot_specs(self) -> dict[str, dict[str, float | bool]]:
		return {
			'xy': {
				'enabled': bool(self.xy.enabled),
				'coord_m': float(self.xy.coord_m),
				'half_thickness_m': float(self.xy.half_thickness_m),
			},
			'xz': {
				'enabled': bool(self.xz.enabled),
				'coord_m': float(self.xz.coord_m),
				'half_thickness_m': float(self.xz.half_thickness_m),
			},
			'yz': {
				'enabled': bool(self.yz.enabled),
				'coord_m': float(self.yz.coord_m),
				'half_thickness_m': float(self.yz.half_thickness_m),
			},
		}


@dataclass(frozen=True)
class UncertaintyPlotConfig:
	enabled: bool
	display_mode: str
	sigma_scale_sec: float
	poor_thresh_km: float
	clip_km: float
	n_ellipse_points: int
	ellipse_lw: float
	ellipse_alpha: float
	event_subsample: dict[str, list[int]] | None
	slice: UncertaintySliceConfig | None


def _require_mapping(obj: Any, field: str) -> dict[str, Any]:
	if not isinstance(obj, dict):
		raise ValueError(f'{field} must be a mapping')
	return obj


def _read_bool(
	obj: dict[str, Any], key: str, *, field: str, default: bool | None = None
) -> bool:
	if key not in obj:
		if default is None:
			raise ValueError(f'{field} is required')
		return bool(default)
	val = obj[key]
	if not isinstance(val, bool):
		raise ValueError(f'{field} must be a boolean')
	return val


def _read_finite_float(
	obj: dict[str, Any], key: str, *, field: str, default: float | None = None
) -> float:
	if key not in obj:
		if default is None:
			raise ValueError(f'{field} is required')
		val = default
	else:
		val = obj[key]
	try:
		out = float(val)
	except (TypeError, ValueError) as e:
		raise ValueError(f'{field} must be a finite float') from e
	if not np.isfinite(out):
		raise ValueError(f'{field} must be a finite float')
	return out


def _read_nonneg_finite_float(obj: dict[str, Any], key: str, *, field: str) -> float:
	out = _read_finite_float(obj, key, field=field)
	if out < 0.0:
		raise ValueError(f'{field} must be >= 0')
	return out


def _read_display_mode(
	obj: dict[str, Any], *, key: str = 'display_mode', field: str
) -> str:
	if key not in obj:
		return 'projection'
	val = obj[key]
	if not isinstance(val, str):
		raise ValueError(f'{field} must be projection or slice')
	mode = val.strip().lower()
	if mode not in {'projection', 'slice'}:
		raise ValueError(f'{field} must be projection or slice')
	return mode


def _validate_exact_keys(
	obj: dict[str, Any], *, field: str, required: set[str]
) -> None:
	got = set(obj.keys())
	if got != required:
		missing = sorted(required - got)
		extra = sorted(got - required)
		parts = []
		if missing:
			parts.append(f'missing={missing}')
		if extra:
			parts.append(f'extra={extra}')
		detail = ', '.join(parts) if parts else 'invalid keys'
		raise ValueError(f'{field} must contain exactly {sorted(required)} ({detail})')


def _load_uncertainty_slice_plane_config(
	obj: Any, *, field: str
) -> UncertaintySlicePlaneConfig:
	plane = _require_mapping(obj, field)
	_validate_exact_keys(
		plane,
		field=field,
		required={'enabled', 'coord_m', 'half_thickness_m'},
	)
	return UncertaintySlicePlaneConfig(
		enabled=_read_bool(plane, 'enabled', field=f'{field}.enabled'),
		coord_m=_read_finite_float(plane, 'coord_m', field=f'{field}.coord_m'),
		half_thickness_m=_read_nonneg_finite_float(
			plane,
			'half_thickness_m',
			field=f'{field}.half_thickness_m',
		),
	)


def _load_uncertainty_slice_config(
	obj: Any, *, field: str, required: bool
) -> UncertaintySliceConfig | None:
	if obj is None:
		if required:
			raise ValueError(
				f'{field} is required when uncertainty_plot.display_mode=slice'
			)
		return None
	slc = _require_mapping(obj, field)
	_validate_exact_keys(slc, field=field, required={'xy', 'xz', 'yz'})
	return UncertaintySliceConfig(
		xy=_load_uncertainty_slice_plane_config(slc['xy'], field=f'{field}.xy'),
		xz=_load_uncertainty_slice_plane_config(slc['xz'], field=f'{field}.xz'),
		yz=_load_uncertainty_slice_plane_config(slc['yz'], field=f'{field}.yz'),
	)


def _apply_uncertainty_plot_event_subsample(
	df_plot: pd.DataFrame,
	true_xyz_m: np.ndarray,
	pred_xyz_m: np.ndarray,
	event_subsample: dict[str, list[int]] | None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
	if event_subsample is None:
		return df_plot, true_xyz_m, pred_xyz_m

	mask = _event_subsample_mask_from_xyz(
		pd.DataFrame(
			{
				'x_m': df_plot['x_m_true'],
				'y_m': df_plot['y_m_true'],
				'z_m': df_plot['z_m_true'],
			}
		),
		stride_ijk=tuple(event_subsample['stride_ijk']),
		keep_n_xyz=None,
	)
	return (
		df_plot.loc[mask].reset_index(drop=True),
		true_xyz_m[mask],
		pred_xyz_m[mask],
	)


def load_config(path: Path) -> Config:
	obj = yaml.safe_load(path.read_text(encoding='utf-8'))
	obj = _require_mapping(obj, 'config')
	up_obj = obj.get('uncertainty_plot')
	up = {} if up_obj is None else _require_mapping(up_obj, 'uncertainty_plot')
	up_display_mode = _read_display_mode(up, field='uncertainty_plot.display_mode')
	up_slice_cfg = None
	if up_display_mode == 'slice':
		up_slice_cfg = _load_uncertainty_slice_config(
			up.get('slice'),
			field='uncertainty_plot.slice',
			required=True,
		)
	up_event_subsample = _validate_event_subsample_config(
		up.get('event_subsample'),
		field='uncertainty_plot.event_subsample',
		allow_keep_n_xyz=False,
	)
	up_cfg = UncertaintyPlotConfig(
		enabled=_read_bool(
			up, 'enabled', field='uncertainty_plot.enabled', default=True
		),
		display_mode=up_display_mode,
		sigma_scale_sec=float(up.get('sigma_scale_sec', 1.0)),
		poor_thresh_km=float(up.get('poor_thresh_km', 5.0)),
		clip_km=float(up.get('clip_km', 10.0)),
		n_ellipse_points=int(up.get('n_ellipse_points', 100)),
		ellipse_lw=float(up.get('ellipse_lw', 0.8)),
		ellipse_alpha=float(up.get('ellipse_alpha', 0.85)),
		event_subsample=up_event_subsample,
		slice=up_slice_cfg,
	)
	hm_obj = obj.get('heatmap')
	hm = {} if hm_obj is None else hm_obj
	hm_slices = hm.get('slices') or {}
	hm_scale = hm.get('scale') or {}
	hm_output = hm.get('output') or {}
	hm_metrics = hm.get('metrics', ['err3d_m', 'horiz_m', 'dz_m'])
	if not isinstance(hm_metrics, list):
		raise TypeError('heatmap.metrics must be a list of strings')
	if not all(isinstance(m, str) for m in hm_metrics):
		raise TypeError('heatmap.metrics must be a list of strings')
	if len(hm_metrics) == 0:
		raise ValueError('heatmap.metrics must be non-empty')
	hm_global = hm_scale.get('global_across_slices', True)
	if hm_global is False:
		raise ValueError('heatmap.scale.global_across_slices must be True (fixed)')
	hm_dz_symmetric = hm_scale.get('dz_symmetric', True)
	if hm_dz_symmetric is False:
		raise ValueError('heatmap.scale.dz_symmetric must be True (fixed)')
	hm_percentile = float(hm_scale.get('percentile', 99.0))
	if not (0.0 < hm_percentile <= 100.0):
		raise ValueError('heatmap.scale.percentile must satisfy 0.0 < p <= 100.0')
	hm_has_vmin = 'vmin' in hm_scale
	hm_has_vmax = 'vmax' in hm_scale
	if hm_has_vmin != hm_has_vmax:
		raise ValueError(
			'heatmap.scale.vmin and heatmap.scale.vmax '
			'must be both specified or both omitted'
		)
	hm_vmin: float | None = None
	hm_vmax: float | None = None
	if hm_has_vmin and hm_has_vmax:
		hm_vmin = _read_finite_float(hm_scale, 'vmin', field='heatmap.scale.vmin')
		hm_vmax = _read_finite_float(hm_scale, 'vmax', field='heatmap.scale.vmax')
		if hm_vmax <= hm_vmin:
			raise ValueError('heatmap.scale.vmax must be > heatmap.scale.vmin')
	hm_cfg = HeatmapConfig(
		enabled=bool(hm.get('enabled', False)),
		metrics=[str(m) for m in hm_metrics],
		slices=HeatmapSlicesConfig(
			xy_all_depths=bool(hm_slices.get('xy_all_depths', True)),
			xz_center_y=bool(hm_slices.get('xz_center_y', True)),
			yz_center_x=bool(hm_slices.get('yz_center_x', True)),
		),
		scale=HeatmapScaleConfig(
			percentile=hm_percentile,
			global_across_slices=bool(hm_global),
			dz_symmetric=bool(hm_dz_symmetric),
			vmin=hm_vmin,
			vmax=hm_vmax,
		),
		output=HeatmapOutputConfig(
			save_npy=bool(hm_output.get('save_npy', True)),
			save_axes_json=bool(hm_output.get('save_axes_json', True)),
			out_dirname=str(hm_output.get('out_dirname', 'heatmaps')),
		),
	)
	return Config(
		dataset_dir=str(obj['dataset_dir']),
		outputs_dir=str(obj['outputs_dir']),
		receiver_geometry=str(obj['receiver_geometry']),
		uncertainty_plot=up_cfg,
		heatmap=hm_cfg,
	)


def _format_slice_meta_value(val: object) -> str:
	if val is None:
		return 'null'
	return str(val)


def _uncertainty_slice_meta_lines(
	slc: UncertaintySliceConfig | None,
) -> list[str]:
	lines: list[str] = []
	plane_cfgs = {
		'xy': slc.xy if slc is not None else None,
		'xz': slc.xz if slc is not None else None,
		'yz': slc.yz if slc is not None else None,
	}
	for plane, cfg in plane_cfgs.items():
		lines.append(
			f'slice.{plane}.enabled: '
			f'{_format_slice_meta_value(None if cfg is None else cfg.enabled)}'
		)
		lines.append(
			f'slice.{plane}.coord_m: '
			f'{_format_slice_meta_value(None if cfg is None else cfg.coord_m)}'
		)
		lines.append(
			f'slice.{plane}.half_thickness_m: '
			f'{_format_slice_meta_value(None if cfg is None else cfg.half_thickness_m)}'
		)
	return lines


def _uncertainty_title(up: UncertaintyPlotConfig) -> str:
	base = 'True vs HypoInverse (3-view) + 1σ ellipses'
	if up.display_mode == 'projection':
		return f'{base} [projection]'
	return f'{base} [slice]'


def _load_stations_from_run_output(
	*,
	run_dir: Path,
	dataset_dir: Path,
	receiver_geometry: str,
) -> tuple[np.ndarray, np.ndarray]:
	"""Load station xyz and DAS mask based on run output station_synth.csv.

	Stations are sourced from run_dir/station_synth.csv to ensure QC follows the
	actual station selection used in the run.
	"""
	station_csv = run_dir / 'station_synth.csv'
	if not station_csv.is_file():
		raise FileNotFoundError(f'missing: {station_csv}')

	df_sta = pd.read_csv(station_csv)
	for c in ['station_code', 'receiver_index']:
		if c not in df_sta.columns:
			raise ValueError(f'station_synth.csv missing column: {c}')

	s_code = df_sta['station_code']
	if s_code.isna().any():
		raise ValueError('station_synth.csv has missing station_code')
	codes = s_code.astype(str).map(str.strip)
	if (codes == '').any():
		raise ValueError('station_synth.csv has empty station_code')

	stations_is_das = codes.str.upper().str.startswith('D').to_numpy(dtype=bool)

	ri = pd.to_numeric(df_sta['receiver_index'], errors='raise')
	if ri.isna().any():
		raise ValueError('station_synth.csv has missing receiver_index')

	vals = ri.to_numpy(float)
	if not np.isfinite(vals).all():
		raise ValueError('receiver_index has non-finite values')
	if not np.equal(vals, np.round(vals)).all():
		raise ValueError('receiver_index must be integer-valued')
	idx = vals.astype(int)

	if idx.size == 0:
		raise ValueError('station_synth.csv has no stations')
	if np.unique(idx).size != idx.size:
		raise ValueError('receiver_index has duplicates')

	require_filename_only(receiver_geometry, 'receiver_geometry')
	geom_path = dataset_dir / 'geometry' / receiver_geometry
	if not geom_path.is_file():
		raise FileNotFoundError(f'missing: {geom_path}')

	recv_xyz_m = np.load(geom_path).astype(float)
	if recv_xyz_m.ndim != 2 or recv_xyz_m.shape[1] != 3:
		raise ValueError(f'receiver geometry must be (N,3), got {recv_xyz_m.shape}')

	min_i = int(idx.min())
	max_i = int(idx.max())
	if min_i < 0 or max_i >= recv_xyz_m.shape[0]:
		raise IndexError(
			'receiver_index out of range: '
			f'min={min_i} max={max_i} receivers={recv_xyz_m.shape[0]}'
		)

	stations_xyz_m = recv_xyz_m[idx]
	return stations_xyz_m, stations_is_das


def run_qc(config_path: Path) -> None:
	if not config_path.is_file():
		raise FileNotFoundError(f'config not found: {config_path}')

	cfg = load_config(config_path)

	dataset_dir = Path(cfg.dataset_dir)
	require_abs(dataset_dir, 'dataset_dir')
	require_dirname_only(cfg.outputs_dir, 'outputs_dir')
	require_filename_only(cfg.receiver_geometry, 'receiver_geometry')
	require_dirname_only(cfg.heatmap.output.out_dirname, 'heatmap.output.out_dirname')

	runs_root = config_path.resolve().parent.parent / 'runs'
	run_dir = runs_root / cfg.outputs_dir
	if not run_dir.is_dir():
		raise FileNotFoundError(f'run_dir not found: {run_dir}')

	index_csv = dataset_dir / 'index.csv'
	if not index_csv.is_file():
		raise FileNotFoundError(f'missing: {index_csv}')

	eval_csv = run_dir / 'eval_metrics.csv'
	if not eval_csv.is_file():
		raise FileNotFoundError(f'missing: {eval_csv}')

	df = pd.read_csv(eval_csv)
	n_eval = len(df)
	n_truth = len(pd.read_csv(index_csv))

	required_cols = ['horiz_m', 'dz_m', 'err3d_m']
	for c in required_cols:
		if c not in df.columns:
			raise ValueError(f'eval_metrics.csv missing column: {c}')

	# 件数整合
	summary_lines = []
	summary_lines.append(f'run_dir: {run_dir}')
	summary_lines.append(f'truth events (index.csv): {n_truth}')
	summary_lines.append(f'eval rows (eval_metrics.csv): {n_eval}')

	# 基本統計
	stats = df[['horiz_m', 'dz_m', 'err3d_m']].describe(percentiles=[0.5, 0.9, 0.95])
	stats_path = run_dir / 'qc_basic_stats.csv'
	stats.to_csv(stats_path)

	# 外れ値トップ10
	cols_show = [
		c
		for c in [
			'seq',
			'event_id',
			'event_id_str',
			'horiz_m',
			'dz_m',
			'err3d_m',
			'RMS',
			'ERH',
			'ERZ',
		]
		if c in df.columns
	]
	outliers = df.sort_values('err3d_m', ascending=False).head(10)[cols_show]
	outliers_path = run_dir / 'qc_outliers_top10.csv'
	outliers.to_csv(outliers_path, index=False)

	# プロット（各図1枚、色指定なし）
	p_err3d = run_dir / 'qc_err3d_hist.png'
	save_hist(df['err3d_m'], p_err3d, '3D error histogram', 'err3d_m')
	print(f'[OK] wrote: {p_err3d}')

	p_horiz = run_dir / 'qc_horiz_hist.png'
	save_hist(df['horiz_m'], p_horiz, 'Horizontal error histogram', 'horiz_m')
	print(f'[OK] wrote: {p_horiz}')

	p_dz = run_dir / 'qc_dz_hist.png'
	save_hist(df['dz_m'], p_dz, 'Depth error histogram', 'dz_m')
	print(f'[OK] wrote: {p_dz}')

	if 'dx_m' in df.columns and 'dy_m' in df.columns:
		p_dxdy = run_dir / 'qc_dxdy_scatter.png'
		save_dxdy_scatter(df['dx_m'].to_numpy(), df['dy_m'].to_numpy(), p_dxdy)
		print(f'[OK] wrote: {p_dxdy}')

	# テキスト要約
	qc_txt = run_dir / 'qc_summary.txt'
	qc_txt.write_text('\n'.join(summary_lines) + '\n', encoding='utf-8')

	print(stats)
	print(f'[OK] wrote: {stats_path}')
	print(f'[OK] wrote: {outliers_path}')
	print(f'[OK] wrote: {qc_txt}')

	if cfg.heatmap.enabled:
		run_heatmap_qc(
			df_eval=df,
			dataset_dir=dataset_dir,
			run_dir=run_dir,
			cfg=cfg.heatmap,
		)

	# --- True vs HypoInverse (XY/XZ/YZ 3-view) + station locations ---
	need_xyz = ['x_m_true', 'y_m_true', 'z_m_true', 'x_m_hyp', 'y_m_hyp', 'z_m_hyp']
	missing_xyz = [c for c in need_xyz if c not in df.columns]
	if missing_xyz:
		raise KeyError(
			f'missing columns for 3-view plot: {missing_xyz}. available={list(df.columns)}'
		)

	true_xyz_m = df[['x_m_true', 'y_m_true', 'z_m_true']].to_numpy(float)
	pred_xyz_m = df[['x_m_hyp', 'y_m_hyp', 'z_m_hyp']].to_numpy(float)

	mask = np.isfinite(true_xyz_m).all(axis=1) & np.isfinite(pred_xyz_m).all(axis=1)
	true_xyz_m = true_xyz_m[mask]
	pred_xyz_m = pred_xyz_m[mask]
	df_plot = df.loc[mask].reset_index(drop=True)

	stations_xyz_m, stations_is_das = _load_stations_from_run_output(
		run_dir=run_dir,
		dataset_dir=dataset_dir,
		receiver_geometry=cfg.receiver_geometry,
	)

	xy_png = run_dir / 'xy_true_vs_hyp.png'
	save_true_pred_xyz_3view(
		true_xyz_m,
		pred_xyz_m,
		xy_png,
		stations_xyz_m=stations_xyz_m,
		stations_is_das=stations_is_das,
		title='True vs HypoInverse (3-view)',
	)
	print(f'[OK] wrote: {xy_png}')

	# --- True vs HypoInverse (3-view) + 1σ uncertainty ellipses ---
	up = cfg.uncertainty_plot
	if not up.enabled:
		print('[WARN] skip uncertainty plot (disabled)')
		return

	print(
		'[INFO] uncertainty_plot: '
		f'display_mode={up.display_mode} '
		f'sigma_scale_sec={up.sigma_scale_sec} '
		f'poor_thresh_km={up.poor_thresh_km} '
		f'clip_km={up.clip_km} '
		f'n_ellipse_points={up.n_ellipse_points} '
		f'ellipse_lw={up.ellipse_lw} '
		f'ellipse_alpha={up.ellipse_alpha}'
	)
	if up.slice is not None:
		print(
			'[INFO] uncertainty_plot.slice: '
			f'xy(enabled={up.slice.xy.enabled}, coord_m={up.slice.xy.coord_m}, '
			f'half_thickness_m={up.slice.xy.half_thickness_m}) '
			f'xz(enabled={up.slice.xz.enabled}, coord_m={up.slice.xz.coord_m}, '
			f'half_thickness_m={up.slice.xz.half_thickness_m}) '
			f'yz(enabled={up.slice.yz.enabled}, coord_m={up.slice.yz.coord_m}, '
			f'half_thickness_m={up.slice.yz.half_thickness_m})'
		)
	if up.event_subsample is not None:
		print(
			'[INFO] uncertainty_plot.event_subsample.'
			f"stride_ijk={up.event_subsample['stride_ijk']}"
		)

	df_plot, true_xyz_m, pred_xyz_m = _apply_uncertainty_plot_event_subsample(
		df_plot,
		true_xyz_m,
		pred_xyz_m,
		up.event_subsample,
	)

	missing_ell = [c for c in ELLIPSE_COLS if c not in df_plot.columns]
	if missing_ell:
		print(f'[WARN] skip uncertainty plot (missing columns): {missing_ell}')
		return

	xy_unc_png = run_dir / 'xy_true_vs_hyp_uncertainty.png'
	save_true_pred_xyz_3view_with_uncertainty(
		true_xyz_m,
		pred_xyz_m,
		df_plot,
		xy_unc_png,
		stations_xyz_m=stations_xyz_m,
		stations_is_das=stations_is_das,
		title=_uncertainty_title(up),
		sigma_scale_sec=float(up.sigma_scale_sec),
		poor_thresh_km=float(up.poor_thresh_km),
		clip_km=float(up.clip_km),
		n_ellipse_points=int(up.n_ellipse_points),
		ellipse_lw=float(up.ellipse_lw),
		ellipse_alpha=float(up.ellipse_alpha),
		display_mode=up.display_mode,
		slice_specs=None if up.slice is None else up.slice.to_plot_specs(),
	)
	print(f'[OK] wrote: {xy_unc_png}')

	meta_path = run_dir / 'uncertainty_plot_meta.txt'
	meta_lines = [
		f'display_mode: {up.display_mode}',
		f'sigma_scale_sec: {up.sigma_scale_sec}',
		f'poor_thresh_km: {up.poor_thresh_km}',
		f'clip_km: {up.clip_km}',
		f'n_ellipse_points: {up.n_ellipse_points}',
		f'ellipse_lw: {up.ellipse_lw}',
		f'ellipse_alpha: {up.ellipse_alpha}',
		(
			'event_subsample: null'
			if up.event_subsample is None
			else f"event_subsample.stride_ijk: {up.event_subsample['stride_ijk']}"
		),
		*_uncertainty_slice_meta_lines(up.slice),
		'ERR: 1.0',
		'ERC: 0',
	]
	meta_path.write_text('\n'.join(meta_lines) + '\n', encoding='utf-8')
	print(f'[OK] wrote: {meta_path}')
