from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2

import numpy as np
import pandas as pd
import yaml

from hypo.arc import write_hypoinverse_arc_from_phases
from hypo.cre import (
	compute_cre_layer_top_shift_km,
	compute_reference_elevation_km,
	compute_typical_station_elevation_km,
	write_cre_from_layer_tops,
	write_cre_meta,
)
from hypo.crh import write_crh
from hypo.phase_jma import extract_phase_records
from hypo.phase_weights import override_phase_weight_by_station_prefix
from hypo.sta import write_hypoinverse_sta
from hypo.station_delays import add_p_and_s_delays_from_elevation
from hypo.uncertainty_ellipsoid import ELLIPSE_COLS

from .builders import (
	_parse_event_subsample_3ints,
	build_epic_df,
	build_meas_df,
	build_station_df,
	build_truth_df,
)
from .hypoinverse_runner import (
	patch_cmd_template_for_cre,
	run_hypoinverse,
	write_cmd_from_template,
)
from .io import write_station_csv
from .metrics import evaluate
from .station_subset import (
	normalize_station_subset,
	validate_station_subset_schema,
)
from .validation import (
	require_abs,
	require_dirname_only,
	require_filename_only,
	validate_elevation_correction_config,
)


@dataclass(frozen=True)
class Config:
	dataset_dir: str
	sim_yaml: str
	outputs_dir: str
	template_cmd: str
	hypoinverse_exe: str
	receiver_geometry: str

	station_subset: dict[str, object]
	lat0: float
	lon0: float
	origin0: str
	dt_sec: float
	max_events: int
	default_depth_km: float
	fix_depth: bool

	arc_use_jma_flag: bool
	arc_p_centroid_top_n: int
	arc_origin_time_offset_sec: float

	model_type: str
	use_station_elev: bool
	cre_reference_margin_m: float
	cre_typical_station_elevation_m: float | None
	cre_n_layers: int
	z_is_depth_positive: bool

	apply_station_elevation_delay: bool

	das_station_prefix: str
	das_phase_weight_code: int
	event_filter: dict[str, list[float | None]] | None
	event_subsample: dict[str, list[int]] | None


@dataclass(frozen=True)
class SimParams:
	vp_kms: float
	vs_kms: float


def _validate_event_subsample_config(
	event_subsample: object,
	*,
	field: str = 'event_subsample',
	allow_keep_n_xyz: bool = True,
) -> dict[str, list[int]] | None:
	if event_subsample is None:
		return None
	if not isinstance(event_subsample, dict):
		raise ValueError(f'{field} must be a mapping')

	allowed = {'stride_ijk', 'keep_n_xyz'} if allow_keep_n_xyz else {'stride_ijk'}
	extra = set(event_subsample.keys()) - allowed
	if not allow_keep_n_xyz and 'keep_n_xyz' in event_subsample:
		raise ValueError(f'{field}.keep_n_xyz is not supported')
	if extra:
		raise ValueError(
			f'{field} contains unknown keys: {sorted(extra)!r}; '
			f'allowed: {sorted(allowed)!r}'
		)

	has_stride = 'stride_ijk' in event_subsample
	has_keep = 'keep_n_xyz' in event_subsample
	if has_stride and has_keep:
		raise ValueError(
			'event_subsample.stride_ijk and event_subsample.keep_n_xyz '
			'cannot be specified at the same time'
		)

	if has_stride:
		return {
			'stride_ijk': list(
				_parse_event_subsample_3ints(
					event_subsample['stride_ijk'],
					key='stride_ijk',
					min_value=1,
					field_prefix=field,
				)
			)
		}
	if has_keep:
		return {
			'keep_n_xyz': list(
				_parse_event_subsample_3ints(
					event_subsample['keep_n_xyz'],
					key='keep_n_xyz',
					min_value=1,
					field_prefix=field,
				)
			)
		}
	return None


def _validate_event_filter_config(
	event_filter: object,
) -> dict[str, list[float | None]] | None:
	if event_filter is None:
		return None
	if not isinstance(event_filter, dict):
		raise ValueError('event_filter must be a mapping')

	allowed = {'z_range_m'}
	extra = set(event_filter.keys()) - allowed
	if extra:
		raise ValueError(
			f'event_filter contains unknown keys: {sorted(extra)!r}; '
			f'allowed: {sorted(allowed)!r}'
		)
	if 'z_range_m' not in event_filter:
		return None

	raw = event_filter['z_range_m']
	if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)):
		raise ValueError('event_filter.z_range_m must be a list of [zmin_m, zmax_m]')
	if len(raw) != 2:
		raise ValueError('event_filter.z_range_m must have exactly 2 elements')

	out: list[float | None] = []
	for i, v in enumerate(raw):
		if v is None:
			out.append(None)
			continue
		if isinstance(v, bool) or not isinstance(v, (int, float)):
			raise ValueError(
				f'event_filter.z_range_m[{i}] must be a number or null'
			)
		out.append(float(v))

	zmin, zmax = out
	if zmin is not None and zmax is not None and zmin > zmax:
		raise ValueError(
			'event_filter.z_range_m invalid range: zmin_m must be <= zmax_m'
		)
	return {'z_range_m': out}


def load_config(path: Path) -> Config:
	obj = yaml.safe_load(path.read_text(encoding='utf-8'))
	mt = str(obj.get('model_type', 'CRH')).strip().upper()
	if mt not in ('CRE', 'CRH'):
		raise ValueError(f"model_type must be 'CRE' or 'CRH', got: {mt!r}")

	if 'station_set' in obj:
		raise ValueError(
			'station_set is deprecated. Use station_subset instead. '
			'Example: station_set: surface -> station_subset: {surface_indices: "all"}'
		)
	if 'station_subset' not in obj:
		raise ValueError('station_subset is required')

	station_subset = validate_station_subset_schema(obj['station_subset'])

	typ_m = obj.get('cre_typical_station_elevation_m', None)
	typical_m = float(typ_m) if typ_m is not None else None
	event_filter = _validate_event_filter_config(obj.get('event_filter'))
	event_subsample = _validate_event_subsample_config(obj.get('event_subsample'))

	return Config(
		dataset_dir=str(obj['dataset_dir']),
		sim_yaml=str(obj['sim_yaml']),
		outputs_dir=str(obj['outputs_dir']),
		template_cmd=str(obj['template_cmd']),
		hypoinverse_exe=str(obj['hypoinverse_exe']),
		receiver_geometry=str(obj['receiver_geometry']),
		station_subset=station_subset,
		lat0=float(obj['lat0']),
		lon0=float(obj['lon0']),
		origin0=str(obj['origin0']),
		dt_sec=float(obj['dt_sec']),
		max_events=int(obj['max_events']),
		default_depth_km=float(obj['default_depth_km']),
		fix_depth=bool(obj['fix_depth']),
		arc_use_jma_flag=bool(obj['arc_use_jma_flag']),
		arc_p_centroid_top_n=int(obj['arc_p_centroid_top_n']),
		arc_origin_time_offset_sec=float(obj['arc_origin_time_offset_sec']),
		model_type=mt,
		use_station_elev=bool(obj.get('use_station_elev', mt == 'CRE')),
		cre_reference_margin_m=float(obj.get('cre_reference_margin_m', 0.0)),
		cre_typical_station_elevation_m=typical_m,
		cre_n_layers=int(obj.get('cre_n_layers', 1)),
		z_is_depth_positive=bool(obj.get('z_is_depth_positive', True)),
		apply_station_elevation_delay=bool(
			obj.get('apply_station_elevation_delay', True)
		),
		das_station_prefix=str(obj.get('das_station_prefix', 'D')),
		das_phase_weight_code=int(obj.get('das_phase_weight_code', 3)),
		event_filter=event_filter,
		event_subsample=event_subsample,
	)


def _read_sim_yaml(sim_yaml: Path) -> SimParams:
	obj = yaml.safe_load(sim_yaml.read_text(encoding='utf-8'))
	model = obj['model']
	return SimParams(
		vp_kms=float(model['vp_mps']) / 1000.0,
		vs_kms=float(model['vs_mps']) / 1000.0,
	)


def _write_config_snapshot(config_path: Path, run_dir: Path) -> Path:
	out = run_dir / 'config_used.yaml'
	copy2(config_path, out)
	return out


def _write_receiver_indices_used(
	run_dir: Path,
	*,
	station_subset: dict[str, object],
	receiver_indices: np.ndarray,
	n_total: int,
	n_surface: int,
	n_das: int,
	n_surface_selected: int,
	n_das_selected: int,
	min_points: int,
) -> Path:
	path = run_dir / 'receiver_indices_used.yaml'
	obj = {
		'station_subset_input': station_subset,
		'n_total_receivers': int(n_total),
		'n_surface_receivers': int(n_surface),
		'n_das_receivers': int(n_das),
		'n_surface_selected': int(n_surface_selected),
		'n_das_selected': int(n_das_selected),
		'min_points': int(min_points),
		'receiver_indices_count': int(receiver_indices.size),
		'receiver_indices': [int(i) for i in receiver_indices.tolist()],
	}
	path.write_text(
		yaml.safe_dump(obj, sort_keys=False, allow_unicode=True),
		encoding='utf-8',
	)
	return path


def _load_station_codes_from_receiver_catalog(
	dataset_dir: Path,
	*,
	expected_len: int,
) -> np.ndarray:
	meta_path = dataset_dir / 'dataset_meta.json'
	if not meta_path.is_file():
		raise FileNotFoundError(f'missing: {meta_path}')
	meta = json.loads(meta_path.read_text(encoding='utf-8'))
	rel = meta.get('optional', {}).get('receiver_catalog_csv_rel', None)
	if rel is None or str(rel).strip() == '':
		raise ValueError(
			'dataset_meta.json missing optional.receiver_catalog_csv_rel (receiver catalog path)'
		)
	catalog_path = dataset_dir / str(rel)
	if not catalog_path.is_file():
		raise FileNotFoundError(f'missing: {catalog_path}')

	df = pd.read_csv(catalog_path)
	for col in ['receiver_index', 'station_code']:
		if col not in df.columns:
			raise ValueError(f'receiver catalog missing column: {col}')

	ridx = df['receiver_index']
	if ridx.isna().any():
		raise ValueError('receiver catalog receiver_index contains NaN')
	if (ridx.map(lambda v: isinstance(v, bool))).any():
		raise ValueError('receiver catalog receiver_index contains bool')
	if not pd.api.types.is_integer_dtype(ridx):
		raise ValueError('receiver catalog receiver_index must be integer dtype')
	ridx_i = ridx.astype(int)
	if (ridx_i.to_numpy() < 0).any():
		raise ValueError('receiver catalog receiver_index contains negative')
	if ridx_i.nunique(dropna=False) != len(ridx_i):
		raise ValueError('receiver catalog receiver_index has duplicates')

	df2 = df[['receiver_index', 'station_code']].copy()
	df2['receiver_index'] = ridx_i
	df2 = df2.sort_values('receiver_index').reset_index(drop=True)

	if len(df2) != int(expected_len):
		raise ValueError(
			'receiver catalog length mismatch: '
			f'len={len(df2)} expected_len={int(expected_len)}'
		)
	expected = np.arange(int(expected_len), dtype=int)
	got = df2['receiver_index'].to_numpy(dtype=int)
	if not np.array_equal(got, expected):
		raise ValueError(
			'receiver catalog receiver_index must be contiguous 0..N-1 in sorted order: '
			f'got min={int(got.min())} max={int(got.max())} expected_len={int(expected_len)}'
		)

	codes = df2['station_code'].astype(str)
	if (codes.str.len() == 0).any():
		raise ValueError('receiver catalog station_code contains empty string')
	return codes.to_numpy(dtype=str)


def build_synth_layer_tops_km(n_layers: int) -> list[float]:
	"""Build base synthetic layer tops (km) for n_layers.

	Returns:
		[0.0, 1.0, 2.0, ..., n_layers-1]

	"""
	n = int(n_layers)
	if n < 1:
		raise ValueError('n_layers must be >= 1')
	return [0.0] + [float(i) for i in range(1, n)]


def write_synth_cre_models(
	run_dir: Path,
	*,
	vp_kms: float,
	vs_kms: float,
	shift_km: float,
	n_layers: int,
) -> tuple[Path, Path]:
	"""Write P/S CRE model files for synthetic evaluation.

	This function only generates base synthetic layer tops and delegates file
	output to write_cre_from_layer_tops().
	"""
	layer_tops_km = build_synth_layer_tops_km(n_layers)
	return write_cre_from_layer_tops(
		run_dir,
		vp_kms=float(vp_kms),
		vs_kms=float(vs_kms),
		layer_tops_km=layer_tops_km,
		shift_km=float(shift_km),
	)


def run_synth_eval(
	config_path: Path, *, runs_root: Path
) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
	if not config_path.is_file():
		raise FileNotFoundError(f'config not found: {config_path}')

	cfg = load_config(config_path)
	validate_elevation_correction_config(
		model_type=str(cfg.model_type),
		use_station_elev=bool(cfg.use_station_elev),
		apply_station_elevation_delay=bool(cfg.apply_station_elevation_delay),
	)

	dataset_dir = Path(cfg.dataset_dir)
	template_cmd = Path(cfg.template_cmd)
	hypoinverse_exe = Path(cfg.hypoinverse_exe)

	require_abs(dataset_dir, 'dataset_dir')
	require_abs(template_cmd, 'template_cmd')
	require_abs(hypoinverse_exe, 'hypoinverse_exe')

	require_filename_only(cfg.sim_yaml, 'sim_yaml')
	require_filename_only(cfg.receiver_geometry, 'receiver_geometry')
	require_dirname_only(cfg.outputs_dir, 'outputs_dir')

	sim_yaml = dataset_dir / 'provenance' / cfg.sim_yaml
	receiver_geometry = dataset_dir / 'geometry' / cfg.receiver_geometry
	index_csv = dataset_dir / 'index.csv'
	events_dir = dataset_dir / 'events'

	if not sim_yaml.is_file():
		raise FileNotFoundError(f'missing: {sim_yaml}')
	if not receiver_geometry.is_file():
		raise FileNotFoundError(f'missing: {receiver_geometry}')
	if not index_csv.is_file():
		raise FileNotFoundError(f'missing: {index_csv}')
	if not events_dir.is_dir():
		raise FileNotFoundError(f'missing: {events_dir}')
	if not template_cmd.is_file():
		raise FileNotFoundError(f'missing: {template_cmd}')
	if not hypoinverse_exe.is_file():
		raise FileNotFoundError(f'missing: {hypoinverse_exe}')

	run_dir = runs_root / cfg.outputs_dir
	run_dir.mkdir(parents=True, exist_ok=True)

	config_snapshot = _write_config_snapshot(config_path, run_dir)
	print(f'[OK] wrote: {config_snapshot}')

	station_csv = run_dir / 'station_synth.csv'
	sta_file = run_dir / 'stations_synth.sta'
	arc_file = run_dir / 'hypoinverse_input.arc'
	p_crh = run_dir / 'P.crh'
	s_crh = run_dir / 'S.crh'
	cmd_file = run_dir / 'synth.cmd'

	prt_file = run_dir / 'hypoinverse_run.prt'
	sum_file = run_dir / 'hypoinverse_run.sum'
	out_arc_file = run_dir / 'hypoinverse_run_out.arc'
	eval_csv = run_dir / 'eval_metrics.csv'
	eval_stats_csv = run_dir / 'eval_stats.csv'

	sim = _read_sim_yaml(sim_yaml)
	origin0 = pd.to_datetime(cfg.origin0)

	recv_xyz_m = np.load(receiver_geometry).astype(float)
	nrec = int(recv_xyz_m.shape[0])
	station_codes_all = _load_station_codes_from_receiver_catalog(
		dataset_dir,
		expected_len=nrec,
	)
	codes = pd.Series(station_codes_all)
	receiver_indices = normalize_station_subset(
		cfg.station_subset,
		codes=codes,
		expected_len=nrec,
		min_points=4,
	)
	stations_is_das = codes.str.upper().str.startswith('D').to_numpy(dtype=bool)
	n_surface = int((~stations_is_das).sum())
	n_das = int(stations_is_das.sum())
	selected_is_das = stations_is_das[receiver_indices]
	n_surface_selected = int((~selected_is_das).sum())
	n_das_selected = int(selected_is_das.sum())
	recv_idx_used_yaml = _write_receiver_indices_used(
		run_dir,
		station_subset=cfg.station_subset,
		receiver_indices=receiver_indices,
		n_total=nrec,
		n_surface=n_surface,
		n_das=n_das,
		n_surface_selected=n_surface_selected,
		n_das_selected=n_das_selected,
		min_points=4,
	)
	print(f'[OK] wrote: {recv_idx_used_yaml}')

	station_df = build_station_df(
		recv_xyz_m,
		receiver_indices,
		station_codes_all,
		cfg.lat0,
		cfg.lon0,
		z_is_depth_positive=bool(cfg.z_is_depth_positive),
	)
	if cfg.apply_station_elevation_delay:
		station_df = add_p_and_s_delays_from_elevation(
			station_df,
			vp_kms=float(sim.vp_kms),
			vs_kms=float(sim.vs_kms),
		)

	truth_df = build_truth_df(
		index_csv,
		cfg.lat0,
		cfg.lon0,
		origin0,
		cfg.dt_sec,
		cfg.max_events,
		event_z_range_m=(
			cfg.event_filter.get('z_range_m') if cfg.event_filter else None
		),
		event_stride_ijk=(
			cfg.event_subsample.get('stride_ijk') if cfg.event_subsample else None
		),
		event_keep_n_xyz=(
			cfg.event_subsample.get('keep_n_xyz') if cfg.event_subsample else None
		),
	)
	print(f'[INFO] selected events: {len(truth_df)}')
	epic_df = build_epic_df(truth_df, cfg.default_depth_km)
	meas_df = build_meas_df(events_dir, truth_df, station_df)

	write_station_csv(station_df, station_csv)
	print(f'[OK] wrote: {station_csv}')
	write_hypoinverse_sta(
		station_csv,
		sta_file,
		force_zero_pdelays=(
			str(cfg.model_type).strip().upper() == 'CRE' and bool(cfg.use_station_elev)
		),
	)
	print(f'[OK] wrote: {sta_file}')

	phases = extract_phase_records(meas_df)
	phases = override_phase_weight_by_station_prefix(
		phases,
		station_prefix=str(cfg.das_station_prefix),
		weight=int(cfg.das_phase_weight_code),
	)

	write_hypoinverse_arc_from_phases(
		epic_df,
		phases,
		station_csv,
		arc_file,
		default_depth_km=float(cfg.default_depth_km),
		use_jma_flag=bool(cfg.arc_use_jma_flag),
		p_centroid_top_n=int(cfg.arc_p_centroid_top_n),
		origin_time_offset_sec=float(cfg.arc_origin_time_offset_sec),
		fix_depth=bool(cfg.fix_depth),
	)
	print(f'[OK] wrote: {arc_file}')

	mt = str(cfg.model_type).strip().upper()
	if mt == 'CRE':
		ref_elev_km = compute_reference_elevation_km(
			station_df,
			elevation_col='Elevation_m',
			margin_m=float(cfg.cre_reference_margin_m),
		)
		typical_elev_km = compute_typical_station_elevation_km(
			explicit_m=cfg.cre_typical_station_elevation_m,
		)
		shift_km = compute_cre_layer_top_shift_km(ref_elev_km, typical_elev_km)
		write_cre_meta(
			run_dir,
			ref_elev_km=ref_elev_km,
			typical_elev_km=typical_elev_km,
			shift_km=shift_km,
		)
		print(f'[OK] wrote: {run_dir / "cre_ref_elev_km.txt"}')
		print(f'[OK] wrote: {run_dir / "cre_typical_station_elev_km.txt"}')
		print(f'[OK] wrote: {run_dir / "cre_layer_top_shift_km.txt"}')

		p_cre, s_cre = write_synth_cre_models(
			run_dir,
			vp_kms=float(sim.vp_kms),
			vs_kms=float(sim.vs_kms),
			shift_km=shift_km,
			n_layers=int(cfg.cre_n_layers),
		)
		print(f'[OK] wrote: {p_cre}')
		print(f'[OK] wrote: {s_cre}')
		patch_cmd_template_for_cre(
			template_cmd,
			cmd_file,
			sta_file=str(sta_file.name),
			p_model=str(p_cre.name),
			s_model=str(s_cre.name),
			ref_elev_km=ref_elev_km,
			use_station_elev=bool(cfg.use_station_elev),
		)
		print(f'[OK] wrote: {cmd_file}')
	else:
		write_crh(p_crh, 'SYNTH_P', [(float(sim.vp_kms), 0.0)])
		write_crh(s_crh, 'SYNTH_S', [(float(sim.vs_kms), 0.0)])
		print(f'[OK] wrote: {p_crh}')
		print(f'[OK] wrote: {s_crh}')
		write_cmd_from_template(template_cmd, cmd_file)
		print(f'[OK] wrote: {cmd_file}')
	run_hypoinverse(hypoinverse_exe, cmd_file, run_dir)

	if not prt_file.is_file():
		raise FileNotFoundError(f'missing: {prt_file}')
	if not sum_file.is_file():
		raise FileNotFoundError(f'missing: {sum_file}')
	if not out_arc_file.is_file():
		raise FileNotFoundError(f'missing: {out_arc_file}')
	print(f'[OK] wrote: {prt_file}')
	print(f'[OK] wrote: {sum_file}')
	print(f'[OK] wrote: {out_arc_file}')

	df_eval = evaluate(truth_df, prt_file, cfg.lat0, cfg.lon0)
	df_eval.to_csv(eval_csv, index=False)
	if not eval_csv.is_file():
		raise FileNotFoundError(f'missing: {eval_csv}')
	print(f'[OK] wrote: {eval_csv}')

	missing_ell = [c for c in ELLIPSE_COLS if c not in df_eval.columns]
	if missing_ell:
		raise ValueError(
			'ell_* columns missing from eval_metrics.csv (ERROR ELLIPSE not parsed): '
			f'{missing_ell}'
		)

	metrics_cols = ['horiz_m', 'dz_m', 'err3d_m', 'RMS', 'ERH', 'ERZ']
	missing = [c for c in metrics_cols if c not in df_eval.columns]
	if missing:
		raise ValueError(f'eval df missing columns: {missing}')

	stats = df_eval[metrics_cols].describe(percentiles=[0.5, 0.9, 0.95])
	stats.to_csv(eval_stats_csv)
	print(f'[OK] wrote: {eval_stats_csv}')

	return run_dir, df_eval, stats
