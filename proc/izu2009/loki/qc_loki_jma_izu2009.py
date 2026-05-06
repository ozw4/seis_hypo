"""QC wrapper for Izu2009 JMA-referenced Loki EqT outputs."""

# ruff: noqa: ANN401, PLR0913

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use('Agg')

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _REPO_ROOT / 'src'

for _path in (_REPO_ROOT, _SRC_DIR):
	_path_str = str(_path)
	if _path_str not in sys.path:
		sys.path.insert(0, _path_str)

from common.geo import haversine_distance_pair_km  # noqa: E402
from common.json_io import read_json, write_json  # noqa: E402
from common.load_config import load_config  # noqa: E402
from common.time_util import to_utc  # noqa: E402
from loki_tools.compare_df import build_compare_df  # noqa: E402
from loki_tools.plot import (  # noqa: E402
	make_extras_lld_jma,
	make_links_lld,
	make_loki_plot_df,
	plot_loki_results_quickcheck,
)
from loki_tools.plot_error_stats import (  # noqa: E402
	make_coherence_bins,
	plot_box,
	plot_hist,
)
from qc.loki.waveforms_with_loki_picks import (  # noqa: E402
	plot_waveforms_with_picks_for_event,
)
from viz.events_map import plot_events_map_and_sections  # noqa: E402
from viz.loki.coherence_xy import plot_loki_event_coherence_xy_overlay  # noqa: E402
from viz.plot_config import PlotConfig  # noqa: E402
from waveform.preprocess import DetrendBandpassSpec  # noqa: E402

JMA_EVENTS_CSV = (
	_REPO_ROOT
	/ 'proc/izu2009/catalog/out/jma_events_izu2009_20091217_20091220_r50km.csv'
)
EVENTS_BASE_DIR = _REPO_ROOT / 'proc/izu2009/loki/events_from_gamma'
LOKI_OUTPUT_DIR = _REPO_ROOT / 'proc/izu2009/loki/output_eqt_gamma'
HEADER_PATH = _REPO_ROOT / 'proc/izu2009/loki/traveltime/db/header.hdr'

REPO_PLOT_CONFIG_YAML = _REPO_ROOT / 'proc/izu2009/loki/config/plot_config.yaml'
REPO_PLOT_CONFIG_IZU_YAML = (
	_REPO_ROOT / 'proc/izu2009/loki/config/plot_config_izu2009.yaml'
)
WORKSPACE_PLOT_CONFIG_YAML = Path('/workspace/data/config/plot_config.yaml')
PLOT_CONFIG_PRESET = 'izu_default'
PREFECTURE_SHP = Path('/workspace/data/N03-20240101_GML/N03-20240101_prefecture.shp')

EVENT_GLOB = '[0-9]*'
MAX_MATCH_DT_SEC = 30.0
MAX_MATCH_DH_KM = 20.0
TOP_N_CMAX = 10
TOP_N_DH = 5
TOP_N_DZ = 5
BASE_SAMPLING_RATE_HZ = 100
COMPONENTS_ORDER = ('U', 'N', 'E')
PLOT_COMPONENTS = ('U', 'N', 'E')
Y_TIME = 'relative'

JMA_REQUIRED_COLUMNS = {
	'event_id',
	'origin_time',
	'latitude_deg',
	'longitude_deg',
	'depth_km',
}

MATCH_COLUMNS = [
	'loki_event_id',
	'gamma_event_index',
	'gamma_origin_time_utc',
	'jma_event_id',
	'jma_origin_time_jst',
	'jma_origin_time_utc',
	'dt_gamma_minus_jma_sec',
	'gamma_jma_dh_km',
	'gamma_latitude_deg',
	'gamma_longitude_deg',
	'gamma_depth_km',
	'jma_latitude_deg',
	'jma_longitude_deg',
	'jma_depth_km',
	'jma_mag1',
	'match_status',
]


def _require_dir(path: Path, label: str) -> None:
	if not path.is_dir():
		raise FileNotFoundError(f'{label} not found: {path}')


def _require_file(path: Path, label: str) -> None:
	if not path.is_file():
		raise FileNotFoundError(f'{label} not found: {path}')


def _resolve_plot_config_yaml(cli_path: Path | None) -> Path:
	"""Resolve plot config from CLI, repo-local candidates, then workspace default."""
	candidates: list[Path] = []
	if cli_path is not None:
		candidates.append(Path(cli_path))
	else:
		candidates.extend(
			[
				REPO_PLOT_CONFIG_YAML,
				REPO_PLOT_CONFIG_IZU_YAML,
				WORKSPACE_PLOT_CONFIG_YAML,
			]
		)

	for path in candidates:
		if path.is_file():
			return path

	lines = [f'  - {path}' for path in candidates]
	raise FileNotFoundError(
		'plot config YAML not found. Provide --plot-config-yaml or create one of:\n'
		+ '\n'.join(lines)
	)


def _resolve_prefecture_shp(cli_path: Path | None) -> Path | None:
	if cli_path is not None:
		return Path(cli_path)
	if PREFECTURE_SHP.is_file():
		return PREFECTURE_SHP
	return None


def _event_dirs(path: Path, event_glob: str) -> list[Path]:
	return sorted(p for p in path.glob(event_glob) if p.is_dir())


def _event_dirs_with_loc(event_dirs: list[Path], loki_output_dir: Path) -> list[Path]:
	out: list[Path] = []
	for event_dir in event_dirs:
		ev_out_dir = loki_output_dir / event_dir.name
		if ev_out_dir.is_dir() and any(ev_out_dir.glob('*.loc')):
			out.append(event_dir)
	return out


def _validate_inputs(
	*,
	jma_events_csv: Path,
	events_base_dir: Path,
	loki_output_dir: Path,
	header_path: Path,
	plot_config_yaml: Path,
	prefecture_shp: Path | None,
	event_glob: str,
) -> tuple[list[Path], dict[str, str]]:
	_require_file(jma_events_csv, 'JMA events CSV')
	_require_dir(events_base_dir, 'events_from_gamma')
	_require_dir(loki_output_dir, 'output_eqt_gamma')
	_require_file(loki_output_dir / 'catalogue', 'Loki catalogue')
	_require_file(header_path, 'Loki header')
	_require_file(plot_config_yaml, 'plot config YAML')
	if prefecture_shp is not None:
		_require_file(prefecture_shp, 'prefecture shapefile')

	loki_event_dirs = _event_dirs(loki_output_dir, event_glob)
	if not loki_event_dirs:
		raise FileNotFoundError(
			f'no Loki event output dirs in {loki_output_dir} (glob={event_glob})'
		)
	if not any(any(p.glob('*.loc')) for p in loki_event_dirs):
		raise FileNotFoundError(f'no .loc files under Loki output: {loki_output_dir}')

	input_event_dirs = _event_dirs(events_base_dir, event_glob)
	if not input_event_dirs:
		raise FileNotFoundError(
			f'no event dirs in {events_base_dir} (glob={event_glob})'
		)

	loki_ids = {p.name for p in loki_event_dirs}
	paired = [p for p in input_event_dirs if p.name in loki_ids]
	if not paired:
		raise RuntimeError(
			'no event ids exist in both events_from_gamma and output_eqt_gamma'
		)

	skipped: dict[str, str] = {}
	matched_with_loc = _event_dirs_with_loc(paired, loki_output_dir)
	for event_dir in paired:
		if event_dir not in matched_with_loc:
			skipped[event_dir.name] = 'missing .loc'

	if not matched_with_loc:
		raise FileNotFoundError(
			'no matched event dirs contain a .loc file under Loki output'
		)

	return matched_with_loc, skipped


def _load_event_json_file(event_dir: Path) -> dict[str, Any]:
	path = event_dir / 'event.json'
	if not path.is_file():
		raise FileNotFoundError(f'event.json not found: {path}')
	obj = read_json(path)
	if not isinstance(obj, dict):
		raise TypeError(f'event.json must contain an object: {path}')
	return obj


def _as_mapping(obj: Any, *, label: str, path: Path) -> dict[str, Any]:
	if obj is None:
		return {}
	if not isinstance(obj, dict):
		raise TypeError(f'{label} must be an object: {path}')
	return obj


def _first_present(*values: Any) -> Any:
	for value in values:
		if value is not None:
			return value
	return None


def _to_jst(ts: Any, *, naive_tz: str) -> pd.Timestamp:
	t = pd.Timestamp(ts)
	if pd.isna(t):
		raise ValueError(f'failed to parse timestamp: {ts!r}')
	if t.tzinfo is None:
		t = t.tz_localize(naive_tz)
	return t.tz_convert('Asia/Tokyo')


def _to_utc(ts: Any, *, naive_tz: str) -> pd.Timestamp:
	return to_utc(pd.Timestamp(ts), naive_tz=naive_tz)


def _format_utc_z(ts: Any) -> str:
	t = _to_utc(ts, naive_tz='UTC')
	return t.to_pydatetime().isoformat(timespec='milliseconds').replace('+00:00', 'Z')


def _format_jst_iso(ts: Any) -> str:
	return (
		_to_jst(ts, naive_tz='Asia/Tokyo')
		.to_pydatetime()
		.isoformat(timespec='milliseconds')
	)


def _to_json_scalar(value: Any) -> Any:
	if value is None:
		return None
	if pd.isna(value):
		return None
	if isinstance(value, np.integer):
		return int(value)
	if isinstance(value, np.floating):
		return float(value)
	if isinstance(value, np.bool_):
		return bool(value)
	return value


def _load_jma_events_csv(path: Path) -> pd.DataFrame:
	_require_file(path, 'JMA events CSV')
	df = pd.read_csv(path, dtype={'event_id': str})
	missing = JMA_REQUIRED_COLUMNS - set(df.columns)
	if missing:
		raise ValueError(f'JMA events CSV missing columns: {sorted(missing)}')

	rows: list[dict[str, Any]] = []
	for r in df.itertuples(index=False):
		origin_raw = r.origin_time
		origin_jst = _to_jst(origin_raw, naive_tz='Asia/Tokyo')
		origin_utc = origin_jst.tz_convert('UTC')
		mag1 = r.mag1 if 'mag1' in df.columns else np.nan
		rows.append(
			{
				'jma_event_id': str(r.event_id),
				'jma_origin_time_jst': origin_jst,
				'jma_origin_time_utc': origin_utc,
				'jma_latitude_deg': float(r.latitude_deg),
				'jma_longitude_deg': float(r.longitude_deg),
				'jma_depth_km': float(r.depth_km),
				'jma_mag1': pd.to_numeric(pd.Series([mag1]), errors='coerce').iloc[0],
			}
		)

	out = pd.DataFrame(rows)
	out['jma_origin_time_jst'] = pd.to_datetime(out['jma_origin_time_jst'])
	out['jma_origin_time_utc'] = pd.to_datetime(out['jma_origin_time_utc'], utc=True)
	return out


def _origin_from_event_json(
	ev: dict[str, Any],
	extra: dict[str, Any],
	gamma: dict[str, Any],
	path: Path,
) -> pd.Timestamp:
	origin = _first_present(
		gamma.get('origin_time_utc'),
		extra.get('origin_time_utc'),
		ev.get('origin_time_utc'),
		gamma.get('origin_time'),
		extra.get('origin_time'),
		ev.get('origin_time'),
	)
	if origin is not None:
		return _to_utc(origin, naive_tz='UTC')

	origin_jst = _first_present(
		gamma.get('origin_time_jst'),
		extra.get('origin_time_jst'),
		ev.get('origin_time_jst'),
	)
	if origin_jst is not None:
		return _to_utc(origin_jst, naive_tz='Asia/Tokyo')

	raise ValueError(
		f'event.json missing GaMMA origin time: {path} '
		'(need extra.gamma.origin_time_utc or fallback origin_time)'
	)


def _load_gamma_reference_events(event_dirs: list[Path]) -> pd.DataFrame:
	rows: list[dict[str, Any]] = []
	for event_dir in event_dirs:
		event_json_path = event_dir / 'event.json'
		ev = _load_event_json_file(event_dir)
		extra = _as_mapping(
			ev.get('extra'), label='event.json extra', path=event_json_path
		)
		gamma = _as_mapping(
			extra.get('gamma'), label='event.json extra.gamma', path=event_json_path
		)

		origin_utc = _origin_from_event_json(ev, extra, gamma, event_json_path)
		lat = _first_present(
			gamma.get('latitude_deg'),
			extra.get('latitude_deg'),
			ev.get('latitude_deg'),
			gamma.get('lat'),
			extra.get('lat'),
			ev.get('lat'),
		)
		lon = _first_present(
			gamma.get('longitude_deg'),
			extra.get('longitude_deg'),
			ev.get('longitude_deg'),
			gamma.get('lon'),
			extra.get('lon'),
			ev.get('lon'),
		)
		depth = _first_present(
			gamma.get('depth_km'),
			extra.get('depth_km'),
			ev.get('depth_km'),
			gamma.get('z_km'),
			extra.get('z_km'),
			ev.get('z_km'),
		)
		if lat is None or lon is None or depth is None:
			raise ValueError(
				f'event.json missing GaMMA lat/lon/depth: {event_json_path}'
			)

		rows.append(
			{
				'loki_event_id': event_dir.name,
				'gamma_event_index': _first_present(
					gamma.get('event_index'),
					extra.get('event_index'),
					ev.get('event_index'),
					ev.get('event_id'),
				),
				'gamma_origin_time_utc': origin_utc,
				'gamma_latitude_deg': float(lat),
				'gamma_longitude_deg': float(lon),
				'gamma_depth_km': float(depth),
			}
		)

	out = pd.DataFrame(rows)
	if out.empty:
		raise RuntimeError('no GaMMA/LOKI reference events loaded')
	out['gamma_origin_time_utc'] = pd.to_datetime(
		out['gamma_origin_time_utc'], utc=True
	)
	return out


def _empty_match_df() -> pd.DataFrame:
	return pd.DataFrame(columns=MATCH_COLUMNS)


def _mark_duplicate_matches(
	match_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	df = match_df.copy()
	candidate = df['match_status'].eq('matched')
	duplicate = pd.Series(data=False, index=df.index)

	if candidate.any():
		matched = df.loc[candidate]
		duplicate_jma = matched['jma_event_id'].duplicated(keep=False)
		duplicate_loki = matched['loki_event_id'].duplicated(keep=False)
		duplicate.loc[matched.index] = (
			duplicate_jma.to_numpy() | duplicate_loki.to_numpy()
		)

	df.loc[duplicate, 'match_status'] = 'duplicate'
	dups = df.loc[duplicate, MATCH_COLUMNS].copy()
	if dups.empty:
		dups = _empty_match_df()
	return df, dups


def _build_jma_loki_match_table(
	*,
	jma_df: pd.DataFrame,
	gamma_df: pd.DataFrame,
	max_match_dt_sec: float,
	max_match_dh_km: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	gamma_sorted = gamma_df.sort_values('gamma_origin_time_utc').reset_index(drop=True)
	jma_sorted = jma_df.sort_values('jma_origin_time_utc').reset_index(drop=True)

	match = pd.merge_asof(
		gamma_sorted,
		jma_sorted,
		left_on='gamma_origin_time_utc',
		right_on='jma_origin_time_utc',
		direction='nearest',
		tolerance=pd.Timedelta(seconds=float(max_match_dt_sec)),
	)

	match['dt_gamma_minus_jma_sec'] = (
		match['gamma_origin_time_utc'] - match['jma_origin_time_utc']
	).dt.total_seconds()
	match.loc[match['jma_origin_time_utc'].isna(), 'dt_gamma_minus_jma_sec'] = np.nan

	has_jma = match['jma_event_id'].notna()
	match['gamma_jma_dh_km'] = np.nan
	if has_jma.any():
		match.loc[has_jma, 'gamma_jma_dh_km'] = haversine_distance_pair_km(
			match.loc[has_jma, 'gamma_latitude_deg'].to_numpy(float),
			match.loc[has_jma, 'gamma_longitude_deg'].to_numpy(float),
			match.loc[has_jma, 'jma_latitude_deg'].to_numpy(float),
			match.loc[has_jma, 'jma_longitude_deg'].to_numpy(float),
		)

	match['match_status'] = 'matched'
	match.loc[~has_jma, 'match_status'] = 'loki_only'
	match.loc[
		has_jma & (match['gamma_jma_dh_km'].astype(float) > float(max_match_dh_km)),
		'match_status',
	] = 'rejected_by_distance'

	match, duplicates = _mark_duplicate_matches(match)
	match = match.reindex(columns=MATCH_COLUMNS).sort_values('loki_event_id')

	matched_jma_ids = set(
		match.loc[match['match_status'].eq('matched'), 'jma_event_id']
		.dropna()
		.astype(str)
	)
	jma_unmatched = jma_df[
		~jma_df['jma_event_id'].astype(str).isin(matched_jma_ids)
	].copy()
	jma_unmatched['match_status'] = 'jma_only'

	loki_unmatched = match.loc[~match['match_status'].eq('matched')].copy()
	if loki_unmatched.empty:
		loki_unmatched = _empty_match_df()
	if duplicates.empty:
		duplicates = _empty_match_df()

	return match, loki_unmatched, jma_unmatched, duplicates


def _write_match_outputs(
	*,
	match_df: pd.DataFrame,
	loki_unmatched_df: pd.DataFrame,
	jma_unmatched_df: pd.DataFrame,
	duplicates_df: pd.DataFrame,
	out_dir: Path,
) -> None:
	out_dir.mkdir(parents=True, exist_ok=True)
	match_df.to_csv(out_dir / 'loki_gamma_to_jma_match.csv', index=False)
	loki_unmatched_df.to_csv(out_dir / 'loki_unmatched_to_jma.csv', index=False)
	jma_unmatched_df.to_csv(out_dir / 'jma_unmatched_to_loki.csv', index=False)
	duplicates_df.to_csv(out_dir / 'jma_match_duplicates.csv', index=False)


def _staged_event_json(row: pd.Series) -> dict[str, Any]:
	loki_event_id = str(row['loki_event_id'])
	jma_origin_jst = _format_jst_iso(row['jma_origin_time_jst'])
	gamma_origin_utc = _format_utc_z(row['gamma_origin_time_utc'])
	jma_mag1 = _to_json_scalar(row.get('jma_mag1'))
	gamma_event_index = _to_json_scalar(row.get('gamma_event_index'))

	return {
		'event_id': loki_event_id,
		'origin_time_jst': jma_origin_jst,
		'latitude_deg': float(row['jma_latitude_deg']),
		'longitude_deg': float(row['jma_longitude_deg']),
		'depth_km': float(row['jma_depth_km']),
		'mag1': jma_mag1,
		'extra': {
			'jma': {
				'event_id': str(row['jma_event_id']),
				'origin_time': jma_origin_jst,
				'latitude_deg': float(row['jma_latitude_deg']),
				'longitude_deg': float(row['jma_longitude_deg']),
				'depth_km': float(row['jma_depth_km']),
				'mag1': jma_mag1,
			},
			'gamma': {
				'event_index': gamma_event_index,
				'origin_time_utc': gamma_origin_utc,
				'latitude_deg': float(row['gamma_latitude_deg']),
				'longitude_deg': float(row['gamma_longitude_deg']),
				'depth_km': float(row['gamma_depth_km']),
			},
			'jma_match': {
				'dt_gamma_minus_jma_sec': float(row['dt_gamma_minus_jma_sec']),
				'gamma_jma_dh_km': float(row['gamma_jma_dh_km']),
			},
		},
	}


def _write_jma_reference_event_staging(
	*,
	match_df: pd.DataFrame,
	staging_base: Path,
) -> list[str]:
	staging_base.mkdir(parents=True, exist_ok=True)
	matched = match_df.loc[match_df['match_status'].eq('matched')].copy()
	if matched.empty:
		raise RuntimeError('no matched JMA/LOKI events available for compare staging')

	written: list[str] = []
	for _, row in matched.sort_values('loki_event_id').iterrows():
		event_id = str(row['loki_event_id'])
		dst_dir = staging_base / event_id
		dst_dir.mkdir(parents=True, exist_ok=True)
		write_json(
			dst_dir / 'event.json',
			_staged_event_json(row),
			ensure_ascii=False,
			indent=2,
		)
		written.append(event_id)

	return written


def _finite_values(series: pd.Series) -> np.ndarray:
	values = pd.to_numeric(series, errors='coerce').to_numpy(dtype=float)
	return values[np.isfinite(values)]


def _require_finite(series: pd.Series, label: str) -> np.ndarray:
	values = _finite_values(series)
	if values.size == 0:
		raise RuntimeError(f'no finite values for {label}')
	return values


def _write_top_csvs(df: pd.DataFrame, out_dir: Path) -> None:
	df.sort_values('cmax', ascending=False).head(20).to_csv(
		out_dir / 'top20_cmax.csv', index=False
	)
	df.sort_values('dh_km', ascending=False).head(20).to_csv(
		out_dir / 'top20_dh_km.csv', index=False
	)
	df.assign(abs_dz_km=np.abs(df['dz_km'].astype(float))).sort_values(
		'abs_dz_km', ascending=False
	).head(20).to_csv(out_dir / 'top20_abs_dz_km.csv', index=False)


def _write_error_stats(compare_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
	required = {'event_id', 'dh_km', 'dz_km', 'dt_origin_sec', 'cmax'}
	missing = required - set(compare_df.columns)
	if missing:
		raise ValueError(f'compare_df missing columns: {sorted(missing)}')

	out_dir.mkdir(parents=True, exist_ok=True)
	df = compare_df.copy()
	for col in ['dh_km', 'dz_km', 'dt_origin_sec', 'cmax']:
		df[col] = pd.to_numeric(df[col], errors='raise').astype(float)

	plot_hist(
		_require_finite(df['dh_km'], 'dh_km'),
		title='Horizontal error dh_km (LOKI vs JMA)',
		xlabel='dh_km [km]',
		out_png=out_dir / 'dh_km_hist.png',
		bins=20,
	)
	plot_hist(
		_require_finite(df['dz_km'], 'dz_km'),
		title='Depth error dz_km (LOKI - JMA)',
		xlabel='dz_km [km]',
		out_png=out_dir / 'dz_km_hist.png',
		bins=20,
	)
	plot_hist(
		_require_finite(df['dt_origin_sec'], 'dt_origin_sec'),
		title='Origin time error dt_origin_sec (LOKI - JMA)',
		xlabel='dt_origin_sec [sec]',
		out_png=out_dir / 'dt_origin_sec_hist.png',
		bins=30,
	)

	plot_box(
		[_require_finite(df['dh_km'], 'dh_km')],
		labels=['all'],
		title='dh_km boxplot (all)',
		ylabel='dh_km [km]',
		out_png=out_dir / 'dh_km_box_all.png',
	)
	plot_box(
		[_require_finite(df['dz_km'], 'dz_km')],
		labels=['all'],
		title='dz_km boxplot (all)',
		ylabel='dz_km [km]',
		out_png=out_dir / 'dz_km_box_all.png',
	)

	df['coh_bin'] = make_coherence_bins(df, 'cmax', n_bins=4)
	labels = sorted(df['coh_bin'].dropna().unique().tolist())
	dh_by = [
		_require_finite(df.loc[df['coh_bin'] == label, 'dh_km'], f'dh_km {label}')
		for label in labels
	]
	dz_by = [
		_require_finite(df.loc[df['coh_bin'] == label, 'dz_km'], f'dz_km {label}')
		for label in labels
	]
	plot_box(
		dh_by,
		labels=labels,
		title='dh_km by coherence bins (cmax)',
		ylabel='dh_km [km]',
		out_png=out_dir / 'dh_km_box_by_cmax.png',
	)
	plot_box(
		dz_by,
		labels=labels,
		title='dz_km by coherence bins (cmax)',
		ylabel='dz_km [km]',
		out_png=out_dir / 'dz_km_box_by_cmax.png',
	)

	_write_top_csvs(df, out_dir)
	return df


def _plot_compare_map(
	*,
	compare_df: pd.DataFrame,
	plot_cfg: PlotConfig,
	out_png: Path,
	prefecture_shp: Path,
) -> None:
	df_loki_plot = make_loki_plot_df(compare_df)
	extras_lld = make_extras_lld_jma(compare_df)
	links_lld = make_links_lld(compare_df)

	mag_col = 'mag_jma' if compare_df['mag_jma'].notna().any() else 'cmax'
	size_col = mag_col

	out_png.parent.mkdir(parents=True, exist_ok=True)
	plot_events_map_and_sections(
		df_loki_plot,
		prefecture_shp=prefecture_shp,
		out_png=out_png,
		mag_col=mag_col,
		size_col=size_col,
		depth_col='depth_km',
		origin_time_col='origin_time',
		lat_col='latitude_deg',
		lon_col='longitude_deg',
		lon_range=tuple(plot_cfg.lon_range),
		lat_range=tuple(plot_cfg.lat_range),
		depth_range=tuple(plot_cfg.depth_range),
		extras_lld=extras_lld,
		links_lld=links_lld,
	)


def _ordered_unique(items: list[str]) -> list[str]:
	seen: set[str] = set()
	out: list[str] = []
	for item in items:
		if item in seen:
			continue
		seen.add(item)
		out.append(item)
	return out


def _select_detail_event_ids(
	df: pd.DataFrame,
	*,
	top_n_cmax: int,
	top_n_dh: int,
	top_n_dz: int,
) -> list[str]:
	selected: list[str] = []
	if top_n_cmax > 0:
		selected.extend(
			df.sort_values('cmax', ascending=False)
			.head(int(top_n_cmax))['event_id']
			.astype(str)
			.tolist()
		)
	if top_n_dh > 0:
		selected.extend(
			df.sort_values('dh_km', ascending=False)
			.head(int(top_n_dh))['event_id']
			.astype(str)
			.tolist()
		)
	if top_n_dz > 0:
		selected.extend(
			df.assign(abs_dz_km=np.abs(df['dz_km'].astype(float)))
			.sort_values('abs_dz_km', ascending=False)
			.head(int(top_n_dz))['event_id']
			.astype(str)
			.tolist()
		)
	return _ordered_unique(selected)


def _record_skip(skipped: dict[str, list[str]], event_id: str, reason: str) -> None:
	skipped.setdefault(event_id, []).append(reason)
	print(f'[WARN] event={event_id}: {reason}')


def _plot_detail_coherence(
	*,
	event_id: str,
	event_dir: Path,
	loki_output_dir: Path,
	header_path: Path,
	skipped: dict[str, list[str]],
) -> bool:
	ev_out_dir = loki_output_dir / event_id
	corr_path = ev_out_dir / 'corrmatrix_trial_0.npy'
	if not corr_path.is_file():
		_record_skip(skipped, event_id, 'missing corrmatrix_trial_0.npy')
		return False

	try:
		out_png = plot_loki_event_coherence_xy_overlay(
			event_dir=event_dir,
			loki_output_dir=loki_output_dir,
			header_path=header_path,
			trial=0,
		)
	except Exception as exc:  # noqa: BLE001
		_record_skip(skipped, event_id, f'coherence plot failed: {exc}')
		return False

	return out_png is not None


def _plot_detail_waveforms(
	*,
	event_dir: Path,
	loki_output_dir: Path,
	header_path: Path,
	pre_spec: DetrendBandpassSpec,
	skipped: dict[str, list[str]],
) -> bool:
	event_id = event_dir.name
	ev_out_dir = loki_output_dir / event_id
	if not sorted(ev_out_dir.glob('*_trial0.phs')):
		_record_skip(skipped, event_id, 'missing *_trial0.phs')
		return False

	try:
		plot_waveforms_with_picks_for_event(
			event_dir=event_dir,
			loki_output_dir=loki_output_dir,
			header_path=header_path,
			base_sampling_rate_hz=BASE_SAMPLING_RATE_HZ,
			components_order=COMPONENTS_ORDER,
			plot_components=PLOT_COMPONENTS,
			y_time=Y_TIME,
			pre_spec=pre_spec,
		)
	except Exception as exc:  # noqa: BLE001
		_record_skip(skipped, event_id, f'waveform plot failed: {exc}')
		return False

	return any(
		(ev_out_dir / f'waveform_with_loki_picks_{comp}.png').is_file()
		for comp in PLOT_COMPONENTS
	)


def _plot_detail_events(
	*,
	event_ids: list[str],
	jma_reference_events_dir: Path,
	events_base_dir: Path,
	loki_output_dir: Path,
	header_path: Path,
) -> tuple[list[str], dict[str, list[str]]]:
	created: list[str] = []
	skipped: dict[str, list[str]] = {}
	pre_spec = DetrendBandpassSpec()

	for event_id in event_ids:
		created_for_event = _plot_detail_coherence(
			event_id=event_id,
			event_dir=jma_reference_events_dir / event_id,
			loki_output_dir=loki_output_dir,
			header_path=header_path,
			skipped=skipped,
		)
		created_for_event = (
			_plot_detail_waveforms(
				event_dir=events_base_dir / event_id,
				loki_output_dir=loki_output_dir,
				header_path=header_path,
				pre_spec=pre_spec,
				skipped=skipped,
			)
			or created_for_event
		)
		if created_for_event:
			created.append(event_id)

	return created, skipped


def _summary_numbers(series: pd.Series, *, absolute: bool = False) -> dict[str, float]:
	values = pd.to_numeric(series, errors='coerce').to_numpy(dtype=float)
	if absolute:
		values = np.abs(values)
	values = values[np.isfinite(values)]
	if values.size == 0:
		return {'min': np.nan, 'median': np.nan, 'p90': np.nan, 'max': np.nan}
	return {
		'min': float(np.min(values)),
		'median': float(np.median(values)),
		'p90': float(np.quantile(values, 0.9)),
		'max': float(np.max(values)),
	}


def _fmt(value: float) -> str:
	if not np.isfinite(float(value)):
		return 'nan'
	return f'{float(value):.6g}'


def _format_skip_lines(skipped: dict[str, str] | dict[str, list[str]]) -> list[str]:
	lines: list[str] = []
	for event_id, reason in sorted(skipped.items()):
		reason_text = '; '.join(reason) if isinstance(reason, list) else str(reason)
		lines.append(f'- {event_id}: {reason_text}')
	return lines


@dataclass(frozen=True)
class _SummaryInputs:
	out_md: Path
	jma_event_count: int
	loki_event_count: int
	match_df: pd.DataFrame
	compare_df: pd.DataFrame
	max_match_dt_sec: float
	max_match_dh_km: float
	output_paths: list[Path]
	skipped_outputs: list[str]
	detail_event_ids: list[str]
	created_detail_event_ids: list[str]
	skipped_events: dict[str, str]
	detail_skipped_events: dict[str, list[str]]


def _match_count(match_df: pd.DataFrame, status: str) -> int:
	return int(match_df['match_status'].eq(status).sum())


def _write_summary(inputs: _SummaryInputs) -> None:
	compare_df = inputs.compare_df
	cmax = _summary_numbers(compare_df['cmax'])
	dh = _summary_numbers(compare_df['dh_km'])
	abs_dz = _summary_numbers(compare_df['dz_km'], absolute=True)
	abs_dt = _summary_numbers(compare_df['dt_origin_sec'], absolute=True)
	jma_only_count = inputs.jma_event_count - int(
		inputs.match_df.loc[
			inputs.match_df['match_status'].eq('matched'), 'jma_event_id'
		]
		.dropna()
		.nunique()
	)

	lines = [
		'# Izu2009 Loki JMA QC Summary',
		'',
		f'- jma_events_count: {inputs.jma_event_count}',
		f'- loki_event_count: {inputs.loki_event_count}',
		f'- matched_count: {_match_count(inputs.match_df, "matched")}',
		f'- loki_only_count: {_match_count(inputs.match_df, "loki_only")}',
		f'- jma_only_count: {jma_only_count}',
		'- rejected_by_distance_count: '
		f'{_match_count(inputs.match_df, "rejected_by_distance")}',
		f'- duplicate_count: {_match_count(inputs.match_df, "duplicate")}',
		f'- max_match_dt_sec: {_fmt(inputs.max_match_dt_sec)}',
		f'- max_match_dh_km: {_fmt(inputs.max_match_dh_km)}',
		f'- compare_event_count: {len(compare_df)}',
		f'- cmax_min: {_fmt(cmax["min"])}',
		f'- cmax_median: {_fmt(cmax["median"])}',
		f'- cmax_max: {_fmt(cmax["max"])}',
		f'- dh_km_median: {_fmt(dh["median"])}',
		f'- dh_km_p90: {_fmt(dh["p90"])}',
		f'- dh_km_max: {_fmt(dh["max"])}',
		f'- abs_dz_km_median: {_fmt(abs_dz["median"])}',
		f'- abs_dz_km_p90: {_fmt(abs_dz["p90"])}',
		f'- abs_dz_km_max: {_fmt(abs_dz["max"])}',
		f'- abs_dt_origin_sec_median: {_fmt(abs_dt["median"])}',
		f'- abs_dt_origin_sec_p90: {_fmt(abs_dt["p90"])}',
		f'- abs_dt_origin_sec_max: {_fmt(abs_dt["max"])}',
		'',
		'## Outputs',
		'',
	]
	lines.extend(f'- {path}' for path in inputs.output_paths)

	lines.extend(['', '## Skipped Outputs', ''])
	if inputs.skipped_outputs:
		lines.extend(f'- {item}' for item in inputs.skipped_outputs)
	else:
		lines.append('- none')

	lines.extend(['', '## Detail Event IDs', ''])
	if inputs.detail_event_ids:
		lines.extend(f'- {event_id}' for event_id in inputs.detail_event_ids)
	else:
		lines.append('- none')

	lines.extend(['', '## Detail Created Event IDs', ''])
	if inputs.created_detail_event_ids:
		lines.extend(f'- {event_id}' for event_id in inputs.created_detail_event_ids)
	else:
		lines.append('- none')

	lines.extend(['', '## Skipped Events', ''])
	skip_lines = _format_skip_lines(inputs.skipped_events)
	detail_skip_lines = _format_skip_lines(inputs.detail_skipped_events)
	if skip_lines or detail_skip_lines:
		lines.extend(skip_lines)
		lines.extend(detail_skip_lines)
	else:
		lines.append('- none')

	inputs.out_md.parent.mkdir(parents=True, exist_ok=True)
	inputs.out_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description='Create QC plots for Izu2009 JMA-referenced Loki outputs.'
	)
	parser.add_argument('--jma-events-csv', type=Path, default=JMA_EVENTS_CSV)
	parser.add_argument('--events-base-dir', type=Path, default=EVENTS_BASE_DIR)
	parser.add_argument('--loki-output-dir', type=Path, default=LOKI_OUTPUT_DIR)
	parser.add_argument('--header-path', type=Path, default=HEADER_PATH)
	parser.add_argument(
		'--plot-config-yaml',
		type=Path,
		default=None,
		help=(
			'Path to plot_config.yaml. If omitted, use repo-local '
			'proc/izu2009/loki/config/plot_config.yaml, then '
			'plot_config_izu2009.yaml, then /workspace/data/config/plot_config.yaml.'
		),
	)
	parser.add_argument(
		'--plot-preset',
		'--plot-config-preset',
		dest='plot_preset',
		default=PLOT_CONFIG_PRESET,
		help='Preset name in plot_config.yaml. Default: izu_default.',
	)
	parser.add_argument(
		'--prefecture-shp',
		type=Path,
		default=None,
		help=(
			'Optional prefecture shapefile for map plots. If omitted, the '
			'standard /workspace/data path is used when present; otherwise '
			'shapefile-dependent plots are skipped.'
		),
	)
	parser.add_argument('--event-glob', default=EVENT_GLOB)
	parser.add_argument('--max-match-dt-sec', type=float, default=MAX_MATCH_DT_SEC)
	parser.add_argument('--max-match-dh-km', type=float, default=MAX_MATCH_DH_KM)
	parser.add_argument('--top-n-cmax', type=int, default=TOP_N_CMAX)
	parser.add_argument('--top-n-dh', type=int, default=TOP_N_DH)
	parser.add_argument('--top-n-dz', type=int, default=TOP_N_DZ)
	parser.add_argument('--skip-detail', action='store_true')
	return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
	"""Run the Izu2009 Loki/JMA QC wrapper."""
	args = _parse_args(argv)

	jma_events_csv = Path(args.jma_events_csv)
	events_base_dir = Path(args.events_base_dir)
	loki_output_dir = Path(args.loki_output_dir)
	header_path = Path(args.header_path)
	plot_config_yaml = _resolve_plot_config_yaml(args.plot_config_yaml)
	prefecture_shp = _resolve_prefecture_shp(args.prefecture_shp)
	qc_dir = loki_output_dir / 'qc'
	jma_match_dir = qc_dir / 'jma_match'
	jma_reference_events_dir = qc_dir / 'jma_reference_events'
	error_stats_dir = qc_dir / 'error_stats'

	event_dirs, skipped_events = _validate_inputs(
		jma_events_csv=jma_events_csv,
		events_base_dir=events_base_dir,
		loki_output_dir=loki_output_dir,
		header_path=header_path,
		plot_config_yaml=plot_config_yaml,
		prefecture_shp=prefecture_shp,
		event_glob=str(args.event_glob),
	)
	qc_dir.mkdir(parents=True, exist_ok=True)
	error_stats_dir.mkdir(parents=True, exist_ok=True)

	jma_df = _load_jma_events_csv(jma_events_csv)
	gamma_df = _load_gamma_reference_events(event_dirs)
	match_df, loki_unmatched_df, jma_unmatched_df, duplicates_df = (
		_build_jma_loki_match_table(
			jma_df=jma_df,
			gamma_df=gamma_df,
			max_match_dt_sec=float(args.max_match_dt_sec),
			max_match_dh_km=float(args.max_match_dh_km),
		)
	)
	_write_match_outputs(
		match_df=match_df,
		loki_unmatched_df=loki_unmatched_df,
		jma_unmatched_df=jma_unmatched_df,
		duplicates_df=duplicates_df,
		out_dir=jma_match_dir,
	)
	matched_loki_event_ids = _write_jma_reference_event_staging(
		match_df=match_df,
		staging_base=jma_reference_events_dir,
	)

	plot_cfg = load_config(PlotConfig, plot_config_yaml, str(args.plot_preset))
	plot_output_paths: list[Path] = []
	skipped_outputs: list[str] = []

	if prefecture_shp is not None:
		quickcheck_png = qc_dir / 'loki_catalogue_quickcheck.png'
		plot_loki_results_quickcheck(
			loki_output_dir=loki_output_dir,
			prefecture_shp=prefecture_shp,
			out_png=quickcheck_png,
			lat_range=tuple(plot_cfg.lat_range),
			lon_range=tuple(plot_cfg.lon_range),
			depth_range=tuple(plot_cfg.depth_range),
		)
		plot_output_paths.append(quickcheck_png)
	else:
		skipped_outputs.append(
			'loki_catalogue_quickcheck.png: --prefecture-shp not provided'
		)

	compare_df = build_compare_df(
		base_input_dir=jma_reference_events_dir,
		loki_output_dir=loki_output_dir,
		header_path=header_path,
		event_glob=str(args.event_glob),
		allowed_event_ids=set(matched_loki_event_ids),
	)
	compare_csv = qc_dir / 'compare_jma_vs_loki.csv'
	compare_df.to_csv(compare_csv, index=False)

	if prefecture_shp is not None:
		compare_png = qc_dir / 'loki_vs_jma.png'
		_plot_compare_map(
			compare_df=compare_df,
			plot_cfg=plot_cfg,
			out_png=compare_png,
			prefecture_shp=prefecture_shp,
		)
		plot_output_paths.append(compare_png)
	else:
		skipped_outputs.append('loki_vs_jma.png: --prefecture-shp not provided')

	stats_df = _write_error_stats(compare_df, error_stats_dir)
	detail_event_ids = _select_detail_event_ids(
		stats_df,
		top_n_cmax=int(args.top_n_cmax),
		top_n_dh=int(args.top_n_dh),
		top_n_dz=int(args.top_n_dz),
	)

	created_detail_event_ids: list[str] = []
	detail_skipped_events: dict[str, list[str]] = {}
	if args.skip_detail:
		detail_skipped_events = {
			event_id: ['detail plotting skipped by --skip-detail']
			for event_id in detail_event_ids
		}
	else:
		created_detail_event_ids, detail_skipped_events = _plot_detail_events(
			event_ids=detail_event_ids,
			jma_reference_events_dir=jma_reference_events_dir,
			events_base_dir=events_base_dir,
			loki_output_dir=loki_output_dir,
			header_path=header_path,
		)

	output_paths = [
		qc_dir / 'compare_jma_vs_loki.csv',
		qc_dir / 'loki_jma_qc_summary.md',
		jma_match_dir / 'loki_gamma_to_jma_match.csv',
		jma_match_dir / 'loki_unmatched_to_jma.csv',
		jma_match_dir / 'jma_unmatched_to_loki.csv',
		jma_match_dir / 'jma_match_duplicates.csv',
		error_stats_dir / 'dh_km_hist.png',
		error_stats_dir / 'dz_km_hist.png',
		error_stats_dir / 'dt_origin_sec_hist.png',
		error_stats_dir / 'dh_km_box_all.png',
		error_stats_dir / 'dz_km_box_all.png',
		error_stats_dir / 'dh_km_box_by_cmax.png',
		error_stats_dir / 'dz_km_box_by_cmax.png',
		error_stats_dir / 'top20_cmax.csv',
		error_stats_dir / 'top20_dh_km.csv',
		error_stats_dir / 'top20_abs_dz_km.csv',
	]
	output_paths[1:1] = plot_output_paths
	_write_summary(
		_SummaryInputs(
			out_md=qc_dir / 'loki_jma_qc_summary.md',
			jma_event_count=len(jma_df),
			loki_event_count=len(event_dirs),
			match_df=match_df,
			compare_df=stats_df,
			max_match_dt_sec=float(args.max_match_dt_sec),
			max_match_dh_km=float(args.max_match_dh_km),
			output_paths=output_paths,
			skipped_outputs=skipped_outputs,
			detail_event_ids=detail_event_ids,
			created_detail_event_ids=created_detail_event_ids,
			skipped_events=skipped_events,
			detail_skipped_events=detail_skipped_events,
		)
	)


if __name__ == '__main__':
	main()
