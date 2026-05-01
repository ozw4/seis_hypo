"""QC wrapper for Izu2009 GaMMA-referenced Loki EqT outputs."""

from __future__ import annotations

import argparse
import sys
import tempfile
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

from common.json_io import read_json, write_json  # noqa: E402
from common.load_config import load_config  # noqa: E402
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

EVENTS_BASE_DIR = _REPO_ROOT / 'proc/izu2009/loki/events_from_gamma'
LOKI_OUTPUT_DIR = _REPO_ROOT / 'proc/izu2009/loki/output_eqt_gamma'
HEADER_PATH = _REPO_ROOT / 'proc/izu2009/loki/traveltime/db/header.hdr'
QC_DIR = LOKI_OUTPUT_DIR / 'qc'
ERROR_STATS_DIR = QC_DIR / 'error_stats'

PLOT_CONFIG_YAML = Path('/workspace/data/config/plot_config.yaml')
PLOT_CONFIG_PRESET = 'izu_default'
PREFECTURE_SHP = Path('/workspace/data/N03-20240101_GML/N03-20240101_prefecture.shp')

EVENT_GLOB = '[0-9]*'
TOP_N_CMAX = 10
TOP_N_DH = 5
TOP_N_DZ = 5
BASE_SAMPLING_RATE_HZ = 100
COMPONENTS_ORDER = ('U', 'N', 'E')
PLOT_COMPONENTS = ('U', 'N', 'E')
Y_TIME = 'relative'

REFERENCE_KEYS = ('latitude_deg', 'longitude_deg', 'depth_km')


def _require_dir(path: Path, label: str) -> None:
	if not path.is_dir():
		raise FileNotFoundError(f'{label} not found: {path}')


def _require_file(path: Path, label: str) -> None:
	if not path.is_file():
		raise FileNotFoundError(f'{label} not found: {path}')


def _event_json_path(event_dir: Path) -> Path:
	return event_dir / 'event.json'


def _load_event_json_file(event_dir: Path) -> dict[str, Any]:
	path = _event_json_path(event_dir)
	if not path.is_file():
		raise FileNotFoundError(f'event.json not found: {path}')
	obj = read_json(path)
	if not isinstance(obj, dict):
		raise TypeError(f'event.json must contain an object: {path}')
	return obj


def _reference_sources(ev: dict[str, Any], event_json_path: Path) -> dict[str, Any]:
	extra = ev.get('extra')
	if extra is not None and not isinstance(extra, dict):
		raise TypeError(f'event.json extra must be an object: {event_json_path}')

	extra_dict = extra if isinstance(extra, dict) else {}
	gamma = extra_dict.get('gamma')
	if gamma is not None and not isinstance(gamma, dict):
		raise TypeError(f'event.json extra.gamma must be an object: {event_json_path}')

	gamma_dict = gamma if isinstance(gamma, dict) else {}
	out: dict[str, Any] = {}
	for key in REFERENCE_KEYS:
		value = ev.get(key)
		if value is None:
			value = extra_dict.get(key)
		if value is None:
			value = gamma_dict.get(key)
		if value is None:
			raise ValueError(
				f'event.json missing reference key {key!r}: {event_json_path} '
				'(need top-level, extra, or extra.gamma)'
			)
		out[key] = value
	return out


def ensure_gamma_reference_keys_for_existing_loki_viz(
	event_dirs: list[Path],
) -> int:
	"""Fill reference hypocenter keys expected by existing Loki viz helpers."""
	changed_count = 0

	for event_dir in event_dirs:
		event_json_path = _event_json_path(event_dir)
		ev = _load_event_json_file(event_dir)
		source = _reference_sources(ev, event_json_path)

		extra = ev.get('extra')
		if extra is None:
			extra = {}
			ev['extra'] = extra
		elif not isinstance(extra, dict):
			raise TypeError(f'event.json extra must be an object: {event_json_path}')

		changed = False
		for key in REFERENCE_KEYS:
			if ev.get(key) is None:
				ev[key] = source[key]
				changed = True
			if extra.get(key) is None:
				extra[key] = source[key]
				changed = True

		if changed:
			write_json(event_json_path, ev, ensure_ascii=False, indent=2)
			changed_count += 1

	return changed_count


def _event_dirs(path: Path, event_glob: str) -> list[Path]:
	return sorted(p for p in path.glob(event_glob) if p.is_dir())


def _event_dirs_with_loc(event_dirs: list[Path], loki_output_dir: Path) -> list[Path]:
	out: list[Path] = []
	for event_dir in event_dirs:
		ev_out_dir = loki_output_dir / event_dir.name
		if any(ev_out_dir.glob('*.loc')):
			out.append(event_dir)
	return out


def _validate_inputs(
	*,
	events_base_dir: Path,
	loki_output_dir: Path,
	header_path: Path,
	event_glob: str,
) -> tuple[list[Path], dict[str, str]]:
	_require_dir(events_base_dir, 'events_from_gamma')
	_require_dir(loki_output_dir, 'output_eqt_gamma')
	_require_file(loki_output_dir / 'catalogue', 'Loki catalogue')
	_require_file(header_path, 'Loki header')

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


def _write_compare_event_json_staging(
	event_dirs: list[Path], staging_base: Path
) -> None:
	for event_dir in event_dirs:
		dst_dir = staging_base / event_dir.name
		dst_dir.mkdir(parents=True, exist_ok=False)

		ev = _load_event_json_file(event_dir)
		ev['event_id'] = event_dir.name
		write_json(dst_dir / 'event.json', ev, ensure_ascii=False, indent=2)


def _build_compare_df_with_staged_event_ids(
	*,
	event_dirs: list[Path],
	loki_output_dir: Path,
	header_path: Path,
	event_glob: str,
) -> pd.DataFrame:
	allowed_event_ids = {p.name for p in event_dirs}
	with tempfile.TemporaryDirectory(prefix='izu2009_loki_qc_') as tmp:
		staging_base = Path(tmp)
		_write_compare_event_json_staging(event_dirs, staging_base)
		return build_compare_df(
			base_input_dir=staging_base,
			loki_output_dir=loki_output_dir,
			header_path=header_path,
			event_glob=event_glob,
			allowed_event_ids=allowed_event_ids,
		)


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
		title='Horizontal error dh_km (LOKI vs GaMMA)',
		xlabel='dh_km [km]',
		out_png=out_dir / 'dh_km_hist.png',
		bins=20,
	)
	plot_hist(
		_require_finite(df['dz_km'], 'dz_km'),
		title='Depth error dz_km (LOKI - GaMMA)',
		xlabel='dz_km [km]',
		out_png=out_dir / 'dz_km_hist.png',
		bins=20,
	)
	plot_hist(
		_require_finite(df['dt_origin_sec'], 'dt_origin_sec'),
		title='Origin time error dt_origin_sec (LOKI - GaMMA)',
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


def _make_gamma_extras_lld(compare_df: pd.DataFrame) -> list[dict]:
	extras_lld = make_extras_lld_jma(compare_df)
	for item in extras_lld:
		if item.get('label') == 'JMA':
			item['label'] = 'GaMMA'
	return extras_lld


def _make_gamma_links_lld(compare_df: pd.DataFrame) -> list[dict]:
	links_lld = make_links_lld(compare_df)
	for item in links_lld:
		if item.get('label') == 'LOKI\u2192JMA':
			item['label'] = 'LOKI-GaMMA'
	return links_lld


def _plot_compare_map(
	*,
	compare_df: pd.DataFrame,
	plot_cfg: PlotConfig,
	out_png: Path,
	prefecture_shp: Path,
) -> None:
	df_loki_plot = make_loki_plot_df(compare_df)
	extras_lld = _make_gamma_extras_lld(compare_df)
	links_lld = _make_gamma_links_lld(compare_df)

	out_png.parent.mkdir(parents=True, exist_ok=True)
	plot_events_map_and_sections(
		df_loki_plot,
		prefecture_shp=prefecture_shp,
		out_png=out_png,
		mag_col='cmax',
		size_col='cmax',
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
	events_base_dir: Path,
	loki_output_dir: Path,
	header_path: Path,
) -> tuple[list[str], dict[str, list[str]]]:
	created: list[str] = []
	skipped: dict[str, list[str]] = {}
	pre_spec = DetrendBandpassSpec()

	for event_id in event_ids:
		event_dir = events_base_dir / event_id
		created_for_event = _plot_detail_coherence(
			event_id=event_id,
			event_dir=event_dir,
			loki_output_dir=loki_output_dir,
			header_path=header_path,
			skipped=skipped,
		)
		created_for_event = (
			_plot_detail_waveforms(
				event_dir=event_dir,
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
	target_event_count: int
	compare_df: pd.DataFrame
	patched_event_json_count: int
	detail_event_ids: list[str]
	created_detail_event_ids: list[str]
	skipped_events: dict[str, str]
	detail_skipped_events: dict[str, list[str]]


def _write_summary(inputs: _SummaryInputs) -> None:
	compare_df = inputs.compare_df
	cmax = _summary_numbers(compare_df['cmax'])
	dh = _summary_numbers(compare_df['dh_km'])
	abs_dz = _summary_numbers(compare_df['dz_km'], absolute=True)
	abs_dt = _summary_numbers(compare_df['dt_origin_sec'], absolute=True)

	lines = [
		'# Izu2009 Loki GaMMA QC Summary',
		'',
		f'- target_event_count: {inputs.target_event_count}',
		f'- compare_event_count: {len(compare_df)}',
		f'- patched_event_json_count: {inputs.patched_event_json_count}',
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
		'## Detail Event IDs',
		'',
	]
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
		description='Create QC plots for Izu2009 GaMMA-referenced Loki outputs.'
	)
	parser.add_argument('--events-base-dir', type=Path, default=EVENTS_BASE_DIR)
	parser.add_argument('--loki-output-dir', type=Path, default=LOKI_OUTPUT_DIR)
	parser.add_argument('--header-path', type=Path, default=HEADER_PATH)
	parser.add_argument('--plot-config-yaml', type=Path, default=PLOT_CONFIG_YAML)
	parser.add_argument('--plot-config-preset', default=PLOT_CONFIG_PRESET)
	parser.add_argument('--prefecture-shp', type=Path, default=PREFECTURE_SHP)
	parser.add_argument('--event-glob', default=EVENT_GLOB)
	parser.add_argument('--top-n-cmax', type=int, default=TOP_N_CMAX)
	parser.add_argument('--top-n-dh', type=int, default=TOP_N_DH)
	parser.add_argument('--top-n-dz', type=int, default=TOP_N_DZ)
	parser.add_argument('--skip-detail', action='store_true')
	return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
	"""Run the Izu2009 Loki/GaMMA QC wrapper."""
	args = _parse_args(argv)

	events_base_dir = Path(args.events_base_dir)
	loki_output_dir = Path(args.loki_output_dir)
	header_path = Path(args.header_path)
	plot_config_yaml = Path(args.plot_config_yaml)
	prefecture_shp = Path(args.prefecture_shp)
	qc_dir = loki_output_dir / 'qc'
	error_stats_dir = qc_dir / 'error_stats'

	_require_file(plot_config_yaml, 'plot config YAML')
	_require_file(prefecture_shp, 'prefecture shapefile')

	event_dirs, skipped_events = _validate_inputs(
		events_base_dir=events_base_dir,
		loki_output_dir=loki_output_dir,
		header_path=header_path,
		event_glob=str(args.event_glob),
	)
	qc_dir.mkdir(parents=True, exist_ok=True)
	error_stats_dir.mkdir(parents=True, exist_ok=True)

	patched_event_json_count = ensure_gamma_reference_keys_for_existing_loki_viz(
		event_dirs
	)

	plot_cfg = load_config(PlotConfig, plot_config_yaml, str(args.plot_config_preset))

	plot_loki_results_quickcheck(
		loki_output_dir=loki_output_dir,
		prefecture_shp=prefecture_shp,
		out_png=qc_dir / 'loki_catalogue_quickcheck.png',
		lat_range=tuple(plot_cfg.lat_range),
		lon_range=tuple(plot_cfg.lon_range),
		depth_range=tuple(plot_cfg.depth_range),
	)

	compare_df = _build_compare_df_with_staged_event_ids(
		event_dirs=event_dirs,
		loki_output_dir=loki_output_dir,
		header_path=header_path,
		event_glob=str(args.event_glob),
	)
	compare_csv = qc_dir / 'compare_gamma_vs_loki.csv'
	compare_df.to_csv(compare_csv, index=False)

	_plot_compare_map(
		compare_df=compare_df,
		plot_cfg=plot_cfg,
		out_png=qc_dir / 'loki_vs_gamma.png',
		prefecture_shp=prefecture_shp,
	)

	stats_df = _write_error_stats(compare_df, error_stats_dir)
	detail_event_ids = _select_detail_event_ids(
		stats_df,
		top_n_cmax=int(args.top_n_cmax),
		top_n_dh=int(args.top_n_dh),
		top_n_dz=int(args.top_n_dz),
	)

	created_detail_event_ids: list[str] = []
	detail_skipped_events: dict[str, list[str]] = {}
	if not args.skip_detail:
		created_detail_event_ids, detail_skipped_events = _plot_detail_events(
			event_ids=detail_event_ids,
			events_base_dir=events_base_dir,
			loki_output_dir=loki_output_dir,
			header_path=header_path,
		)

	_write_summary(
		_SummaryInputs(
			out_md=qc_dir / 'loki_qc_summary.md',
			target_event_count=len(event_dirs),
			compare_df=stats_df,
			patched_event_json_count=patched_event_json_count,
			detail_event_ids=[] if args.skip_detail else detail_event_ids,
			created_detail_event_ids=created_detail_event_ids,
			skipped_events=skipped_events,
			detail_skipped_events=detail_skipped_events,
		)
	)


if __name__ == '__main__':
	main()
