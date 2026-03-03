from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def _find_src_dir(start: Path) -> Path:
	p = start.resolve()
	for d in [p] + list(p.parents):
		src = d / 'src'
		if (src / 'common' / 'yaml_config.py').is_file():
			return src
	raise FileNotFoundError(
		'could not locate repo src/ (expected src/common/yaml_config.py)'
	)


_SRC = _find_src_dir(Path(__file__).parent)
if str(_SRC) not in sys.path:
	sys.path.insert(0, str(_SRC))

from common.yaml_config import read_yaml_mapping  # noqa: E402

_RECIPE_KEYS = {'dataset_dir', 'outputs_dir', 'paths', 'selection', 'output'}


def _req(d: dict[str, Any], key: str) -> Any:
	if key not in d:
		raise KeyError(f'missing required key: {key}')
	return d[key]


def _resolve_path(p: str, *, base: Path) -> Path:
	pp = Path(str(p)).expanduser()
	if not pp.is_absolute():
		pp = base / pp
	return pp.resolve()


def _load_dataset_meta(dataset_dir: Path) -> dict[str, Any]:
	meta_path = dataset_dir / 'dataset_meta.json'
	if not meta_path.is_file():
		raise FileNotFoundError(f'missing: {meta_path}')
	obj = json.loads(meta_path.read_text(encoding='utf-8'))
	if not isinstance(obj, dict):
		raise ValueError('dataset_meta.json must be a JSON object')
	return obj


def _resolve_sim_yaml_name(dataset_dir: Path, meta: dict[str, Any], value: Any) -> str:
	if value is not None and str(value).strip().lower() != 'auto':
		name = Path(str(value)).name
		if not (dataset_dir / 'provenance' / name).is_file():
			raise FileNotFoundError(f'missing: {dataset_dir / "provenance" / name}')
		return name

	prov_dir = dataset_dir / 'provenance'
	if not prov_dir.is_dir():
		raise FileNotFoundError(f'missing: {prov_dir}')

	cands = sorted([p for p in prov_dir.glob('*.yaml') if p.is_file()])
	if not cands:
		raise FileNotFoundError(f'no *.yaml found in: {prov_dir}')

	cfg_path = meta.get('provenance', {}).get('config_path', None)
	if cfg_path:
		base = Path(str(cfg_path)).name
		for p in cands:
			if p.name == base:
				return p.name

	if len(cands) == 1:
		return cands[0].name

	raise ValueError(
		'multiple provenance yamls found. Set paths.sim_yaml explicitly. '
		f'candidates={[p.name for p in cands]}'
	)


def _resolve_receiver_geometry_name(
	dataset_dir: Path, meta: dict[str, Any], value: Any
) -> str:
	if value is not None and str(value).strip().lower() != 'auto':
		name = Path(str(value)).name
		if not (dataset_dir / 'geometry' / name).is_file():
			raise FileNotFoundError(f'missing: {dataset_dir / "geometry" / name}')
		return name

	rel = meta.get('geometry', {}).get('merged_npy_rel', None)
	if rel is None or str(rel).strip() == '':
		raise ValueError('dataset_meta.json missing geometry.merged_npy_rel')

	name = Path(str(rel)).name
	if not (dataset_dir / 'geometry' / name).is_file():
		raise FileNotFoundError(f'missing: {dataset_dir / "geometry" / name}')
	return name


def _load_receiver_catalog(dataset_dir: Path, meta: dict[str, Any]) -> pd.DataFrame:
	rel = meta.get('optional', {}).get('receiver_catalog_csv_rel', None)
	if rel is None or str(rel).strip() == '':
		raise ValueError('dataset_meta.json missing optional.receiver_catalog_csv_rel')
	path = dataset_dir / str(rel)
	if not path.is_file():
		raise FileNotFoundError(f'missing: {path}')

	df = pd.read_csv(path)
	required = {
		'receiver_index',
		'station_code',
		'is_das',
		'component_recv_id',
		'component_index',
		'x_m',
		'y_m',
		'z_m',
	}
	missing = sorted(required - set(df.columns))
	if missing:
		raise ValueError(f'receiver catalog missing columns: {missing}')

	ridx = df['receiver_index']
	if ridx.isna().any():
		raise ValueError('receiver catalog receiver_index contains NaN')
	if (ridx.map(lambda v: isinstance(v, bool))).any():
		raise ValueError('receiver catalog receiver_index contains bool')
	if not pd.api.types.is_integer_dtype(ridx):
		raise ValueError('receiver catalog receiver_index must be integer dtype')

	df2 = df.copy()
	df2['receiver_index'] = ridx.astype(int)
	df2 = df2.sort_values('receiver_index').reset_index(drop=True)

	expected = np.arange(len(df2), dtype=int)
	got = df2['receiver_index'].to_numpy(dtype=int)
	if not np.array_equal(got, expected):
		raise ValueError(
			'receiver catalog receiver_index must be contiguous 0..N-1 in sorted order: '
			f'got min={int(got.min())} max={int(got.max())} expected_len={len(df2)}'
		)

	codes = df2['station_code'].astype(str)
	if (codes.str.len() == 0).any():
		raise ValueError('receiver catalog station_code contains empty string')

	return df2


def _add_local_indices(df_sorted: pd.DataFrame) -> pd.DataFrame:
	df = df_sorted.copy()
	is_das_code = df['station_code'].astype(str).str.upper().str.startswith('D')
	df['is_das_code'] = is_das_code

	surface_pos = np.where(~is_das_code.to_numpy(dtype=bool))[0]
	das_pos = np.where(is_das_code.to_numpy(dtype=bool))[0]

	surface_index = np.full(len(df), -1, dtype=int)
	das_index = np.full(len(df), -1, dtype=int)
	surface_index[surface_pos] = np.arange(surface_pos.size, dtype=int)
	das_index[das_pos] = np.arange(das_pos.size, dtype=int)

	df['surface_index'] = surface_index
	df['das_index'] = das_index
	return df


def _parse_range_2(value: Any, *, label: str) -> tuple[float, float]:
	if not isinstance(value, (list, tuple)) or len(value) != 2:
		raise ValueError(f'{label} must be [min,max], got: {value!r}')
	a = float(value[0])
	b = float(value[1])
	if b < a:
		raise ValueError(f'{label} must satisfy max>=min, got: {value!r}')
	return a, b


def _select_surface_indices(df: pd.DataFrame, spec: dict[str, Any]) -> list[int]:
	if not bool(spec.get('enabled', True)):
		return []

	recv_id = str(_req(spec, 'recv_id'))
	comp = spec.get('component_indices')

	df_s = df[
		(~df['is_das_code']) & (df['component_recv_id'].astype(str) == recv_id)
	].copy()
	if df_s.empty:
		raise ValueError(f'no surface receivers found for recv_id={recv_id!r}')

	if isinstance(comp, str) and comp.strip().lower() == 'all':
		pass
	else:
		if not isinstance(comp, list):
			raise ValueError(
				"selection.surface.component_indices must be 'all' or list[int]"
			)
		ci = pd.Series(comp)
		if (ci.map(lambda v: isinstance(v, bool))).any() or not ci.map(
			lambda v: isinstance(v, int)
		).all():
			raise ValueError('selection.surface.component_indices entries must be int')
		df_s = df_s[df_s['component_index'].astype(int).isin([int(v) for v in comp])]
		if df_s.empty:
			raise ValueError('surface selection results in 0 stations')

	return sorted(set(df_s['surface_index'].astype(int).tolist()))


def _select_das_indices(df: pd.DataFrame, spec: dict[str, Any]) -> list[int]:
	wells = _req(spec, 'wells')
	if not isinstance(wells, list) or not wells:
		raise ValueError('selection.das.wells must be a non-empty list')

	zrng = spec.get('depth_range_m')
	zmin, zmax = (None, None)
	if zrng is not None:
		zmin, zmax = _parse_range_2(zrng, label='selection.das.depth_range_m')

	decimate = int(spec.get('decimate', 1))
	if decimate < 1:
		raise ValueError('selection.das.decimate must be >= 1')

	all_selected: list[int] = []
	for w in wells:
		wid = str(w)
		df_w = df[
			(df['is_das_code']) & (df['component_recv_id'].astype(str) == wid)
		].copy()
		if df_w.empty:
			raise ValueError(f'no DAS receivers found for well recv_id={wid!r}')

		if zmin is not None and zmax is not None:
			z = df_w['z_m'].astype(float)
			df_w = df_w[(z >= float(zmin)) & (z <= float(zmax))].copy()

		cidx = df_w['component_index'].astype(int)
		df_w = df_w[(cidx % int(decimate)) == 0].copy()

		if df_w.empty:
			raise ValueError(f'DAS selection results in 0 stations for well={wid!r}')

		all_selected.extend(df_w['das_index'].astype(int).tolist())

	return sorted(set(all_selected))


def _split_defaults(recipe: dict[str, Any]) -> dict[str, Any]:
	defaults: dict[str, Any] = {}
	for k, v in recipe.items():
		if k in _RECIPE_KEYS:
			continue
		if k == 'station_subset':
			raise ValueError(
				'do not set station_subset in recipe. It is generated from selection.'
			)
		defaults[k] = v
	return defaults


def build_config(recipe_path: Path) -> tuple[dict[str, Any], Path]:
	recipe = read_yaml_mapping(recipe_path)
	base = recipe_path.parent

	dataset_dir = _resolve_path(str(_req(recipe, 'dataset_dir')), base=base)
	if not dataset_dir.is_dir():
		raise FileNotFoundError(
			f'dataset_dir must be an existing directory: {dataset_dir}'
		)

	meta = _load_dataset_meta(dataset_dir)

	paths = _req(recipe, 'paths')
	if not isinstance(paths, dict):
		raise ValueError('paths must be a mapping')

	template_cmd = _resolve_path(str(_req(paths, 'template_cmd')), base=base)
	hypoinverse_exe = _resolve_path(str(_req(paths, 'hypoinverse_exe')), base=base)
	if not template_cmd.is_file():
		raise FileNotFoundError(f'template_cmd not found: {template_cmd}')
	if not hypoinverse_exe.is_file():
		raise FileNotFoundError(f'hypoinverse_exe not found: {hypoinverse_exe}')
	if not template_cmd.is_absolute() or not hypoinverse_exe.is_absolute():
		raise ValueError(
			'template_cmd and hypoinverse_exe must resolve to absolute paths'
		)

	sim_yaml_name = _resolve_sim_yaml_name(
		dataset_dir, meta, paths.get('sim_yaml', 'auto')
	)
	recv_geom_name = _resolve_receiver_geometry_name(
		dataset_dir, meta, paths.get('receiver_geometry', 'auto')
	)

	df_cat = _add_local_indices(_load_receiver_catalog(dataset_dir, meta))

	selection = _req(recipe, 'selection')
	if not isinstance(selection, dict):
		raise ValueError('selection must be a mapping')

	surface_spec = selection.get('surface', {}) or {}
	if not isinstance(surface_spec, dict):
		raise ValueError('selection.surface must be a mapping')

	das_spec = _req(selection, 'das')
	if not isinstance(das_spec, dict):
		raise ValueError('selection.das must be a mapping')

	surface_indices = _select_surface_indices(df_cat, surface_spec)
	das_indices = _select_das_indices(df_cat, das_spec)

	if len(surface_indices) == 0 and len(das_indices) == 0:
		raise ValueError('station_subset selects no stations')

	defaults = _split_defaults(recipe)

	cfg: dict[str, Any] = {}
	cfg.update(defaults)

	cfg['dataset_dir'] = str(dataset_dir)
	cfg['sim_yaml'] = sim_yaml_name
	cfg['outputs_dir'] = str(_req(recipe, 'outputs_dir'))
	cfg['template_cmd'] = str(template_cmd)
	cfg['hypoinverse_exe'] = str(hypoinverse_exe)
	cfg['receiver_geometry'] = recv_geom_name
	cfg['station_subset'] = {
		'surface_indices': surface_indices,
		'das_indices': das_indices,
	}
	if 'event_subsample' not in cfg:
		cfg['event_subsample'] = {'stride_ijk': [2, 2, 2]}
	if 'event_filter' not in cfg:
		cfg['event_filter'] = {'z_range_m': [500.0, None]}
	out = _req(recipe, 'output')
	if not isinstance(out, dict):
		raise ValueError('output must be a mapping')
	out_path = _resolve_path(str(_req(out, 'config_yaml_path')), base=base)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	return cfg, out_path


def parse_args(argv: list[str]) -> argparse.Namespace:
	p = argparse.ArgumentParser(
		prog='make_synth_eval_config_from_recipe',
		description='Build seis_hypo synth_hypoinverse_eval config YAML from a single-file recipe YAML.',
	)
	p.add_argument('recipe', type=Path, help='recipe YAML path')
	return p.parse_args(argv)


def main(argv: list[str]) -> None:
	args = parse_args(argv)
	recipe_path = args.recipe.expanduser().resolve()
	cfg, out_path = build_config(recipe_path)
	out_path.write_text(
		yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
		encoding='utf-8',
	)
	sn = cfg['station_subset']
	print(f'[OK] wrote: {out_path}')
	print(
		f'[OK] surface selected: {len(sn["surface_indices"])}  das selected: {len(sn["das_indices"])}'
	)


if __name__ == '__main__':
	main(sys.argv[1:])
