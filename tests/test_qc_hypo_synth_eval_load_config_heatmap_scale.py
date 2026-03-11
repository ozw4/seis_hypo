from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from qc.hypo.synth_eval import load_config


def _write_config(tmp_path: Path, *, heatmap_scale: dict[str, object]) -> Path:
	cfg_path = tmp_path / 'qc_config.yaml'
	cfg = {
		'dataset_dir': str((tmp_path / 'dataset').resolve()),
		'outputs_dir': 'run1',
		'receiver_geometry': 'geom.npy',
		'heatmap': {
			'enabled': True,
			'scale': heatmap_scale,
		},
	}
	cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
	return cfg_path


def test_load_config_reads_heatmap_explicit_scale(tmp_path: Path) -> None:
	cfg_path = _write_config(
		tmp_path,
		heatmap_scale={
			'percentile': 99.0,
			'global_across_slices': True,
			'dz_symmetric': True,
			'vmin': 0.0,
			'vmax': 500.0,
		},
	)

	cfg = load_config(cfg_path)

	assert cfg.heatmap.scale.vmin == 0.0
	assert cfg.heatmap.scale.vmax == 500.0


@pytest.mark.parametrize(
	'heatmap_scale',
	[
		{
			'percentile': 99.0,
			'global_across_slices': True,
			'dz_symmetric': True,
			'vmin': 0.0,
		},
		{
			'percentile': 99.0,
			'global_across_slices': True,
			'dz_symmetric': True,
			'vmax': 500.0,
		},
	],
)
def test_load_config_rejects_half_specified_heatmap_scale(
	tmp_path: Path, heatmap_scale: dict[str, object]
) -> None:
	cfg_path = _write_config(tmp_path, heatmap_scale=heatmap_scale)

	with pytest.raises(
		ValueError,
		match=re.escape(
			'heatmap.scale.vmin and heatmap.scale.vmax must be both specified or both omitted'
		),
	):
		load_config(cfg_path)


@pytest.mark.parametrize(
	('vmin', 'vmax'),
	[
		(1.0, 1.0),
		(2.0, 1.0),
	],
)
def test_load_config_rejects_non_increasing_heatmap_scale(
	tmp_path: Path, vmin: float, vmax: float
) -> None:
	cfg_path = _write_config(
		tmp_path,
		heatmap_scale={
			'percentile': 99.0,
			'global_across_slices': True,
			'dz_symmetric': True,
			'vmin': vmin,
			'vmax': vmax,
		},
	)

	with pytest.raises(
		ValueError, match=re.escape('heatmap.scale.vmax must be > heatmap.scale.vmin')
	):
		load_config(cfg_path)
