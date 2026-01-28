from __future__ import annotations

from pathlib import Path

import pytest

from hypo.synth_eval.pipeline import build_synth_layer_tops_km, write_synth_cre_models


def _read_crh_layers(path: Path) -> tuple[str, list[tuple[float, float]]]:
	lines = Path(path).read_text(encoding='utf-8').splitlines()
	assert lines

	model = lines[0].strip()
	layers: list[tuple[float, float]] = []
	for line in lines[1:]:
		assert len(line) >= 10
		v = float(line[:5])
		top = float(line[5:10])
		layers.append((v, top))

	return model, layers


def test_build_synth_layer_tops_km_rules() -> None:
	assert build_synth_layer_tops_km(1) == [0.0]
	assert build_synth_layer_tops_km(3) == [0.0, 1.0, 2.0]


def test_build_synth_layer_tops_km_rejects_lt_1() -> None:
	with pytest.raises(ValueError):
		build_synth_layer_tops_km(0)


def test_write_synth_cre_models_smoke(tmp_path: Path) -> None:
	run_dir = tmp_path / 'run' / 'out'
	p_cre, s_cre = write_synth_cre_models(
		run_dir,
		vp_kms=6.0,
		vs_kms=3.5,
		shift_km=0.25,
		n_layers=3,
	)

	assert p_cre.is_file()
	assert s_cre.is_file()

	p_model, p_layers = _read_crh_layers(p_cre)
	s_model, s_layers = _read_crh_layers(s_cre)

	assert p_model == 'CRE_P'
	assert s_model == 'CRE_S'

	assert [top for _, top in p_layers] == [0.0, 1.25, 2.25]
	assert [top for _, top in s_layers] == [0.0, 1.25, 2.25]

	assert all(v == 6.0 for v, _ in p_layers)
	assert all(v == 3.5 for v, _ in s_layers)
