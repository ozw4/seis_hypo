from __future__ import annotations

from pathlib import Path

import pytest

from hypo.cre import apply_layer_top_shift_km, write_cre_from_layer_tops


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


@pytest.mark.parametrize(
	'layer_tops_km',
	[
		[0.0, 1.0, 1.0],  # equal
		[0.0, 2.0, 1.0],  # decreasing
	],
)
def test_apply_layer_top_shift_km_rejects_non_increasing(
	layer_tops_km: list[float],
) -> None:
	with pytest.raises(ValueError):
		apply_layer_top_shift_km(layer_tops_km, shift_km=0.1)


def test_apply_layer_top_shift_km_applies_shift_only_to_i_ge_1() -> None:
	tops = apply_layer_top_shift_km([0.0, 1.0, 2.0], shift_km=0.3)
	assert tops[0] == 0.0
	assert tops[1:] == [1.3, 2.3]


def test_write_cre_from_layer_tops_writes_p_and_s_cre(tmp_path: Path) -> None:
	run_dir = tmp_path / 'run' / 'out'
	p_cre, s_cre = write_cre_from_layer_tops(
		run_dir,
		vp_kms=6.0,
		vs_kms=3.5,
		layer_tops_km=[0.0, 1.0, 2.0],
		shift_km=0.3,
	)

	assert p_cre == run_dir / 'P.cre'
	assert s_cre == run_dir / 'S.cre'
	assert p_cre.is_file()
	assert s_cre.is_file()

	p_model, p_layers = _read_crh_layers(p_cre)
	s_model, s_layers = _read_crh_layers(s_cre)

	assert p_model == 'CRE_P'
	assert s_model == 'CRE_S'

	assert [top for _, top in p_layers] == [0.0, 1.3, 2.3]
	assert [top for _, top in s_layers] == [0.0, 1.3, 2.3]

	assert all(v == 6.0 for v, _ in p_layers)
	assert all(v == 3.5 for v, _ in s_layers)
