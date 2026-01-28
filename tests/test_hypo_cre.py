from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hypo.cre import (
	compute_cre_layer_top_shift_km,
	compute_reference_elevation_km,
	compute_typical_station_elevation_km,
	write_cre_meta,
)


def test_compute_reference_elevation_km_max_plus_margin() -> None:
	df = pd.DataFrame({'Elevation_m': [100.0, 1200.0, 800.0]})
	got = compute_reference_elevation_km(df, margin_m=300.0)
	assert got == 1.5


def test_compute_reference_elevation_km_clamps_negative_to_zero() -> None:
	df = pd.DataFrame({'Elevation_m': [-200.0, -50.0]})
	got = compute_reference_elevation_km(df, margin_m=0.0)
	assert got == 0.0


@pytest.mark.parametrize(
	'station_df,elevation_col',
	[
		(None, 'Elevation_m'),
		(pd.DataFrame({'Other': [1.0]}), 'Elevation_m'),
		(pd.DataFrame({'Elevation_m': []}), 'Elevation_m'),
		(pd.DataFrame({'Elevation_m': [float('nan'), float('nan')]}), 'Elevation_m'),
	],
)
def test_compute_reference_elevation_km_input_validation(
	station_df: pd.DataFrame | None,
	elevation_col: str,
) -> None:
	with pytest.raises(ValueError):
		compute_reference_elevation_km(station_df, elevation_col=elevation_col)


def test_compute_typical_station_elevation_km_rules() -> None:
	assert compute_typical_station_elevation_km(explicit_m=None) == 0.0
	assert compute_typical_station_elevation_km(explicit_m=1234.0) == 1.234


def test_compute_cre_layer_top_shift_km_is_difference() -> None:
	got = compute_cre_layer_top_shift_km(1.5, 0.2)
	assert got == 1.3


def test_write_cre_meta_writes_three_scalars(tmp_path: Path) -> None:
	run_dir = tmp_path / 'run' / 'out'

	ref = 1.5
	typical = 0.2
	shift = 1.3

	write_cre_meta(run_dir, ref_elev_km=ref, typical_elev_km=typical, shift_km=shift)

	p_ref = run_dir / 'cre_ref_elev_km.txt'
	p_typ = run_dir / 'cre_typical_station_elev_km.txt'
	p_shf = run_dir / 'cre_layer_top_shift_km.txt'

	assert p_ref.is_file()
	assert p_typ.is_file()
	assert p_shf.is_file()

	t_ref = p_ref.read_text(encoding='utf-8')
	t_typ = p_typ.read_text(encoding='utf-8')
	t_shf = p_shf.read_text(encoding='utf-8')

	assert t_ref.endswith('\n')
	assert t_typ.endswith('\n')
	assert t_shf.endswith('\n')

	assert float(t_ref.strip()) == ref
	assert float(t_typ.strip()) == typical
	assert float(t_shf.strip()) == shift
