"""GaMMA configuration builder."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from gamma.utils import estimate_eps

if TYPE_CHECKING:
	import pandas as pd


def build_gamma_config(  # noqa: PLR0913
	*,
	stations_df: pd.DataFrame,
	vel: dict | None,
	method: str,
	use_dbscan: bool,
	use_amplitude: bool,
	oversample_factor_bgmm: int,
	use_eikonal_1d: bool,
	eikonal_h_km: float,
	xy_margin_km: float,
	z_range_km: tuple[float, float],
	dbscan_eps_sec: float | None,
	dbscan_eps_sigma: float,
	dbscan_eps_mult: float,
	dbscan_min_samples: int,
	dbscan_min_cluster_size: int,
	dbscan_max_time_space_ratio: float,
	ncpu: int,
	min_picks_per_eq: int,
	min_p_picks_per_eq: int,
	min_s_picks_per_eq: int,
	max_sigma11_sec: float,
	max_sigma22_log10_ms: float,
	max_sigma12_cov: float,
) -> dict:
	"""Build the config dict passed to ``gamma.utils.association``."""
	x_min = float(stations_df['x(km)'].min()) - float(xy_margin_km)
	x_max = float(stations_df['x(km)'].max()) + float(xy_margin_km)
	y_min = float(stations_df['y(km)'].min()) - float(xy_margin_km)
	y_max = float(stations_df['y(km)'].max()) + float(xy_margin_km)

	cfg: dict = {}
	cfg['use_dbscan'] = bool(use_dbscan)
	cfg['use_amplitude'] = bool(use_amplitude)
	cfg['method'] = str(method)
	cfg['oversample_factor'] = (
		int(oversample_factor_bgmm) if cfg['method'] == 'BGMM' else 1
	)

	cfg['dims'] = ['x(km)', 'y(km)', 'z(km)']
	cfg['x(km)'] = (x_min, x_max)
	cfg['y(km)'] = (y_min, y_max)
	cfg['z(km)'] = (float(z_range_km[0]), float(z_range_km[1]))

	cfg['bfgs_bounds'] = (
		(x_min, x_max),
		(y_min, y_max),
		(cfg['z(km)'][0], cfg['z(km)'][1]),
		(None, None),
	)

	if use_eikonal_1d:
		if vel is None:
			raise ValueError('vel is required when use_eikonal_1d is True')
		vp0 = float(np.asarray(vel['p'], dtype=float)[0])
		vs0 = float(np.asarray(vel['s'], dtype=float)[0])
		cfg['vel'] = {'p': vp0, 's': vs0}
		cfg['eikonal'] = {
			'vel': vel,
			'h': float(eikonal_h_km),
			'xlim': cfg['x(km)'],
			'ylim': cfg['y(km)'],
			'zlim': cfg['z(km)'],
		}
	else:
		cfg['vel'] = {'p': 6.0, 's': 6.0 / 1.75}
		cfg['eikonal'] = None

	if cfg['use_dbscan']:
		vp_for_eps = float(cfg['vel']['p'])
		if dbscan_eps_sec is None:
			eps0 = float(
				estimate_eps(stations_df, vp_for_eps, sigma=float(dbscan_eps_sigma))
			)
		else:
			eps0 = float(dbscan_eps_sec)

		eps0 *= float(dbscan_eps_mult)
		if not np.isfinite(eps0) or eps0 <= 0.0:
			raise ValueError(f'Bad dbscan_eps after scaling: {eps0}')

		cfg['dbscan_eps'] = eps0
		cfg['dbscan_min_samples'] = int(dbscan_min_samples)
		cfg['dbscan_min_cluster_size'] = int(dbscan_min_cluster_size)
		cfg['dbscan_max_time_space_ratio'] = float(dbscan_max_time_space_ratio)

	cfg['ncpu'] = int(ncpu)

	cfg['min_picks_per_eq'] = int(min_picks_per_eq)
	cfg['min_p_picks_per_eq'] = int(min_p_picks_per_eq)
	cfg['min_s_picks_per_eq'] = int(min_s_picks_per_eq)
	cfg['max_sigma11'] = float(max_sigma11_sec)
	cfg['max_sigma22'] = float(max_sigma22_log10_ms)
	cfg['max_sigma12'] = float(max_sigma12_cov)

	return cfg
