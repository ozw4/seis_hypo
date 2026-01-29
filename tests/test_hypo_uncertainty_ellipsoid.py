from __future__ import annotations

import math

import numpy as np
import pytest

from hypo.uncertainty_ellipsoid import (
	cov_2d_to_ellipse_params,
	cov_xyz_to_cov_2d_km2,
	error_ellipse_to_cov_xyz_km2,
	projected_ellipses_from_error_ellipse,
	projected_ellipses_from_record,
	unit_vector_from_az_dip,
)


def _assert_vec_close(v: np.ndarray, expected: np.ndarray, atol: float = 1e-12) -> None:
	v = np.asarray(v, dtype=float)
	expected = np.asarray(expected, dtype=float)
	assert v.shape == expected.shape
	assert np.allclose(v, expected, atol=atol, rtol=0.0)


def _assert_unit(v: np.ndarray, atol: float = 1e-12) -> None:
	n = float(np.linalg.norm(v))
	assert n == pytest.approx(1.0, rel=0.0, abs=atol)


def _angle_mod_pi_diff(a: float, b: float) -> float:
	# Axis direction is invariant under theta -> theta + pi.
	d = (a - b) % math.pi
	return float(min(d, math.pi - d))


def _assert_angle_close_mod_pi(
	theta: float, expected: float, atol: float = 1e-7
) -> None:
	assert _angle_mod_pi_diff(float(theta), float(expected)) <= atol


# ========= 1) unit_vector_from_az_dip =========


def test_unit_vector_from_az_dip_cardinals_and_norm() -> None:
	u_n = unit_vector_from_az_dip(0.0, 0.0)  # N, horizontal
	u_e = unit_vector_from_az_dip(90.0, 0.0)  # E, horizontal
	u_d = unit_vector_from_az_dip(123.0, 90.0)  # Down, az irrelevant

	_assert_vec_close(u_n, np.array([0.0, 1.0, 0.0]))
	_assert_vec_close(u_e, np.array([1.0, 0.0, 0.0]))
	_assert_vec_close(u_d, np.array([0.0, 0.0, 1.0]))

	_assert_unit(u_n)
	_assert_unit(u_e)
	_assert_unit(u_d)


@pytest.mark.parametrize(
	'az,dip', [(np.nan, 0.0), (0.0, np.nan), (np.inf, 0.0), (0.0, np.inf)]
)
def test_unit_vector_from_az_dip_rejects_non_finite(az: float, dip: float) -> None:
	with pytest.raises(ValueError):
		unit_vector_from_az_dip(az, dip)


# ========= 2) error_ellipse_to_cov_xyz_km2 =========


def test_error_ellipse_to_cov_xyz_km2_diag_basis_and_sigma_scale() -> None:
	# Principal axes aligned with E/N/Down:
	# u1 = E (az=90,dip=0), u2 = N (az=0,dip=0), u3 = Down (dip=90).
	s1, s2, s3 = 2.0, 3.0, 4.0

	S = error_ellipse_to_cov_xyz_km2(
		s1,
		90.0,
		0.0,
		s2,
		0.0,
		0.0,
		s3,
		0.0,
		90.0,
		sigma_scale_sec=1.0,
	)
	expected = np.diag([s1**2, s2**2, s3**2])
	assert np.allclose(S, expected, atol=1e-12, rtol=0.0)

	S2 = error_ellipse_to_cov_xyz_km2(
		s1,
		90.0,
		0.0,
		s2,
		0.0,
		0.0,
		s3,
		0.0,
		90.0,
		sigma_scale_sec=2.0,
	)
	assert np.allclose(S2, expected * 4.0, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize('sigma', [0.0, -1.0, np.nan, np.inf])
def test_error_ellipse_to_cov_xyz_km2_rejects_invalid_sigma(sigma: float) -> None:
	with pytest.raises(ValueError):
		error_ellipse_to_cov_xyz_km2(
			1.0,
			90.0,
			0.0,
			1.0,
			0.0,
			0.0,
			1.0,
			0.0,
			90.0,
			sigma_scale_sec=sigma,
		)


@pytest.mark.parametrize('s', [-1.0, np.nan, np.inf])
def test_error_ellipse_to_cov_xyz_km2_rejects_invalid_serr(s: float) -> None:
	with pytest.raises(ValueError):
		error_ellipse_to_cov_xyz_km2(
			s,
			90.0,
			0.0,
			1.0,
			0.0,
			0.0,
			1.0,
			0.0,
			90.0,
		)


def test_error_ellipse_to_cov_xyz_km2_rejects_non_orthogonal_axes() -> None:
	# u1 == u2 -> dot=1, must fail orthogonality check.
	with pytest.raises(ValueError, match='not orthogonal'):
		error_ellipse_to_cov_xyz_km2(
			1.0,
			0.0,
			0.0,
			1.0,
			0.0,
			0.0,
			1.0,
			0.0,
			90.0,
		)


# ========= 3) cov_xyz_to_cov_2d_km2 =========


def test_cov_xyz_to_cov_2d_km2_plane_extraction_order_is_fixed() -> None:
	s1, s2, s3 = 2.0, 3.0, 4.0
	S = np.diag([s1**2, s2**2, s3**2])

	S_xy = cov_xyz_to_cov_2d_km2(S, 'xy')
	S_xz = cov_xyz_to_cov_2d_km2(S, 'xz')
	S_yz = cov_xyz_to_cov_2d_km2(S, 'yz')  # (Z, Y) order

	assert np.allclose(S_xy, np.diag([s1**2, s2**2]), atol=1e-12, rtol=0.0)
	assert np.allclose(S_xz, np.diag([s1**2, s3**2]), atol=1e-12, rtol=0.0)
	assert np.allclose(S_yz, np.diag([s3**2, s2**2]), atol=1e-12, rtol=0.0)


def test_cov_xyz_to_cov_2d_km2_rejects_bad_shape_and_plane() -> None:
	with pytest.raises(ValueError):
		cov_xyz_to_cov_2d_km2(np.zeros((2, 2)), 'xy')
	with pytest.raises(ValueError):
		cov_xyz_to_cov_2d_km2(np.eye(3), 'ab')


# ========= 4) cov_2d_to_ellipse_params =========


def test_cov_2d_to_ellipse_params_diagonal_cases() -> None:
	a, b, th = cov_2d_to_ellipse_params(np.array([[4.0, 0.0], [0.0, 1.0]]))
	assert a == pytest.approx(2.0, rel=0.0, abs=1e-12)
	assert b == pytest.approx(1.0, rel=0.0, abs=1e-12)
	_assert_angle_close_mod_pi(th, 0.0)

	a, b, th = cov_2d_to_ellipse_params(np.array([[1.0, 0.0], [0.0, 4.0]]))
	assert a == pytest.approx(2.0, rel=0.0, abs=1e-12)
	assert b == pytest.approx(1.0, rel=0.0, abs=1e-12)
	_assert_angle_close_mod_pi(th, math.pi / 2.0)


def test_cov_2d_to_ellipse_params_rotated_45deg() -> None:
	th0 = math.pi / 4.0
	R = np.array(
		[
			[math.cos(th0), -math.sin(th0)],
			[math.sin(th0), math.cos(th0)],
		],
		dtype=float,
	)
	D = np.diag([9.0, 4.0])
	C = R @ D @ R.T

	a, b, th = cov_2d_to_ellipse_params(C)
	assert a == pytest.approx(3.0, rel=0.0, abs=1e-10)
	assert b == pytest.approx(2.0, rel=0.0, abs=1e-10)
	_assert_angle_close_mod_pi(th, th0, atol=1e-7)


def test_cov_2d_to_ellipse_params_clamps_small_negative_eigs() -> None:
	C = np.array([[1.0, 0.0], [0.0, -1e-12]], dtype=float)
	a, b, th = cov_2d_to_ellipse_params(C, tol=1e-10)
	assert a == pytest.approx(1.0, rel=0.0, abs=1e-12)
	assert b == pytest.approx(0.0, rel=0.0, abs=1e-8)
	assert np.isfinite(th)


def test_cov_2d_to_ellipse_params_rejects_large_negative_eigs_and_bad_shape() -> None:
	with pytest.raises(ValueError):
		cov_2d_to_ellipse_params(np.zeros((3, 3)))

	C = np.array([[1.0, 0.0], [0.0, -1e-3]], dtype=float)
	with pytest.raises(ValueError, match='negative eigenvalue'):
		cov_2d_to_ellipse_params(C, tol=1e-10)


# ========= 5) projected_ellipses_from_error_ellipse =========


def test_projected_ellipses_from_error_ellipse_axis_aligned_consistency() -> None:
	s1, s2, s3 = 2.0, 3.0, 4.0
	out = projected_ellipses_from_error_ellipse(
		s1,
		90.0,
		0.0,
		s2,
		0.0,
		0.0,
		s3,
		0.0,
		90.0,
		sigma_scale_sec=1.0,
	)

	assert out['ell_3d_max_km'] == pytest.approx(4.0, rel=0.0, abs=1e-12)

	# XY: (E,N)
	assert sorted([out['a_xy_km'], out['b_xy_km']]) == pytest.approx(
		sorted([s1, s2]), rel=0.0, abs=1e-12
	)
	# XZ: (E,Down)
	assert sorted([out['a_xz_km'], out['b_xz_km']]) == pytest.approx(
		sorted([s1, s3]), rel=0.0, abs=1e-12
	)
	# YZ: (Down,North) in that order for covariance extraction, but radii are a/b so order-free check is fine
	assert sorted([out['a_yz_km'], out['b_yz_km']]) == pytest.approx(
		sorted([s3, s2]), rel=0.0, abs=1e-12
	)

	assert out['ell_3d_max_km'] == pytest.approx(max(s1, s2, s3), rel=0.0, abs=1e-12)


def test_projected_ellipses_from_error_ellipse_sigma_scales_radii_linearly() -> None:
	s1, s2, s3 = 2.0, 3.0, 4.0
	out1 = projected_ellipses_from_error_ellipse(
		s1,
		90.0,
		0.0,
		s2,
		0.0,
		0.0,
		s3,
		0.0,
		90.0,
		sigma_scale_sec=1.0,
	)
	out2 = projected_ellipses_from_error_ellipse(
		s1,
		90.0,
		0.0,
		s2,
		0.0,
		0.0,
		s3,
		0.0,
		90.0,
		sigma_scale_sec=2.0,
	)

	assert out2['a_xy_km'] == pytest.approx(out1['a_xy_km'] * 2.0, rel=0.0, abs=1e-12)
	assert out2['b_xy_km'] == pytest.approx(out1['b_xy_km'] * 2.0, rel=0.0, abs=1e-12)
	assert out2['ell_3d_max_km'] == pytest.approx(
		out1['ell_3d_max_km'] * 2.0, rel=0.0, abs=1e-12
	)


# ========= 6) projected_ellipses_from_record =========


def test_projected_ellipses_from_record_matches_direct_call() -> None:
	rec = {
		'ell_s1_km': 0.12,
		'ell_az1_deg': 91,
		'ell_dip1_deg': 21,
		'ell_s2_km': 0.09,
		'ell_az2_deg': 0,
		'ell_dip2_deg': 0,
		'ell_s3_km': 0.08,
		'ell_az3_deg': 271,
		'ell_dip3_deg': 68,
	}

	out_a = projected_ellipses_from_record(rec, sigma_scale_sec=1.0)
	out_b = projected_ellipses_from_error_ellipse(
		rec['ell_s1_km'],
		rec['ell_az1_deg'],
		rec['ell_dip1_deg'],
		rec['ell_s2_km'],
		rec['ell_az2_deg'],
		rec['ell_dip2_deg'],
		rec['ell_s3_km'],
		rec['ell_az3_deg'],
		rec['ell_dip3_deg'],
		sigma_scale_sec=1.0,
	)

	assert out_a == out_b
