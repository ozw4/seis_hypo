from __future__ import annotations

from collections.abc import Mapping

import numpy as np

ELLIPSE_COLS = (
	'ell_s1_km',
	'ell_az1_deg',
	'ell_dip1_deg',
	'ell_s2_km',
	'ell_az2_deg',
	'ell_dip2_deg',
	'ell_s3_km',
	'ell_az3_deg',
	'ell_dip3_deg',
)


def unit_vector_from_az_dip(az_deg: float, dip_deg: float) -> np.ndarray:
	"""AZ/DIP -> unit vector in (X=East, Y=North, Z=Depth[down+]).

	Spec (fixed):
	- AZ: degrees east of north (0=N, 90=E)
	- DIP: degrees measured downward from horizontal (0=horizontal, 90=vertical down)
	"""
	az_deg = float(az_deg)
	dip_deg = float(dip_deg)
	if not np.isfinite(az_deg) or not np.isfinite(dip_deg):
		raise ValueError(f'invalid az/dip (non-finite): az={az_deg}, dip={dip_deg}')

	az = np.deg2rad(az_deg)
	dip = np.deg2rad(dip_deg)

	cdip = float(np.cos(dip))
	u_e = float(np.sin(az) * cdip)
	u_n = float(np.cos(az) * cdip)
	u_d = float(np.sin(dip))

	u = np.array([u_e, u_n, u_d], dtype=float)
	n = float(np.linalg.norm(u))
	if not np.isfinite(n) or n <= 0.0:
		raise ValueError(f'invalid unit vector from az/dip: az={az_deg}, dip={dip_deg}')
	return u / n


def _symmetrize(a: np.ndarray) -> np.ndarray:
	return 0.5 * (a + a.T)


def _clamp_small_negative_eigs(
	eigvals: np.ndarray, tol: float, context: str
) -> np.ndarray:
	min_eig = float(np.min(eigvals))
	if min_eig < -tol:
		raise ValueError(
			f'{context}: covariance has negative eigenvalue {min_eig:g} < -tol({tol:g})'
		)
	e = eigvals.copy()
	e[e < 0.0] = 0.0
	return e


def _check_orthogonality(U: np.ndarray, eps: float) -> None:
	"""Fail-fast check for principal-axis orthogonality (no correction)."""
	G = U.T @ U
	d12 = float(G[0, 1])
	d13 = float(G[0, 2])
	d23 = float(G[1, 2])
	if max(abs(d12), abs(d13), abs(d23)) > float(eps):
		raise ValueError(
			f'principal axes are not orthogonal within eps={eps:g}: '
			f'dot12={d12:g}, dot13={d13:g}, dot23={d23:g}'
		)


def error_ellipse_to_cov_xyz_km2(
	s1_km: float,
	az1_deg: float,
	dip1_deg: float,
	s2_km: float,
	az2_deg: float,
	dip2_deg: float,
	s3_km: float,
	az3_deg: float,
	dip3_deg: float,
	*,
	sigma_scale_sec: float = 1.0,
	tol: float = 1e-10,
	eps_orth: float = 5e-2,
) -> np.ndarray:
	"""ERROR ELLIPSE principal axes -> 3D covariance Σ_xyz (km^2).

	Σ = U diag([s1^2, s2^2, s3^2]) U^T, with U=[u1 u2 u3] (columns).
	External scaling (sigma_scale_sec=σ): Σ *= σ^2.

	Notes:
	- Orthogonality is checked (fail-fast) using dot-products of the unit vectors.
	- Small negative eigenvalues due to FP error are clamped to 0; large negatives raise.

	"""
	if sigma_scale_sec <= 0.0 or not np.isfinite(float(sigma_scale_sec)):
		raise ValueError(f'invalid sigma_scale_sec: {sigma_scale_sec!r}')
	if eps_orth <= 0.0 or not np.isfinite(float(eps_orth)):
		raise ValueError(f'invalid eps_orth: {eps_orth!r}')

	s = np.array([float(s1_km), float(s2_km), float(s3_km)], dtype=float)
	if np.any(~np.isfinite(s)):
		raise ValueError(f'invalid SERR value(s): {s!r}')
	if np.any(s < 0.0):
		raise ValueError(f'SERR must be non-negative (km): {s!r}')

	u1 = unit_vector_from_az_dip(az1_deg, dip1_deg)
	u2 = unit_vector_from_az_dip(az2_deg, dip2_deg)
	u3 = unit_vector_from_az_dip(az3_deg, dip3_deg)
	U = np.column_stack([u1, u2, u3]).astype(float, copy=False)
	_check_orthogonality(U, float(eps_orth))

	D = np.diag(s**2)
	S = U @ D @ U.T
	S = _symmetrize(S)
	S = S * float(sigma_scale_sec) ** 2

	w = np.linalg.eigvalsh(S)
	w = _clamp_small_negative_eigs(w, float(tol), 'Σ_xyz')
	if not np.all(np.isfinite(w)):
		raise ValueError('Σ_xyz: non-finite eigenvalues')
	return S


def cov_xyz_to_cov_2d_km2(cov_xyz_km2: np.ndarray, plane: str) -> np.ndarray:
	"""Extract 2x2 marginal covariance for a given section.

	plane:
	- 'xy': [X,Y]
	- 'xz': [X,Z]
	- 'yz': [Z,Y]  (fixed order to match plotting: x=Depth(Z), y=North(Y))
	"""
	S = np.asarray(cov_xyz_km2, dtype=float)
	if S.shape != (3, 3):
		raise ValueError(f'cov_xyz must be 3x3, got {S.shape}')

	p = plane.lower()
	if p == 'xy':
		idx = (0, 1)
	elif p == 'xz':
		idx = (0, 2)
	elif p == 'yz':
		idx = (2, 1)
	else:
		raise ValueError(f'unknown plane: {plane!r}')

	S2 = S[np.ix_(idx, idx)]
	return _symmetrize(S2)


def cov_2d_to_ellipse_params(
	cov_2d_km2: np.ndarray, *, tol: float = 1e-10
) -> tuple[float, float, float]:
	"""2x2 covariance -> (a_km, b_km, theta_rad) for 1σ ellipse.

	theta_rad: CCW angle from +x axis to major axis (v1).
	"""
	C = np.asarray(cov_2d_km2, dtype=float)
	if C.shape != (2, 2):
		raise ValueError(f'cov_2d must be 2x2, got {C.shape}')
	C = _symmetrize(C)

	w, V = np.linalg.eigh(C)  # ascending
	w = _clamp_small_negative_eigs(w, float(tol), 'Σ_2d')

	order = np.argsort(w)[::-1]
	w = w[order]
	V = V[:, order]

	a = float(np.sqrt(w[0]))
	b = float(np.sqrt(w[1]))
	if not np.isfinite(a) or not np.isfinite(b):
		raise ValueError(f'non-finite ellipse radii: a={a}, b={b}')

	v1 = V[:, 0]
	theta = float(np.arctan2(v1[1], v1[0]))
	if not np.isfinite(theta):
		raise ValueError('non-finite ellipse angle')

	return a, b, theta


def projected_ellipses_from_error_ellipse(
	s1_km: float,
	az1_deg: float,
	dip1_deg: float,
	s2_km: float,
	az2_deg: float,
	dip2_deg: float,
	s3_km: float,
	az3_deg: float,
	dip3_deg: float,
	*,
	sigma_scale_sec: float = 1.0,
	tol: float = 1e-10,
) -> dict[str, float]:
	"""Convenience: ERROR ELLIPSE -> 3 projected 1σ ellipses (XY/XZ/YZ)."""
	S = error_ellipse_to_cov_xyz_km2(
		s1_km,
		az1_deg,
		dip1_deg,
		s2_km,
		az2_deg,
		dip2_deg,
		s3_km,
		az3_deg,
		dip3_deg,
		sigma_scale_sec=sigma_scale_sec,
		tol=tol,
	)

	S_xy = cov_xyz_to_cov_2d_km2(S, 'xy')
	S_xz = cov_xyz_to_cov_2d_km2(S, 'xz')
	S_yz = cov_xyz_to_cov_2d_km2(S, 'yz')

	a_xy, b_xy, th_xy = cov_2d_to_ellipse_params(S_xy, tol=tol)
	a_xz, b_xz, th_xz = cov_2d_to_ellipse_params(S_xz, tol=tol)
	a_yz, b_yz, th_yz = cov_2d_to_ellipse_params(S_yz, tol=tol)

	ell_3d_max = float(
		max(float(s1_km), float(s2_km), float(s3_km)) * float(sigma_scale_sec)
	)

	return {
		'a_xy_km': a_xy,
		'b_xy_km': b_xy,
		'theta_xy_rad': th_xy,
		'a_xz_km': a_xz,
		'b_xz_km': b_xz,
		'theta_xz_rad': th_xz,
		'a_yz_km': a_yz,
		'b_yz_km': b_yz,
		'theta_yz_rad': th_yz,
		'ell_3d_max_km': ell_3d_max,
	}


def projected_ellipses_from_record(
	rec: Mapping[str, float], *, sigma_scale_sec: float = 1.0, tol: float = 1e-10
) -> dict[str, float]:
	"""Mapping/Series compatible wrapper (expects ell_* keys from eval_metrics)."""
	return projected_ellipses_from_error_ellipse(
		rec['ell_s1_km'],
		rec['ell_az1_deg'],
		rec['ell_dip1_deg'],
		rec['ell_s2_km'],
		rec['ell_az2_deg'],
		rec['ell_dip2_deg'],
		rec['ell_s3_km'],
		rec['ell_az3_deg'],
		rec['ell_dip3_deg'],
		sigma_scale_sec=sigma_scale_sec,
		tol=tol,
	)
