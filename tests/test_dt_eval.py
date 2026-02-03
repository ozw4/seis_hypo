from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from pick.dt_eval import eval_dt_row


def test_eval_dt_row_computes_ref_idx_dt_and_formats_jst_iso_ms() -> None:
	# t0/t_ref are treated as JST
	t0 = datetime(2026, 2, 3, 0, 0, 0)  # naive JST
	t_ref = t0 + timedelta(seconds=1.234)  # ref at +1.234s
	fs = 100.0
	est_idx = 130.0  # +1.300s => dt = +0.066s

	out = eval_dt_row(
		t0_jst=t0,
		t_ref=t_ref,
		fs_hz=fs,
		est_pick_idx=est_idx,
		found_peak=True,
		tol_sec=[0.05, 0.10, 0.20],
		score_at_pick=0.9,
		n_peaks=2,
		search_i0=10,
		search_i1=20,
		fail_reason='',
	)

	assert out is not None
	assert out['ref_pick_idx'] == 123  # round(1.234 * 100)
	assert out['est_pick_idx'] == 130.0
	assert out['dt_sec'] == pytest.approx(0.066, abs=1e-12)

	# ISO format: milliseconds + +09:00
	assert isinstance(out['t_ref_iso'], str)
	assert out['t_ref_iso'].endswith('+09:00')
	assert len(out['t_ref_iso'].split('.')) == 2
	assert len(out['t_ref_iso'].split('.')[1].split('+')[0]) == 3  # .sss

	assert isinstance(out['t_est_iso'], str)
	assert out['t_est_iso'].endswith('+09:00')

	# good_* flags
	assert out['good_0p05'] == 0
	assert out['good_0p10'] == 1
	assert out['good_0p20'] == 1

	# passthrough fields
	assert out['score_at_pick'] == pytest.approx(0.9)
	assert out['n_peaks'] == 2
	assert out['search_i0'] == 10
	assert out['search_i1'] == 20
	assert out['found_peak'] == 1


def test_eval_dt_row_missing_peak_outputs_empty_est_and_nans_and_good_zero() -> None:
	t0 = datetime(2026, 2, 3, 0, 0, 0)
	t_ref = t0 + timedelta(seconds=1.0)

	out = eval_dt_row(
		t0_jst=t0,
		t_ref=t_ref,
		fs_hz=100.0,
		est_pick_idx=float('nan'),
		found_peak=True,  # should be normalized to 0 because est_pick_idx not finite
		tol_sec=[0.05, 0.10, 0.20],
		keep_missing_rows=True,
		score_at_pick=0.9,
		n_peaks=5,
		search_i0=None,
		search_i1=None,
		fail_reason='no_peak',
	)

	assert out is not None
	assert out['found_peak'] == 0
	assert out['t_est_iso'] == ''
	assert np.isnan(float(out['est_pick_idx']))
	assert np.isnan(float(out['dt_sec']))
	assert np.isnan(float(out['score_at_pick']))
	assert out['n_peaks'] == 0  # forced to 0 when missing
	assert out['good_0p05'] == 0
	assert out['good_0p10'] == 0
	assert out['good_0p20'] == 0
	assert np.isnan(float(out['search_i0']))
	assert np.isnan(float(out['search_i1']))
	assert out['fail_reason'] == 'no_peak'


def test_eval_dt_row_keep_missing_false_returns_none() -> None:
	t0 = datetime(2026, 2, 3, 0, 0, 0)
	t_ref = t0 + timedelta(seconds=1.0)

	out = eval_dt_row(
		t0_jst=t0,
		t_ref=t_ref,
		fs_hz=100.0,
		est_pick_idx=None,
		found_peak=False,
		tol_sec=[0.05, 0.10],
		keep_missing_rows=False,
	)
	assert out is None


def test_eval_dt_row_accepts_tz_aware_and_converts_to_jst() -> None:
	# t0/t_ref given as UTC aware; should be converted to JST for formatting and idx
	utc = timezone.utc
	t0_utc = datetime(2026, 2, 2, 15, 0, 0, tzinfo=utc)  # == 2026-02-03 00:00 JST
	t_ref_utc = t0_utc + timedelta(seconds=1.0)

	out = eval_dt_row(
		t0_jst=t0_utc,
		t_ref=t_ref_utc,
		fs_hz=10.0,
		est_pick_idx=10.0,
		found_peak=True,
		tol_sec=[0.2],
	)

	assert out is not None
	assert out['ref_pick_idx'] == 10  # 1.0s * 10Hz
	assert out['t_ref_iso'].endswith('+09:00')
