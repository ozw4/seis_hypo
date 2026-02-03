from __future__ import annotations

import numpy as np
import pytest

from pick.picks_from_probs import extract_pick_near_ref


def test_extract_pick_near_ref_inclusive_window_picks_peak_at_i1() -> None:
	fs = 10.0
	pre_sec = 1.0
	post_sec = 3.0

	score = np.zeros(100, dtype=float)
	ref = 50.0
	# i0 = 50 - 10 = 40, i1 = 50 + 30 = 80 (inclusive)
	score[80] = 0.9  # peak at i1 should be included

	out = extract_pick_near_ref(
		score,
		ref,
		fs_hz=fs,
		search_pre_sec=pre_sec,
		search_post_sec=post_sec,
		thr=0.2,
		min_sep_sec=0.0,
	)

	assert out['found_peak'] is True
	assert out['search_i0'] == 40
	assert out['search_i1'] == 80
	assert out['est_pick_idx'] == 80.0
	assert out['score_at_pick'] == pytest.approx(0.9)
	assert out['n_peaks'] == 1
	assert out['fail_reason'] == ''


def test_extract_pick_near_ref_clips_search_window_near_start() -> None:
	fs = 10.0
	score = np.zeros(40, dtype=float)

	# ref near start: i0 becomes negative, should clip to 0
	ref = 2.0
	# pre_n=10, post_n=30 => i0=-8 -> 0, i1=32
	score[0] = 0.6
	score[1] = 0.0  # ensure edge peak at 0

	out = extract_pick_near_ref(
		score,
		ref,
		fs_hz=fs,
		search_pre_sec=1.0,
		search_post_sec=3.0,
		thr=0.2,
		min_sep_sec=0.0,
	)

	assert out['found_peak'] is True
	assert out['search_i0'] == 0
	assert out['search_i1'] == 32
	assert out['est_pick_idx'] == 0.0
	assert out['score_at_pick'] == pytest.approx(0.6)
	assert out['fail_reason'] == ''


def test_extract_pick_near_ref_tie_break_earliest_time_on_equal_score() -> None:
	fs = 10.0
	score = np.zeros(120, dtype=float)

	ref = 50.0
	# window: [40..80]
	score[60] = 0.8
	score[70] = 0.8  # same score; should pick earlier (60)

	out = extract_pick_near_ref(
		score,
		ref,
		fs_hz=fs,
		search_pre_sec=1.0,
		search_post_sec=3.0,
		thr=0.2,
		min_sep_sec=0.0,  # keep both peaks; tie-break happens here
	)

	assert out['found_peak'] is True
	assert out['est_pick_idx'] == 60.0
	assert out['score_at_pick'] == pytest.approx(0.8)
	assert out['n_peaks'] == 2
	assert out['fail_reason'] == ''


def test_extract_pick_near_ref_min_sep_sec_is_applied_in_samples() -> None:
	fs = 100.0
	score = np.zeros(200, dtype=float)

	# Two peaks within 0.2 sec => within 20 samples. Keep higher one.
	score[50] = 0.6
	score[60] = 0.9  # within 10 samples < 20, should win

	out = extract_pick_near_ref(
		score,
		ref_pick_idx=55.0,
		fs_hz=fs,
		search_pre_sec=0.5,
		search_post_sec=0.5,
		thr=0.2,
		min_sep_sec=0.2,  # => 20 samples
	)

	assert out['found_peak'] is True
	assert out['est_pick_idx'] == 60.0
	assert out['score_at_pick'] == pytest.approx(0.9)
	assert out['n_peaks'] == 1
	assert out['fail_reason'] == ''


def test_extract_pick_near_ref_no_peak_returns_nans_and_reason() -> None:
	score = np.zeros(50, dtype=float)

	out = extract_pick_near_ref(
		score,
		ref_pick_idx=10.0,
		fs_hz=10.0,
		search_pre_sec=1.0,
		search_post_sec=1.0,
		thr=0.2,
		min_sep_sec=0.2,
	)

	assert out['found_peak'] is False
	assert out['n_peaks'] == 0
	assert out['fail_reason'] == 'no_peak'
	assert np.isnan(float(out['est_pick_idx']))
	assert np.isnan(float(out['score_at_pick']))
