from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

_JST = timezone(timedelta(hours=9))


def _as_jst(dt: datetime) -> datetime:
	if dt.tzinfo is None:
		return dt.replace(tzinfo=_JST)
	return dt.astimezone(_JST)


def _format_jst_iso(dt: datetime) -> str:
	return _as_jst(dt).isoformat(timespec='milliseconds')


def _tol_key(tol: float) -> str:
	s = f'{float(tol):.2f}'
	return f'good_{s.replace(".", "p")}'


def eval_dt_row(
	*,
	t0_jst: datetime,
	t_ref: datetime,
	fs_hz: float,
	est_pick_idx: float | None,
	found_peak: bool,
	tol_sec: list[float],
	keep_missing_rows: bool = True,
	score_at_pick: float | None = None,
	n_peaks: int | None = None,
	search_i0: int | None = None,
	search_i1: int | None = None,
	fail_reason: str = '',
) -> dict[str, object] | None:
	fs_val = float(fs_hz)
	if not np.isfinite(fs_val) or fs_val <= 0.0:
		raise ValueError('fs_hz must be > 0')

	t0_jst_aware = _as_jst(t0_jst)
	t_ref_jst = _as_jst(t_ref)
	ref_pick_idx = int(round((t_ref_jst - t0_jst_aware).total_seconds() * fs_val))
	t_ref_iso = _format_jst_iso(t_ref_jst)

	est_idx_val = float('nan') if est_pick_idx is None else float(est_pick_idx)
	found_peak_out = bool(found_peak) and np.isfinite(est_idx_val)

	if not found_peak_out:
		if not keep_missing_rows:
			return None
		row = {
			't_ref_iso': t_ref_iso,
			'ref_pick_idx': int(ref_pick_idx),
			'found_peak': 0,
			't_est_iso': '',
			'est_pick_idx': float('nan'),
			'dt_sec': float('nan'),
			'score_at_pick': float('nan'),
			'n_peaks': 0,
			'fail_reason': str(fail_reason),
		}
		for tol in tol_sec:
			row[_tol_key(float(tol))] = 0
		row['search_i0'] = float('nan') if search_i0 is None else int(search_i0)
		row['search_i1'] = float('nan') if search_i1 is None else int(search_i1)
		return row

	t_est_jst = t0_jst_aware + timedelta(seconds=float(est_idx_val) / fs_val)
	dt_sec = (t_est_jst - t_ref_jst).total_seconds()

	row = {
		't_ref_iso': t_ref_iso,
		'ref_pick_idx': int(ref_pick_idx),
		'found_peak': 1,
		't_est_iso': _format_jst_iso(t_est_jst),
		'est_pick_idx': float(est_idx_val),
		'dt_sec': float(dt_sec),
		'score_at_pick': float('nan')
		if score_at_pick is None or not np.isfinite(float(score_at_pick))
		else float(score_at_pick),
		'n_peaks': 0 if n_peaks is None else int(n_peaks),
		'fail_reason': str(fail_reason),
	}
	for tol in tol_sec:
		tol_v = float(tol)
		row[_tol_key(tol_v)] = 1 if abs(dt_sec) <= tol_v else 0
	row['search_i0'] = float('nan') if search_i0 is None else int(search_i0)
	row['search_i1'] = float('nan') if search_i1 is None else int(search_i1)
	return row
