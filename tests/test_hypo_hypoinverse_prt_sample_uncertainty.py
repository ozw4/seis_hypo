from __future__ import annotations

import re
from pathlib import Path

import pytest

from hypo.hypoinverse_prt import load_hypoinverse_summary_from_prt

_SAMPLE_SUMMARY_COUNT = 24


def _sample_prt_path() -> Path:
	p = Path(__file__).resolve().parent / 'data' / 'hypoinverse_run.prt'
	assert p.exists(), f'missing test fixture: {p}'
	return p


# A) 実サンプル統合テスト:
# - 全イベントが落ちずに取れる
# - ell_* / NSTA系 / eig_* / origin_time_err_sec が揃う
def test_prt_sample_parses_all_events_and_required_fields() -> None:
	prt = _sample_prt_path()
	df = load_hypoinverse_summary_from_prt(prt)

	assert len(df) == _SAMPLE_SUMMARY_COUNT
	assert df['seq'].tolist() == list(range(1, _SAMPLE_SUMMARY_COUNT + 1))
	assert df['sequence_no_prt'].tolist() == list(range(1, _SAMPLE_SUMMARY_COUNT + 1))
	assert df['id_no_prt'].notna().all()

	ell_cols = [
		'ell_s1_km',
		'ell_az1_deg',
		'ell_dip1_deg',
		'ell_s2_km',
		'ell_az2_deg',
		'ell_dip2_deg',
		'ell_s3_km',
		'ell_az3_deg',
		'ell_dip3_deg',
	]
	nsta_cols = [
		'NSTA',
		'NPHS',
		'DMIN',
		'MODEL',
		'GAP',
		'ITR',
		'NFM',
		'NWR',
		'NWS',
		'NVR',
	]
	eig_cols = ['eig_adj1', 'eig_adj2', 'eig_adj3', 'eig_adj4']
	time_err_cols = ['origin_time_err_sec']

	for c in ell_cols + nsta_cols + eig_cols + time_err_cols:
		assert c in df.columns
	for c in ['sequence_no_prt', 'id_no_prt']:
		assert c in df.columns

	assert df[ell_cols].notna().all().all()
	assert df[nsta_cols].notna().all().all()
	# このサンプルは全イベントに EIGENVALUES がある前提
	assert df[eig_cols].notna().all().all()
	assert df[time_err_cols].notna().all().all()


# B) ゴールデン値チェック（1イベントだけ）
def test_prt_sample_first_event_error_ellipse_has_expected_values() -> None:
	prt = _sample_prt_path()
	df = load_hypoinverse_summary_from_prt(prt)
	row0 = df.iloc[0]

	assert row0['ell_s1_km'] == pytest.approx(0.12, rel=0, abs=1e-12)
	assert int(row0['ell_az1_deg']) == 91
	assert int(row0['ell_dip1_deg']) == 21

	assert row0['ell_s2_km'] == pytest.approx(0.09, rel=0, abs=1e-12)
	assert int(row0['ell_az2_deg']) == 0
	assert int(row0['ell_dip2_deg']) == 0

	assert row0['ell_s3_km'] == pytest.approx(0.08, rel=0, abs=1e-12)
	assert int(row0['ell_az3_deg']) == 271
	assert int(row0['ell_dip3_deg']) == 68

	# このサンプルは全イベントに EIGENVALUES がある前提
	assert row0['eig_adj1'] == pytest.approx(18.036, rel=0, abs=1e-12)
	assert row0['eig_adj2'] == pytest.approx(1.950, rel=0, abs=1e-12)
	assert row0['eig_adj3'] == pytest.approx(1.683, rel=0, abs=1e-12)
	assert row0['eig_adj4'] == pytest.approx(1.245, rel=0, abs=1e-12)
	assert row0['origin_time_err_sec'] == pytest.approx(0.022, rel=0, abs=1e-12)


# C) サンプル改変ネガティブ（最小2本）
def test_prt_sample_negative_missing_error_ellipse_and_truncated_nsta_values(
	tmp_path: Path,
) -> None:
	prt = _sample_prt_path()
	text = prt.read_text(encoding='ascii', errors='strict')
	lines = text.splitlines()

	# C-1: ERROR ELLIPSE を1件だけ削除 → summary到達時に ValueError
	i_ell = next(i for i, l in enumerate(lines) if 'ERROR ELLIPSE' in l.upper())
	lines_no_ell = list(lines)
	lines_no_ell.pop(i_ell)
	p1 = tmp_path / 'no_error_ellipse.prt'
	p1.write_text('\n'.join(lines_no_ell) + '\n', encoding='ascii', newline='\n')
	with pytest.raises(
		ValueError, match=re.escape('ERROR ELLIPSE is missing for event summary')
	):
		load_hypoinverse_summary_from_prt(p1)

	# C-2: NSTA NPHS の値行欠損（ヘッダでファイルを切る）→ ValueError
	nsta_re = re.compile(r'^\s*NSTA\s+NPHS\b', re.IGNORECASE)
	i_nsta = next(i for i, l in enumerate(lines) if nsta_re.match(l))
	p2 = tmp_path / 'truncated_nsta.prt'
	p2.write_text('\n'.join(lines[: i_nsta + 1]) + '\n', encoding='ascii', newline='\n')
	with pytest.raises(
		ValueError, match=re.escape('NSTA/NPHS header line is truncated')
	):
		load_hypoinverse_summary_from_prt(p2)
