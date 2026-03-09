from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from hypo.hypoinverse_prt import (
	load_hypoinverse_summary_from_prt,
	parse_summary_line,
)

if TYPE_CHECKING:
	from pathlib import Path

_SOLVED_SUMMARY_LINE_E = (
	' 2026-01-01  0000  0.00  35  1.35  140E 2.04   0.30  0.00  0.11  0.07'
	'                       0.30'
)
_SOLVED_SUMMARY_LINE_W = _SOLVED_SUMMARY_LINE_E[:38] + 'W' + _SOLVED_SUMMARY_LINE_E[39:]
_SOLVED_SUMMARY_LINE_2 = _SOLVED_SUMMARY_LINE_E.replace(
	' 0000  0.00',
	' 0001 10.00',
	1,
)
_UNSOLVED_SUMMARY_LINE = (
	' 2025-11-01  0610  5.66   0  0.00    0  0.00  10.00  1.38  1.75 83.33'
	'                      10.00'
)
_ERROR_ELLIPSE_LINE = (
	' ERROR ELLIPSE: <SERR AZ DIP>-<   0.12  91 21>-<   0.09   0  0>-<   0.08 271 68>'
)
_NSTA_HEADER_LINE = (
	' NSTA NPHS  DMIN MODEL GAP ITR NFM NWR NWS NVR REMRKS-AVH  N.XMG-XMMAD-T'
	'   N.FMG-FMMAD-T  L F X'
)
_NSTA_VALUES_LINE = (
	'  220  220   0.1  CRES 205   5   0 220 110 220'
	'                                                 '
)
_SUMMARY_HEADER_LINE = (
	' YEAR MO DA  --ORIGIN--  --LAT N-  --LON W--  DEPTH-G RMS   ERH   ERZ'
	'  XMAG1 FMAG1 PMAG GEOID-DEP'
)


def _write_prt(tmp_path: Path, name: str, lines: list[str]) -> Path:
	path = tmp_path / name
	path.write_text('\n'.join(lines) + '\n', encoding='ascii', newline='\n')
	return path


def _solved_event_lines(seq: int, event_id: int, summary_line: str) -> list[str]:
	return [
		f'1  1 JAN 2026,  0:00  SEQUENCE NO.{seq:5d}, ID NO.{event_id:10d}',
		'              EIGENVALUES',
		'     (18.036  1.950  1.683  1.245)',
		_ERROR_ELLIPSE_LINE,
		_SUMMARY_HEADER_LINE,
		summary_line,
		_NSTA_HEADER_LINE,
		_NSTA_VALUES_LINE,
	]


def _unsolved_event_lines(seq: int, event_id: int) -> list[str]:
	return [
		f'1  1 NOV 2025,  6:10  SEQUENCE NO.{seq:5d}, ID NO.{event_id:10d}',
		f'*** 3 PHASES CANT SOLVE SEQUENCE NO.{seq:5d}, ID NO.{event_id:10d}',
		_SUMMARY_HEADER_LINE,
		_UNSOLVED_SUMMARY_LINE,
		'*** ABANDON EVENT WITH ONLY 3 READINGS:  25 11  1  6 10',
	]


def test_parse_summary_line_parses_normal_longitude_with_hemisphere() -> None:
	rec_e = parse_summary_line(_SOLVED_SUMMARY_LINE_E)
	assert rec_e['lon_deg_hyp'] == pytest.approx(140 + 2.04 / 60.0)

	rec_w = parse_summary_line(_SOLVED_SUMMARY_LINE_W)
	assert rec_w['lon_deg_hyp'] == pytest.approx(-(140 + 2.04 / 60.0))


def test_parse_summary_line_allows_blank_hemisphere_only_for_dummy_zero_longitude(
) -> None:
	rec = parse_summary_line(_UNSOLVED_SUMMARY_LINE)
	assert rec['lat_deg_hyp'] == 0.0
	assert rec['lon_deg_hyp'] == 0.0

	invalid_line = _SOLVED_SUMMARY_LINE_E[:38] + ' ' + _SOLVED_SUMMARY_LINE_E[39:]
	with pytest.raises(ValueError, match='invalid longitude fields'):
		parse_summary_line(invalid_line)


def test_load_hypoinverse_summary_from_prt_reads_solved_event_only(
	tmp_path: Path,
) -> None:
	prt = _write_prt(
		tmp_path,
		'solved_only.prt',
		_solved_event_lines(1, 1, _SOLVED_SUMMARY_LINE_E),
	)

	df = load_hypoinverse_summary_from_prt(prt)

	assert len(df) == 1
	assert df['seq'].tolist() == [1]
	assert df.loc[0, 'NSTA'] == 220
	assert df.loc[0, 'eig_adj1'] == pytest.approx(18.036, rel=0, abs=1e-12)
	assert '_skip_event' not in df.columns


def test_load_hypoinverse_summary_from_prt_skips_unsolved_and_abandoned_events(
	tmp_path: Path,
) -> None:
	lines = (
		_solved_event_lines(1, 1, _SOLVED_SUMMARY_LINE_E)
		+ _unsolved_event_lines(2, 10)
		+ _solved_event_lines(3, 11, _SOLVED_SUMMARY_LINE_2)
	)
	prt = _write_prt(tmp_path, 'mixed_events.prt', lines)

	df = load_hypoinverse_summary_from_prt(prt)

	assert len(df) == 2
	assert df['seq'].tolist() == [1, 2]
	assert df['origin_time_hyp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist() == [
		'2026-01-01 00:00:00',
		'2026-01-01 00:01:10',
	]
	assert (df['lat_deg_hyp'] > 0).all()
	assert (df['lon_deg_hyp'] != 0).all()
	assert '_skip_event' not in df.columns


def test_load_hypoinverse_summary_from_prt_fails_when_all_events_are_unsolved(
	tmp_path: Path,
) -> None:
	prt = _write_prt(
		tmp_path,
		'all_unsolved.prt',
		_unsolved_event_lines(6, 10),
	)

	with pytest.raises(
		RuntimeError,
		match=r'no valid solved events parsed from \.prt',
	):
		load_hypoinverse_summary_from_prt(prt)
