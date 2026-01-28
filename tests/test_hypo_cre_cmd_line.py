from __future__ import annotations

import pytest

from hypo.cre import format_cre_cmd_line
from hypo.synth_eval.hypoinverse_runner import _parse_cmd_model_number


def test_format_cre_cmd_line_true_flag_and_rounding() -> None:
	s = format_cre_cmd_line(1, 'P.cre', 0.1234567, use_station_elev=True)
	assert s == "CRE 1 'P.cre' 0.123457 T"


def test_format_cre_cmd_line_false_flag() -> None:
	s = format_cre_cmd_line(2, 'S.cre', 1.0, use_station_elev=False)
	assert s == "CRE 2 'S.cre' 1.000000 F"


@pytest.mark.parametrize('bad', [0, 3, -1, 99])
def test_format_cre_cmd_line_rejects_invalid_model_id(bad: int) -> None:
	with pytest.raises(ValueError):
		format_cre_cmd_line(bad, 'P.cre', 0.0)


@pytest.mark.parametrize(
	'token,expected',
	[
		('1', 1),
		(' 2 ', 2),
		('+1', 1),
		('-2', -2),
	],
)
def test_parse_cmd_model_number_accepts_signed_int(token: str, expected: int) -> None:
	assert _parse_cmd_model_number(token) == expected


@pytest.mark.parametrize('token', ['', ' ', 'x', '1x', '+', '--1', '1.0'])
def test_parse_cmd_model_number_rejects_invalid(token: str) -> None:
	with pytest.raises(ValueError):
		_parse_cmd_model_number(token)
