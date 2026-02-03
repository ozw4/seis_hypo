from __future__ import annotations

from collections.abc import Callable

import numpy as np
from obspy import Stream

from pick.stalta_probs import StaltaProbSpec, build_probs_by_station_stalta

ProbDict = dict[str, dict[str, np.ndarray]]


def build_probs_by_station(
	picker_name: str,
	ref_stream: Stream,
	*,
	fs: float,
	component: str = 'U',
	phase: str = 'P',
	stalta_spec: StaltaProbSpec | None = None,
) -> ProbDict:
	"""Dispatcher to build station->phase probability series."""
	name = str(picker_name).strip().lower()
	dispatch: dict[str, Callable[..., ProbDict]] = {
		'stalta': _build_probs_stalta,
	}
	if name not in dispatch:
		raise ValueError(f'unsupported picker_name: {picker_name!r}')
	return dispatch[name](
		ref_stream,
		fs=fs,
		component=component,
		phase=phase,
		stalta_spec=stalta_spec,
	)


def _build_probs_stalta(
	ref_stream: Stream,
	*,
	fs: float,
	component: str,
	phase: str,
	stalta_spec: StaltaProbSpec | None,
) -> ProbDict:
	if stalta_spec is None:
		raise ValueError("stalta_spec is required when picker_name='stalta'")
	return build_probs_by_station_stalta(
		ref_stream,
		fs=fs,
		component=component,
		phase=phase,
		spec=stalta_spec,
	)
