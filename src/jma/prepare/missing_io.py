from __future__ import annotations

import csv
from collections.abc import Callable, Iterator
from pathlib import Path

_Normalizer = Callable[[str], str]


def _iter_missing_rows(path: Path) -> Iterator[tuple[str, str, str]]:
	for raw in Path(path).read_text(encoding='utf-8').splitlines():
		line = raw.strip()
		if not line:
			continue
		parts = next(
			csv.reader([line], delimiter='\t', quoting=csv.QUOTE_NONE)
		)
		if len(parts) != 2:
			raise ValueError(
				f'invalid line (expected 2 TSV fields) in {path.name}: {raw}'
			)
		station = parts[0].strip()
		network_code = parts[1].strip()
		if not station or not network_code:
			raise ValueError(
				f'invalid station/network_code in {path.name}: {raw}'
			)
		yield raw, station, network_code


def read_missing_pairs(
	path: Path,
	*,
	normalize_station: _Normalizer | None = None,
	normalize_network: _Normalizer | None = None,
) -> list[tuple[str, str]]:
	out: list[tuple[str, str]] = []
	seen: set[tuple[str, str]] = set()

	for raw, station, network_code in _iter_missing_rows(path):
		sta = normalize_station(station) if normalize_station else station
		net = normalize_network(network_code) if normalize_network else network_code
		if not sta or not net:
			raise ValueError(
				f'invalid station/network_code in {path.name}: {raw}'
			)
		key = (sta, net)
		if key in seen:
			continue
		seen.add(key)
		out.append(key)

	return out


def read_missing_by_network(
	path: Path,
	*,
	normalize_station: _Normalizer | None = None,
	normalize_network: _Normalizer | None = None,
) -> dict[str, list[str]]:
	out: dict[str, list[str]] = {}
	pairs = read_missing_pairs(
		path,
		normalize_station=normalize_station,
		normalize_network=normalize_network,
	)
	for sta, net in pairs:
		out.setdefault(net, []).append(sta)

	for net in list(out.keys()):
		out[net] = sorted(out[net])

	return out
