from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def _as_pair_floats(value: Sequence[float], label: str) -> tuple[float, float]:
	if len(value) != 2:
		raise ValueError(f'{label} must have length 2, got {len(value)}')
	return (float(value[0]), float(value[1]))


def _as_pair_latlon(value: Sequence[float], label: str) -> tuple[float, float]:
	lat, lon = _as_pair_floats(value, label)
	if not (-90.0 <= lat <= 90.0):
		raise ValueError(f'{label}[lat] out of range: {lat}')
	if not (-180.0 <= lon <= 180.0):
		raise ValueError(f'{label}[lon] out of range: {lon}')
	return (lat, lon)


def _as_range(value: Sequence[float], label: str) -> tuple[float, float]:
	v0, v1 = _as_pair_floats(value, label)
	if v1 < v0:
		raise ValueError(f'{label} must be [min, max], got {value}')
	return (v0, v1)


@dataclass(frozen=True)
class PlotConfig:
	"""可視化用の領域・フィルタ設定。

	YAML想定:
		mobara_default:
			lon_range: [139.0, 141.5]
			lat_range: [34.8, 36.3]
			depth_range: [0.0, 150.0]
			well_coord: [35.511111, 140.1925]  # [lat, lon]
			min_mag: null
			max_mag: null

		japan_default:
			lon_range: [128.0, 151.0]
			lat_range: [30.0, 46.0]
			depth_range: [0.0, 700.0]
			well_coord: null
			min_mag: null
			max_mag: null
	"""

	lon_range: tuple[float, float]
	lat_range: tuple[float, float]
	depth_range: tuple[float, float]
	well_coord: tuple[float, float] | None = None  # [lat, lon]
	min_mag: float | None = None
	max_mag: float | None = None

	def __post_init__(self) -> None:
		lon_r = _as_range(self.lon_range, 'lon_range')
		lat_r = _as_range(self.lat_range, 'lat_range')
		dep_r = _as_range(self.depth_range, 'depth_range')

		if self.well_coord is None:
			well = None
		else:
			well = _as_pair_latlon(self.well_coord, 'well_coord')

		min_mag = float(self.min_mag) if self.min_mag is not None else None
		max_mag = float(self.max_mag) if self.max_mag is not None else None

		if min_mag is not None and max_mag is not None and max_mag < min_mag:
			raise ValueError(f'max_mag must be >= min_mag, got {min_mag}, {max_mag}')

		object.__setattr__(self, 'lon_range', lon_r)
		object.__setattr__(self, 'lat_range', lat_r)
		object.__setattr__(self, 'depth_range', dep_r)
		object.__setattr__(self, 'well_coord', well)
		object.__setattr__(self, 'min_mag', min_mag)
		object.__setattr__(self, 'max_mag', max_mag)
