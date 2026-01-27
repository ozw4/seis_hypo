"""EqT backend helpers."""

from pick.stream_io import station_zne_from_stream as _station_zne_from_stream

__all__ = ['station_zne_from_stream']


def station_zne_from_stream(*args, **kwargs):
	kwargs.setdefault('log_label', 'EqT')
	return _station_zne_from_stream(*args, **kwargs)
