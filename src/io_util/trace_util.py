def station_key(network: str | None, station: str) -> str:
	return f'{network}.{station}' if network else station


def trace_station_comp(tr) -> tuple[str, str]:
	net = getattr(tr.stats, 'network', None)
	sta = getattr(tr.stats, 'station', None)
	cha = getattr(tr.stats, 'channel', None)
	if sta is None or cha is None:
		raise ValueError('trace.stats.station/channel missing')

	sta_full = station_key(str(net) if net is not None else None, str(sta))
	comp = str(cha)[-1]  # ...U / ...N / ...E
	return sta_full, comp
