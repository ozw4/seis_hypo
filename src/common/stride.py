def normalize_channel_stride(channel_stride: int | None) -> int | None:
	if channel_stride is None:
		return None
	stride = int(channel_stride)
	if stride <= 0:
		raise ValueError(f'channel_stride must be >= 1 when set, got {stride}')
	if stride <= 1:
		return None
	return stride
