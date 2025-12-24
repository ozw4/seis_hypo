import numpy as np


def _stack_overlap_1d(
	dst: np.ndarray, src: np.ndarray, start: int, mode: str = 'max'
) -> None:
	"""dst[start:start+len(src)] に src を縫い付ける（NaNは未埋め扱い）。

	`dst` は最終的に欲しい長さ（例: N_eff）、`src` はモデル出力窓の長さ（例: in_samples=6000）。
	N_eff < in_samples のときは dst 範囲に収まる分だけ src を切って縫う。

	mode:
	  - "max": 重複区間は max で合成（ピークを潰しにくい）
	  - "mean": 重複区間は平均で合成（平滑化寄り、ピークは下がりやすい）

	NaNの扱い:
	  - dst側がNaN、src側が非NaN -> srcで埋める
	  - 両方非NaN -> modeに応じて合成
	  - src側がNaN -> 何もしない（dstを維持）
	"""
	if dst.ndim != 1 or src.ndim != 1:
		raise ValueError(f'dst/src must be 1D: dst={dst.shape} src={src.shape}')
	if start < 0:
		raise ValueError(f'start must be >= 0, got {start}')
	if mode not in ('max', 'mean'):
		raise ValueError(f"mode must be 'max' or 'mean', got {mode}")

	n_dst = int(dst.shape[0])
	if n_dst == 0 or start >= n_dst:
		return

	n_src = int(src.shape[0])
	end = min(start + n_src, n_dst)
	n_put = end - start
	if n_put <= 0:
		return

	src2 = src[:n_put]
	sl = slice(start, end)

	cur = dst[sl]
	cur_nan = np.isnan(cur)
	src_nan = np.isnan(src2)

	# dst が全部 NaN なら src をそのまま入れる（srcのNaNも含めてコピーする）
	if cur_nan.all():
		dst[sl] = src2
		return

	out = cur.copy()

	# dst が NaN で src が有効なところは埋める
	mask_fill = cur_nan & ~src_nan
	out[mask_fill] = src2[mask_fill]

	# 両方有効なところは合成
	mask_both = ~cur_nan & ~src_nan
	if mask_both.any():
		if mode == 'max':
			out[mask_both] = np.maximum(out[mask_both], src2[mask_both])
		else:  # mean
			out[mask_both] = (out[mask_both] + src2[mask_both]) * 0.5

	dst[sl] = out
