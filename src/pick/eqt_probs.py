# file: src/pick/eqt_probs.py
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from obspy import Stream
from scipy.signal import resample_poly
from seisbench.models import EQTransformer

from pick.eqt_io import station_zne_from_stream
from waveform.filters import zscore_channelwise

_EQT_MODELS: dict[tuple[str, int], EQTransformer] = {}


def _is_local_weights_spec(weights: str) -> bool:
	w = str(weights).strip()
	if not w:
		return False
	if w.startswith(('~', '/', './', '../')):
		return True
	if ('/' in w) or ('\\' in w):
		return True
	suf = Path(w).suffix.lower()
	return suf in {'.pt', '.pth', '.ckpt'}


def _extract_state_dict(obj: object) -> Mapping[str, torch.Tensor]:
	if isinstance(obj, Mapping):
		# 純粋な state_dict
		if obj and all(isinstance(k, str) for k in obj.keys()):
			vals = list(obj.values())
			if vals and all(torch.is_tensor(v) for v in vals):
				return obj  # type: ignore[return-value]

		# よくあるチェックポイント形式: {"state_dict": ...}
		if 'state_dict' in obj and isinstance(obj['state_dict'], Mapping):
			sd = obj['state_dict']
			if sd and all(isinstance(k, str) for k in sd.keys()):
				vals = list(sd.values())
				if vals and all(torch.is_tensor(v) for v in vals):
					return sd  # type: ignore[return-value]

	raise ValueError(
		'Invalid checkpoint format. Please save a pure state_dict (Mapping[str, Tensor]) '
		'or a dict with a "state_dict" key containing that mapping.'
	)


def _get_eqt(weights: str, in_samples: int) -> EQTransformer:
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	in_samples_i = int(in_samples)

	if _is_local_weights_spec(weights):
		p = Path(str(weights)).expanduser()
		if not p.is_file():
			raise FileNotFoundError(f'EqT weights file not found: {p}')

		key = (f'file:{p.resolve()}', in_samples_i)
		m = _EQT_MODELS.get(key)
		if m is not None:
			return m

		# ユーザー希望の形式
		model = EQTransformer(in_channels=3, in_samples=in_samples_i)
		obj = torch.load(str(p), map_location=device)
		state_dict = _extract_state_dict(obj)
		model.load_state_dict(state_dict)
		model.eval().to(device)

		_EQT_MODELS[key] = model
		return model

	# seisbench の pretrained 名
	key = (f'pretrained:{weights}', in_samples_i)
	m = _EQT_MODELS.get(key)
	if m is not None:
		return m

	model = EQTransformer.from_pretrained(weights)
	if int(model.in_samples) != in_samples_i:
		raise ValueError(
			f'in_samples mismatch: model={model.in_samples} requested={in_samples_i}'
		)

	model.eval().to(device)
	_EQT_MODELS[key] = model
	return model


def _stack_overlap_max(dst: np.ndarray, src: np.ndarray, start: int) -> None:
	"""dst[start:start+len(src)] に src を max で縫い付ける（NaNは未埋め扱い）。

	- dst は最終的に欲しい長さ（例: N_eff）
	- src はモデル出力窓の長さ（例: in_samples=6000）
	- N_eff < in_samples のときは、dst 範囲に収まる分だけ src を切って縫う
	"""
	if dst.ndim != 1 or src.ndim != 1:
		raise ValueError(f'dst/src must be 1D: dst={dst.shape} src={src.shape}')
	if start < 0:
		raise ValueError(f'start must be >= 0, got {start}')

	n_dst = int(dst.shape[0])
	if n_dst == 0:
		return
	if start >= n_dst:
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
	if cur_nan.all():
		dst[sl] = src2
		return

	src_nan = np.isnan(src2)
	out = cur.copy()

	mask_src_only = cur_nan & ~src_nan
	out[mask_src_only] = src2[mask_src_only]

	mask_both = ~cur_nan & ~src_nan
	out[mask_both] = np.maximum(out[mask_both], src2[mask_both])

	dst[sl] = out


def _stitch_eqt_batch(
	buf: list[tuple[int, torch.Tensor]],
	model: EQTransformer,
	det: np.ndarray,
	probP: np.ndarray,
	probS: np.ndarray,
) -> None:
	starts, tensors = zip(*buf, strict=False)
	B = torch.cat(list(tensors), dim=0)  # (B, C, L)

	y_det, y_p, y_s = model(B)
	y_det = y_det.detach().cpu().numpy()
	y_p = y_p.detach().cpu().numpy()
	y_s = y_s.detach().cpu().numpy()

	for s0, d0, p0, s0s in zip(starts, y_det, y_p, y_s, strict=False):
		s0i = int(s0)
		_stack_overlap_max(det, d0, s0i)
		_stack_overlap_max(probP, p0, s0i)
		_stack_overlap_max(probS, s0s, s0i)

	buf.clear()


ScoreKind = Literal['P', 'S', 'D', 'event_max', 'event_sum']


def backend_eqt_probs(
	zne: np.ndarray,
	fs: int,
	*,
	weights: str = 'original',
	in_samples: int = 6000,
	overlap: int = 3000,
	batch_size: int = 64,
	target_fs: int = 100,
	score_kind: ScoreKind = 'P',
) -> tuple[np.ndarray, int, dict[str, Any]]:
	"""EqTransformerで P/S/Detection の確率時系列を生成。

	入力:
	  - zne: (3, N) または (N, 3)
	  - fs: 入力サンプリング[Hz]

	出力:
	  - score: score_kindで選んだ1系列（LOKI統合時はダミーでも可）
	  - delay: 0
	  - meta['probs']: {'P','S','D'}（いずれも length N_eff）
	"""
	if zne.ndim != 2:
		raise ValueError(f'zne must be 2D, got shape={zne.shape}')

	C, N = zne.shape
	if C != 3 and zne.shape[1] == 3:
		zne = zne.T
		C, N = zne.shape

	if C == 1:
		zne = np.vstack([zne, np.zeros((2, N), dtype=zne.dtype)])
		C, N = zne.shape

	if C != 3:
		raise ValueError(f'expected 3 components, got C={C} shape={zne.shape}')

	if int(fs) != int(target_fs):
		zne = resample_poly(zne, up=int(target_fs), down=int(fs), axis=1)
		fs_eff = int(target_fs)
	else:
		fs_eff = int(fs)

	N_eff = int(zne.shape[1])

	L = int(in_samples)
	H = L - int(overlap)
	if H <= 0:
		raise ValueError('overlap must be smaller than in_samples')

	model = _get_eqt(weights, int(in_samples))
	device = next(model.parameters()).device

	det = np.full(N_eff, np.nan, dtype=np.float32)
	probP = np.full(N_eff, np.nan, dtype=np.float32)
	probS = np.full(N_eff, np.nan, dtype=np.float32)

	def _to_tensor(w: np.ndarray) -> torch.Tensor:
		t = torch.from_numpy(w[None, :, :]).to(device)
		return zscore_channelwise(t, axis=-1, eps=1e-6)

	with torch.no_grad():
		buf: list[tuple[int, torch.Tensor]] = []

		if N_eff < L:
			w = np.zeros((3, L), dtype=np.float32)
			w[:, :N_eff] = zne[:, :N_eff].astype(np.float32, copy=False)
			buf.append((0, _to_tensor(w)))
			_stitch_eqt_batch(buf, model, det, probP, probS)
		else:
			for s in range(0, N_eff - L + 1, H):
				w = zne[:, s : s + L].astype(np.float32, copy=False)
				buf.append((int(s), _to_tensor(w)))
				if len(buf) >= int(batch_size):
					_stitch_eqt_batch(buf, model, det, probP, probS)
			if buf:
				_stitch_eqt_batch(buf, model, det, probP, probS)

	det = np.nan_to_num(det, nan=0.0)
	probP = np.nan_to_num(probP, nan=0.0)
	probS = np.nan_to_num(probS, nan=0.0)

	probs = {'D': det, 'P': probP, 'S': probS}

	if score_kind == 'P':
		score = probP.astype(np.float64, copy=False)
	elif score_kind == 'S':
		score = probS.astype(np.float64, copy=False)
	elif score_kind == 'D':
		score = det.astype(np.float64, copy=False)
	elif score_kind == 'event_sum':
		score = (probP + probS).astype(np.float64)
	else:  # event_max
		score = np.maximum(probP, probS).astype(np.float64)

	meta: dict[str, Any] = {
		'kind': 'eqtransformer_probs',
		'weights': weights,
		'in_samples': int(in_samples),
		'overlap': int(overlap),
		'batch_size': int(batch_size),
		'fs_eff': int(fs_eff),
		'probs': probs,
	}
	return score, 0, meta


def build_probs_by_station(
	st: Stream,
	*,
	fs: float,
	eqt_weights: str,
	eqt_in_samples: int,
	eqt_overlap: int,
	eqt_batch_size: int,
) -> dict[str, dict[str, np.ndarray]]:
	zne_by_sta = station_zne_from_stream(st)

	probs_by_sta: dict[str, dict[str, np.ndarray]] = {}
	npts = int(st[0].stats.npts)

	for sta, zne in zne_by_sta.items():
		score, delay, meta = backend_eqt_probs(
			zne,
			float(fs),
			weights=str(eqt_weights),
			in_samples=int(eqt_in_samples),
			overlap=int(eqt_overlap),
			batch_size=int(eqt_batch_size),
		)

		probs = meta.get('probs', None)
		if not isinstance(probs, dict):
			raise ValueError("meta['probs'] missing or invalid")

		p = probs.get('P', None)
		s = probs.get('S', None)
		if p is None or s is None:
			raise ValueError(f'missing P/S probs: station={sta}')

		p = np.asarray(p, dtype=np.float32)
		s = np.asarray(s, dtype=np.float32)
		if p.ndim != 1 or s.ndim != 1:
			raise ValueError(f'P/S probs must be 1D: station={sta}')
		if int(p.shape[0]) != npts or int(s.shape[0]) != npts:
			raise ValueError(
				f'P/S probs length mismatch: station={sta} got={(p.shape[0], s.shape[0])} expected={npts}'
			)

		probs_by_sta[sta] = {'P': p, 'S': s}

	if not probs_by_sta:
		raise ValueError('no station probs built')

	return probs_by_sta
