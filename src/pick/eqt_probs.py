# file: src/pick/eqt_probs.py
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from obspy import Stream
from scipy.signal import resample_poly
from seisbench.models import EQTransformer

from pick.overlap import stack_overlap_1d
from pick.probs_common import (
	build_probs_by_station_common,
	iterate_overlapping_windows,
	normalize_zne,
)
from pick.weights_util import _extract_state_dict, _is_local_weights_spec
from waveform.filters import zscore_tracewise

_EQT_MODELS: dict[tuple[str, int], EQTransformer] = {}


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
		stack_overlap_1d(det, d0, s0i)
		stack_overlap_1d(probP, p0, s0i)
		stack_overlap_1d(probS, s0s, s0i)

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
	zne = normalize_zne(zne)

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
		return zscore_tracewise(t, axis=-1, eps=1e-6)

	with torch.no_grad():
		iterate_overlapping_windows(
			zne,
			window_len=L,
			hop_len=H,
			batch_size=int(batch_size),
			to_tensor=_to_tensor,
			process_batch=lambda b: _stitch_eqt_batch(b, model, det, probP, probS),
		)

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
	backend_kwargs = {
		'weights': str(eqt_weights),
		'in_samples': int(eqt_in_samples),
		'overlap': int(eqt_overlap),
		'batch_size': int(eqt_batch_size),
		'log_label': 'EqT',
	}
	return build_probs_by_station_common(
		st,
		fs=float(fs),
		backend_fn=backend_eqt_probs,
		backend_kwargs=backend_kwargs,
	)
