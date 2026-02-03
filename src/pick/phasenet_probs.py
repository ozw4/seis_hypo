# file: src/pick/phasenet_probs.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from obspy import Stream
from scipy.signal import resample_poly
from seisbench.models import PhaseNet

from pick.overlap import stack_overlap_1d
from pick.probs_common import (
	extract_station_probs,
	iterate_overlapping_windows,
	normalize_zne,
)
from pick.stream_io import station_zne_from_stream
from pick.weights_util import _extract_state_dict, _is_local_weights_spec

_PHASENET_MODELS: dict[tuple[str, int], PhaseNet] = {}


def _get_phasenet(weights: str, in_samples: int) -> PhaseNet:
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	in_samples_i = int(in_samples)

	if _is_local_weights_spec(weights):
		p = Path(str(weights)).expanduser()
		if not p.is_file():
			raise FileNotFoundError(f'PhaseNet weights file not found: {p}')

		key = (f'file:{p.resolve()}', in_samples_i)
		m = _PHASENET_MODELS.get(key)
		if m is not None:
			return m

		model = PhaseNet()
		if int(model.in_samples) != in_samples_i:
			raise ValueError(
				f'in_samples mismatch: model={model.in_samples} requested={in_samples_i}'
			)
		obj = torch.load(str(p), map_location=device)
		state_dict = _extract_state_dict(obj)
		model.load_state_dict(state_dict)
		model.eval().to(device)

		_PHASENET_MODELS[key] = model
		return model

	key = (f'pretrained:{weights}', in_samples_i)
	m = _PHASENET_MODELS.get(key)
	if m is not None:
		return m

	model = PhaseNet.from_pretrained(str(weights))
	if int(model.in_samples) != in_samples_i:
		raise ValueError(
			f'in_samples mismatch: model={model.in_samples} requested={in_samples_i}'
		)

	model.eval().to(device)
	_PHASENET_MODELS[key] = model
	return model


def _labels_to_indices(labels: str) -> tuple[int, int, int]:
	lab = str(labels)
	if 'P' not in lab or 'S' not in lab:
		raise ValueError(
			f"PhaseNet labels must include 'P' and 'S', got labels={lab!r}"
		)
	idx_p = int(lab.index('P'))
	idx_s = int(lab.index('S'))
	idx_n = int(lab.index('N')) if 'N' in lab else -1
	return idx_n, idx_p, idx_s


def _stitch_phasenet_batch(
	buf: list[tuple[int, torch.Tensor]],
	model: PhaseNet,
	probN: np.ndarray,
	probP: np.ndarray,
	probS: np.ndarray,
	idx_n: int,
	idx_p: int,
	idx_s: int,
	*,
	overlap_mode: str,
) -> None:
	starts, tensors = zip(*buf, strict=False)
	B = torch.cat(list(tensors), dim=0)  # (B, C, L)

	y = model(B)  # (B, K, L)
	y = y.detach().cpu().numpy()

	for s0, y0 in zip(starts, y, strict=False):
		s0i = int(s0)
		if idx_n >= 0:
			stack_overlap_1d(probN, y0[idx_n], s0i, mode=overlap_mode)
		stack_overlap_1d(probP, y0[idx_p], s0i, mode=overlap_mode)
		stack_overlap_1d(probS, y0[idx_s], s0i, mode=overlap_mode)

	buf.clear()


ScoreKind = Literal['P', 'S', 'N', 'event_max', 'event_sum']


def backend_phasenet_probs(
	zne: np.ndarray,
	fs: float,
	*,
	weights: str = 'instance',
	in_samples: int = 3001,
	overlap: int = 1500,
	batch_size: int = 256,
	target_fs: int = 100,
	score_kind: ScoreKind = 'P',
	overlap_mode: Literal['max', 'mean'] = 'max',
) -> tuple[np.ndarray, int, dict[str, Any]]:
	"""PhaseNetで P/S（と任意でN）の確率時系列を生成。

	入力:
	  - zne: (3, N) または (N, 3)
	  - fs: 入力サンプリング[Hz]

	出力:
	  - score: score_kindで選んだ1系列（LOKI統合時はダミーでも可）
	  - delay: 0
	  - meta['probs']: {'P','S','N'}（いずれも length N_eff）
	"""
	zne = normalize_zne(zne)

	fs_i = int(round(float(fs)))
	if fs_i <= 0:
		raise ValueError(f'fs must be positive, got fs={fs}')

	if fs_i != int(target_fs):
		zne = resample_poly(zne, up=int(target_fs), down=int(fs_i), axis=1)
		fs_eff = int(target_fs)
	else:
		fs_eff = int(fs_i)

	N_eff = int(zne.shape[1])

	L = int(in_samples)
	H = L - int(overlap)
	if H <= 0:
		raise ValueError('overlap must be smaller than in_samples')

	model = _get_phasenet(weights, int(in_samples))
	device = next(model.parameters()).device
	idx_n, idx_p, idx_s = _labels_to_indices(getattr(model, 'labels', 'NPS'))

	probN = np.full(N_eff, np.nan, dtype=np.float32)
	probP = np.full(N_eff, np.nan, dtype=np.float32)
	probS = np.full(N_eff, np.nan, dtype=np.float32)

	def _to_tensor(w: np.ndarray) -> torch.Tensor:
		t = torch.from_numpy(w[None, :, :]).to(device)
		return model.annotate_batch_pre(t, {})

	with torch.no_grad():
		iterate_overlapping_windows(
			zne,
			window_len=L,
			hop_len=H,
			batch_size=int(batch_size),
			to_tensor=_to_tensor,
			process_batch=lambda b: _stitch_phasenet_batch(
				b,
				model,
				probN,
				probP,
				probS,
				idx_n,
				idx_p,
				idx_s,
				overlap_mode=overlap_mode,
			),
		)

	probN = np.nan_to_num(probN, nan=0.0)
	probP = np.nan_to_num(probP, nan=0.0)
	probS = np.nan_to_num(probS, nan=0.0)

	probs: dict[str, np.ndarray] = {'P': probP, 'S': probS}
	if idx_n >= 0:
		probs['N'] = probN

	if score_kind == 'P':
		score = probP.astype(np.float64, copy=False)
	elif score_kind == 'S':
		score = probS.astype(np.float64, copy=False)
	elif score_kind == 'N':
		score = probN.astype(np.float64, copy=False)
	elif score_kind == 'event_sum':
		score = (probP + probS).astype(np.float64)
	else:  # event_max
		score = np.maximum(probP, probS).astype(np.float64)

	meta: dict[str, Any] = {
		'kind': 'phasenet_probs',
		'weights': weights,
		'in_samples': int(in_samples),
		'overlap': int(overlap),
		'batch_size': int(batch_size),
		'fs_eff': int(fs_eff),
		'labels': str(getattr(model, 'labels', '')),
		'probs': probs,
	}
	return score, 0, meta


def build_probs_by_station(
	st: Stream,
	*,
	fs: float,
	phasenet_weights: str,
	phasenet_in_samples: int,
	phasenet_overlap: int,
	phasenet_batch_size: int,
	overlap_mode: Literal['max', 'mean'] = 'max',
) -> dict[str, dict[str, np.ndarray]]:
	zne_by_sta = station_zne_from_stream(st, log_label='PhaseNet')

	probs_by_sta: dict[str, dict[str, np.ndarray]] = {}
	npts = int(st[0].stats.npts)

	for sta, zne in zne_by_sta.items():
		score, delay, meta = backend_phasenet_probs(
			zne,
			float(fs),
			weights=str(phasenet_weights),
			in_samples=int(phasenet_in_samples),
			overlap=int(phasenet_overlap),
			batch_size=int(phasenet_batch_size),
			overlap_mode=overlap_mode,
		)

		probs_by_sta[sta] = extract_station_probs(meta, sta, npts)

	if not probs_by_sta:
		raise ValueError('no station probs built')

	return probs_by_sta
