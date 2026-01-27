# file: src/pipelines/loki_stalta_pipelines.py
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from obspy import Stream

from common.config import LokiWaveformStackingInputs, LokiWaveformStackingPipelineConfig
from common.json_io import write_json
from common.stride import normalize_channel_stride
from io_util.stream import build_stream_from_forge_event_npy
from loki_tools.build_loki import build_loki_with_header
from loki_tools.loki_parse import read_phs_token_by_station
from loki_tools.prob_stream import build_loki_ps_prob_stream
from pick.stalta_probs import (  # type: ignore
	StaltaProbSpec,
	build_probs_by_station_stalta,
)
from pipelines.loki_waveform_stacking_pipelines import (
	list_event_dirs_filtered_forge_das,
)
from waveform.preprocess import preprocess_stream_detrend_bandpass, spec_from_inputs


def _subsample_stream_by_stride(st: Stream, *, stride: int) -> tuple[Stream, list[int]]:
	kept_indices = list(range(0, len(st), stride))
	st_sub = Stream(traces=[st[i] for i in kept_indices])
	return st_sub, kept_indices


def _require_one_trial_phs(event_out_dir: Path, *, trial: int) -> Path:
	phs = sorted(event_out_dir.glob(f'*trial{int(trial)}.phs'))
	if not phs:
		raise FileNotFoundError(f'no *trial{trial}.phs in {event_out_dir}')
	if len(phs) != 1:
		raise ValueError(
			f'multiple *trial{trial}.phs in {event_out_dir}: {[p.name for p in phs]}'
		)
	return phs[0]


def _sort_json_obj(obj: object) -> object:
	if isinstance(obj, dict):
		return {k: _sort_json_obj(obj[k]) for k in sorted(obj, key=str)}
	if isinstance(obj, list):
		return [_sort_json_obj(v) for v in obj]
	return obj


def _ones_prob(npts: int) -> np.ndarray:
	if npts <= 0:
		raise ValueError(f'npts must be > 0, got {npts}')
	return np.ones(int(npts), dtype=np.float32)


def _reset_dir_empty(root: Path) -> None:
	root = Path(root)
	root.mkdir(parents=True, exist_ok=True)
	for p in root.iterdir():
		if p.is_dir():
			shutil.rmtree(p)
		else:
			p.unlink()


def _initialize_loki_direct_input(
	cfg: LokiWaveformStackingPipelineConfig,
	*,
	output_subdir: str,
) -> tuple[object, Path, Path]:
	out_dir = Path(cfg.loki_output_path) / str(output_subdir)
	out_dir.mkdir(parents=True, exist_ok=True)

	stream_data_root = Path(cfg.loki_data_path) / '_streaming_direct_input'
	loki, _header, _header_path = build_loki_with_header(
		cfg,
		data_path=stream_data_root,
		output_path=out_dir,
	)
	_reset_dir_empty(stream_data_root)
	print(f'[STALTA-PASS1-DAS] output: {out_dir}')
	print(f'[STALTA-PASS1-DAS] streaming data_path: {stream_data_root}')
	return loki, out_dir, stream_data_root


def prepare_stalta_prob_stream(
	event_dir: Path,
	*,
	event_name: str,
	component: str,
	channel_prefix: str,
	das_channel_code: str,
	pre_enable: bool,
	pre_spec: object,
	fs_expected: float,
	p_spec: StaltaProbSpec,
	stride: int | None,
) -> tuple[Stream, dict[str, int]]:
	st = build_stream_from_forge_event_npy(
		event_dir,
		channel_code=str(das_channel_code),
	)
	orig_n_channels = len(st)
	kept_n_channels = orig_n_channels
	if stride is not None:
		st, kept_indices = _subsample_stream_by_stride(st, stride=stride)
		kept_n_channels = len(st)
		print(
			f'[STALTA-PASS1-DAS] channel stride enabled: event={event_name} '
			f'stride={stride} kept={kept_n_channels} original={orig_n_channels} '
			f'indices={kept_indices[:10]}'
			f'{"..." if len(kept_indices) > 10 else ""}'
		)

	if pre_enable:
		preprocess_stream_detrend_bandpass(
			st,
			spec=pre_spec,
			fs_expected=fs_expected,
		)
	elif abs(float(st[0].stats.sampling_rate) - fs_expected) > 1e-6:
		raise ValueError(
			f'sampling_rate mismatch: event={event_name} '
			f'fs={st[0].stats.sampling_rate} expected={fs_expected}'
		)

	probs_p = build_probs_by_station_stalta(
		st,
		fs=fs_expected,
		component=str(component),
		phase='P',
		spec=p_spec,
	)

	npts = int(st[0].stats.npts)

	# Pだけ作ったprobを、S=1で埋めて direct_input の前提(P/S両方)を満たす
	probs_ps: dict[str, dict[str, np.ndarray]] = {}
	ones = _ones_prob(npts)
	for sta, d in probs_p.items():
		p = d.get('P')
		if p is None:
			raise ValueError(f'missing P prob at station={sta} event={event_name}')
		pp = np.asarray(p, dtype=np.float32)
		if pp.ndim != 1 or pp.size != npts:
			raise ValueError(
				f'invalid P prob shape at station={sta} event={event_name} '
				f'got={pp.shape} expected=({npts},)'
			)
		probs_ps[str(sta)] = {'P': pp, 'S': ones.copy()}

	st_prob_ps = build_loki_ps_prob_stream(
		ref_stream=st,
		probs_by_station=probs_ps,
		channel_prefix=str(channel_prefix),
		require_both_ps=True,
	)

	print(
		f'[STALTA-PASS1-DAS] prepared prob stream: event={event_name} '
		f'n_traces={len(st_prob_ps)} stations={len(probs_ps)} '
		f'pre={"on" if pre_enable else "off"} dir={event_dir}'
	)

	return st_prob_ps, {
		'channels_original': int(orig_n_channels),
		'channels_kept': int(kept_n_channels),
	}


def run_loki_for_event(
	loki: object,
	cfg: LokiWaveformStackingPipelineConfig,
	*,
	event_name: str,
	stream_data_root: Path,
	out_dir: Path,
	st_prob_ps: Stream,
	loki_kwargs: dict[str, object],
	trial: int,
) -> tuple[Path, Path]:
	_reset_dir_empty(stream_data_root)
	event_tmp_dir = stream_data_root / event_name
	event_tmp_dir.mkdir(parents=True, exist_ok=True)

	# LOKIの実装差（event名 or event_pathで引く）を吸収するため、キーを複数張る
	streams_by_event = {
		event_name: st_prob_ps,
		str(event_tmp_dir): st_prob_ps,
		event_tmp_dir: st_prob_ps,
	}

	# 念のため、LOKI側が保持するイベントリストを上書きできるなら1件に固定
	if hasattr(loki, 'data_tree'):
		loki.data_tree = [str(event_tmp_dir)]
	if hasattr(loki, 'events'):
		loki.events = [str(event_name)]

	loki.location(
		extension=cfg.extension,
		comp=['P', 'S'],
		precision=cfg.precision,
		search=cfg.search,
		streams_by_event=streams_by_event,
		**loki_kwargs,
	)

	ev_out_dir = out_dir / event_name
	phs_path = _require_one_trial_phs(ev_out_dir, trial=int(trial))
	return ev_out_dir, phs_path


def write_pick_json(
	*,
	event_name: str,
	ev_out_dir: Path,
	phs_path: Path,
	trial: int,
	pick_json_name: str,
	channel_stride: int | None,
	channels_original: int,
	channels_kept: int,
) -> Path:
	# phase='P'/'S' と station -> token(str) 返却の仕様は旧実装と一致させる。
	p_tok = read_phs_token_by_station(phs_path, phase='P')

	out_json = ev_out_dir / str(pick_json_name)
	obj: dict[str, object] = {
		'format': 'loki-stalta-pass1-picks-v1',
		'event_id': str(event_name),
		'trial': int(trial),
		'phs_filename': phs_path.name,
		'phase_run': 'P',
		'p_token_by_station': p_tok,
		's_token_by_station': {},
	}
	if channel_stride is not None:
		obj['channel_stride'] = int(channel_stride)
		obj['channels_original'] = int(channels_original)
		obj['channels_kept'] = int(channels_kept)
	write_json(
		out_json,
		_sort_json_obj(obj),
		ensure_ascii=False,
		indent=2,
	)
	with out_json.open('a', encoding='utf-8') as f:
		f.write('\n')

	print(
		f'[STALTA-PASS1-DAS] saved picks json: event={event_name} '
		f'stations={len(p_tok)} path={out_json}'
	)
	return out_json


def pipeline_loki_waveform_stacking_stalta_pass1(
	cfg: LokiWaveformStackingPipelineConfig,
	inputs: LokiWaveformStackingInputs,
	*,
	# DASは1成分Zが前提（build_stream_from_forge_event_npyが channel_code末尾Zを要求）
	component: str = 'Z',
	# LOKI direct_input用のprob streamは channel末尾が P/S になる必要がある（例: HHP/HHS）
	channel_prefix: str = 'HH',
	# ForgeDAS入力のTrace.stats.channel（末尾Z必須）
	das_channel_code: str = 'DASZ',
	output_subdir: str = 'pass1_stalta_p',
	trial: int = 0,
	pick_json_name: str = 'pass1_picks_trial0.json',
	p_spec: object | None = None,
	channel_stride: int | None = None,
) -> dict[str, Path]:
	"""ForgeDAS入力で STALTA direct_input の Pass1(P重視run) を逐次実行し、trialの .phs を JSON に保存する。

	重要:
	- LOKI direct_input は内部で P/S 両方を参照するため、Sはニュートラル(定数1)系列を付与して回す。
	- comp は ['P','S'] で回す（Pのみは KeyError になる）。
	- LOKIが data_path を走査してイベントを決める実装に備え、逐次用の隔離data_pathを使う。
	"""
	l1, out_dir, stream_data_root = _initialize_loki_direct_input(
		cfg,
		output_subdir=str(output_subdir),
	)

	event_dirs = list_event_dirs_filtered_forge_das(cfg)

	pre_enable = bool(getattr(inputs, 'pre_enable', True))
	pre_spec = spec_from_inputs(inputs)
	fs_expected = float(inputs.base_sampling_rate_hz)

	if p_spec is None:
		p_spec = StaltaProbSpec(
			transform='raw',
			sta_sec=0.1,
			lta_sec=0.5,
			smooth_sec=None,
			clip_p=99.5,
			log1p=False,
		)

	loki_kwargs: dict[str, object] = {
		'npr': int(getattr(inputs, 'npr', 2)),
		'model': str(getattr(inputs, 'model', 'jma2001')),
	}
	stride = normalize_channel_stride(channel_stride)

	pick_json_by_event: dict[str, Path] = {}

	for event_dir in event_dirs:
		event_name = event_dir.name

		st_prob_ps, channel_meta = prepare_stalta_prob_stream(
			event_dir,
			event_name=event_name,
			component=str(component),
			channel_prefix=str(channel_prefix),
			das_channel_code=str(das_channel_code),
			pre_enable=pre_enable,
			pre_spec=pre_spec,
			fs_expected=fs_expected,
			p_spec=p_spec,
			stride=stride,
		)
		ev_out_dir, phs_path = run_loki_for_event(
			l1,
			cfg,
			event_name=event_name,
			stream_data_root=stream_data_root,
			out_dir=out_dir,
			st_prob_ps=st_prob_ps,
			loki_kwargs=loki_kwargs,
			trial=int(trial),
		)
		out_json = write_pick_json(
			event_name=event_name,
			ev_out_dir=ev_out_dir,
			phs_path=phs_path,
			trial=int(trial),
			pick_json_name=str(pick_json_name),
			channel_stride=stride,
			channels_original=channel_meta['channels_original'],
			channels_kept=channel_meta['channels_kept'],
		)
		pick_json_by_event[str(event_name)] = out_json

	return pick_json_by_event
