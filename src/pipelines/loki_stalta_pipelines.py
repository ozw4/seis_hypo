# file: src/pipelines/loki_stalta_pipelines.py
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from loki.loki import Loki

from common.config import LokiWaveformStackingInputs, LokiWaveformStackingPipelineConfig
from io_util.stream import build_stream_from_forge_event_npy
from loki_tools.prob_stream import build_loki_ps_prob_stream
from pick.stalta_probs import (  # type: ignore
	StaltaProbSpec,
	build_probs_by_station_stalta,
)
from pipelines.loki_waveform_stacking_pipelines import (
	list_event_dirs_filtered_forge_das,
)
from waveform.preprocess import preprocess_stream_detrend_bandpass, spec_from_inputs


def _require_one_trial_phs(event_out_dir: Path, *, trial: int) -> Path:
	phs = sorted(event_out_dir.glob(f'*trial{int(trial)}.phs'))
	if not phs:
		raise FileNotFoundError(f'no *trial{trial}.phs in {event_out_dir}')
	if len(phs) != 1:
		raise ValueError(
			f'multiple *trial{trial}.phs in {event_out_dir}: {[p.name for p in phs]}'
		)
	return phs[0]


def _read_phs_token_by_station(phs_path: Path, *, phase: str) -> dict[str, str]:
	"""LOKI出力 .phs を station -> token の生文字列で読む（2列/3列以上に対応）。

	想定:
	- 2列: station token
	- 3列+: station Ptoken Stoken ...

	規約:
	- phase='P': cols[1]
	- phase='S': cols[2] if exists else cols[1]
	"""
	phs_path = Path(phs_path)
	phase_s = str(phase)
	if phase_s not in ('P', 'S'):
		raise ValueError(f"phase must be 'P' or 'S', got {phase!r}")

	out: dict[str, str] = {}
	for ln in phs_path.read_text(encoding='utf-8', errors='strict').splitlines():
		if not ln:
			continue
		if ln.startswith('#'):
			continue
		cols = ln.split()
		if not cols:
			continue
		if cols[0].lower() == 'station':
			continue
		if len(cols) < 2:
			raise ValueError(
				f"invalid .phs line (need >=2 cols): {phs_path} line='{ln}'"
			)

		sta = str(cols[0])

		if phase_s == 'P':
			tok = str(cols[1])
		else:
			tok = str(cols[2]) if len(cols) >= 3 else str(cols[1])

		if sta in out:
			raise ValueError(f'duplicate station in phs: station={sta} file={phs_path}')
		out[sta] = tok

	if not out:
		raise ValueError(f'no phs rows parsed: {phs_path}')
	return out


def _write_json(path: Path, obj: dict) -> None:
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	txt = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)
	path.write_text(txt + '\n', encoding='utf-8')


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
) -> dict[str, Path]:
	"""ForgeDAS入力で STALTA direct_input の Pass1(P重視run) を逐次実行し、trialの .phs を JSON に保存する。

	重要:
	- LOKI direct_input は内部で P/S 両方を参照するため、Sはニュートラル(定数1)系列を付与して回す。
	- comp は ['P','S'] で回す（Pのみは KeyError になる）。
	- LOKIが data_path を走査してイベントを決める実装に備え、逐次用の隔離data_pathを使う。
	"""
	out_dir = Path(cfg.loki_output_path) / str(output_subdir)
	out_dir.mkdir(parents=True, exist_ok=True)

	header_path = Path(cfg.loki_db_path) / Path(cfg.loki_hdr_filename)
	if not header_path.is_file():
		raise FileNotFoundError(f'header not found: {header_path}')

	event_dirs = list_event_dirs_filtered_forge_das(cfg)

	pre_enable = bool(getattr(inputs, 'pre_enable', True))
	pre_spec = spec_from_inputs(inputs)
	fs_expected = float(inputs.base_sampling_rate_hz)

	if p_spec is None:
		p_spec = StaltaProbSpec(transform='raw', sta_sec=0.2, lta_sec=2.0)

	loki_kwargs: dict[str, object] = {
		'npr': int(getattr(inputs, 'npr', 2)),
		'model': str(getattr(inputs, 'model', 'jma2001')),
	}

	# 逐次専用の隔離 data_path（ここに「今処理中の1イベント」だけ置く）
	stream_data_root = Path(cfg.loki_data_path) / '_streaming_direct_input'
	_reset_dir_empty(stream_data_root)

	l1 = Loki(
		str(stream_data_root),
		str(out_dir),
		str(cfg.loki_db_path),
		str(header_path),
		mode='locator',
	)
	print(f'[STALTA-PASS1-DAS] output: {out_dir}')
	print(f'[STALTA-PASS1-DAS] streaming data_path: {stream_data_root}')

	pick_json_by_event: dict[str, Path] = {}

	for event_dir in event_dirs:
		event_name = event_dir.name

		# data_path配下を「このイベントだけ」にする
		_reset_dir_empty(stream_data_root)
		event_tmp_dir = stream_data_root / event_name
		event_tmp_dir.mkdir(parents=True, exist_ok=True)

		st = build_stream_from_forge_event_npy(
			event_dir,
			channel_code=str(das_channel_code),
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
			probs_ps[str(sta)] = {'P': pp, 'S': ones}

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

		# LOKIの実装差（event名 or event_pathで引く）を吸収するため、キーを複数張る
		streams_by_event = {
			event_name: st_prob_ps,
			str(event_tmp_dir): st_prob_ps,
			event_tmp_dir: st_prob_ps,
		}

		# 念のため、LOKI側が保持するイベントリストを上書きできるなら1件に固定
		if hasattr(l1, 'data_tree'):
			l1.data_tree = [str(event_tmp_dir)]
		if hasattr(l1, 'events'):
			l1.events = [str(event_name)]

		l1.location(
			extension=cfg.extension,
			comp=['P', 'S'],
			precision=cfg.precision,
			search=cfg.search,
			streams_by_event=streams_by_event,
			**loki_kwargs,
		)

		ev_out_dir = out_dir / event_name
		phs_path = _require_one_trial_phs(ev_out_dir, trial=int(trial))
		p_tok = _read_phs_token_by_station(phs_path, phase='P')

		out_json = ev_out_dir / str(pick_json_name)
		obj = {
			'format': 'loki-stalta-pass1-picks-v1',
			'event_id': str(event_name),
			'trial': int(trial),
			'phs_filename': phs_path.name,
			'phase_run': 'P',
			'p_token_by_station': p_tok,
			's_token_by_station': {},
		}
		_write_json(out_json, obj)
		pick_json_by_event[str(event_name)] = out_json

		print(
			f'[STALTA-PASS1-DAS] saved picks json: event={event_name} '
			f'stations={len(p_tok)} path={out_json}'
		)

	return pick_json_by_event
