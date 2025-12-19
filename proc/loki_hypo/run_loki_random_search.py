# %%
# file: proc/loki_hypo/run_loki_random_search_bp_stalta_derivative.py
from __future__ import annotations

import datetime as dt
import itertools
import json
import re
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import LokiWaveformStackingInputs, LokiWaveformStackingPipelineConfig
from common.load_config import load_config
from common.run_snapshot import save_many_yaml_and_effective
from pipelines.loki_waveform_stacking_pipelines import (
	list_event_dirs_filtered,
	pipeline_loki_waveform_stacking,
)
from qc.loki_compare_qc import run_loki_vs_jma_qc
from viz.plot_config import PlotConfig


def _num_tag(x: float) -> str:
	s = f'{x:.5g}'
	s = s.replace('.', 'p')
	s = re.sub(r'[^0-9A-Za-z_]+', '_', s)
	return s


def _metrics_from_compare_csv(csv_path: Path) -> dict[str, float]:
	df = pd.read_csv(csv_path)
	required = {'e_w3d_km', 'dt_origin_sec'}
	missing = required - set(df.columns)
	if missing:
		raise ValueError(
			f'compare csv missing columns: {sorted(missing)} in {csv_path}'
		)

	e = df['e_w3d_km'].astype(float).to_numpy()
	dt_sec = df['dt_origin_sec'].astype(float).to_numpy()
	abs_dt = np.abs(dt_sec)

	return {
		'n_events': float(len(df)),
		'median_e_w3d_km': float(np.median(e)),
		'p90_e_w3d_km': float(np.quantile(e, 0.9)),
		'median_abs_dt_sec': float(np.median(abs_dt)),
		'p90_abs_dt_sec': float(np.quantile(abs_dt, 0.9)),
	}


def _safe_cleanup_loki_data(*, loki_output_path: Path, loki_data_path: Path) -> None:
	if not loki_data_path.is_dir():
		return
	out_s = str(loki_output_path.resolve())
	data_s = str(loki_data_path.resolve())
	if not (data_s == out_s or data_s.startswith(out_s + '/')):
		raise RuntimeError(
			f'refusing to delete outside output: data={data_s} out={out_s}'
		)
	shutil.rmtree(loki_data_path)


def _design_stopbands(
	*, fpass_lo: float, fpass_hi: float, nyquist: float
) -> tuple[float, float]:
	fstop_lo = max(0.05, fpass_lo * 0.8)
	fstop_hi = min(nyquist * 0.98, fpass_hi * 1.2)
	if not (0.0 < fstop_lo < fpass_lo < fpass_hi < fstop_hi <= nyquist):
		raise ValueError(
			'Invalid band edges: '
			f'fstop_lo={fstop_lo} fpass_lo={fpass_lo} fpass_hi={fpass_hi} fstop_hi={fstop_hi} nyq={nyquist}'
		)
	return float(fstop_lo), float(fstop_hi)


def main() -> None:
	# ===== presets =====
	pipeline_yaml = Path('/workspace/data/config/loki_waveform_pipeline.yaml')
	pipeline_preset = 'mobara'
	cfg0 = load_config(
		LokiWaveformStackingPipelineConfig, pipeline_yaml, pipeline_preset
	)
	inputs0 = load_config(
		LokiWaveformStackingInputs, cfg0.inputs_yaml, cfg0.inputs_preset
	)

	plot_config_yaml = Path('/workspace/data/config/plot_config.yaml')
	plot_setting = 'mobara_default'
	plot_cfg = load_config(PlotConfig, plot_config_yaml, plot_setting)

	# ===== search space: bandpass =====
	low_list = [0.5, 1.0, 2.0, 2.5, 3.0]
	high_list = [10, 15, 20, 25, 30]
	bp_pairs = [
		(lo, hi) for lo, hi in itertools.product(low_list, high_list) if lo < hi
	]

	fs = float(inputs0.base_sampling_rate_hz)
	nyq = fs / 2.0
	if max(high_list) >= nyq:
		raise ValueError(
			f'high must be < Nyquist={nyq}, got max(high)={max(high_list)}'
		)

	# ===== search budget =====
	n_runs = 300
	seed = 20251219
	rng = np.random.default_rng(seed)
	save_every = 5
	# ===== search space: STALTA-ish =====
	# あなたのYAMLでは min=max で固定値運用なので、サンプル値を min=max に入れる
	tshortp_lo, tshortp_hi = 0.10, 0.40
	tshorts_lo, tshorts_hi = 0.15, 0.90
	slrat_min, slrat_max = 2, 10  # int
	derivative_choices = [False, True]

	# 公平性：イベント集合を固定
	event_dirs0 = list_event_dirs_filtered(cfg0)
	allowed_event_ids = {p.name for p in event_dirs0}

	base_out = Path(cfg0.loki_output_path)
	base_out.mkdir(parents=True, exist_ok=True)
	run_root = base_out / 'random_search_bp_stalta_derivative'
	run_root.mkdir(parents=True, exist_ok=True)
	partial_csv = run_root / 'random_search_summary_partial.csv'
	start = dt.datetime.now()
	rows: list[dict[str, object]] = []
	best_key = (float('inf'), float('inf'))
	best_tag = ''
	for i in range(int(n_runs)):
		# --- sample bandpass ---
		lo, hi = bp_pairs[int(rng.integers(0, len(bp_pairs)))]
		fstop_lo, fstop_hi = _design_stopbands(
			fpass_lo=float(lo), fpass_hi=float(hi), nyquist=nyq
		)

		# --- sample tshortp/tshorts with constraint tshorts > tshortp ---
		tp = float(rng.uniform(tshortp_lo, tshortp_hi))
		ts_min = max(tshorts_lo, tp + 0.05)
		if ts_min >= tshorts_hi:
			raise ValueError(
				f'invalid tshort range: tp={tp} -> ts_min={ts_min} >= {tshorts_hi}'
			)
		ts = float(rng.uniform(ts_min, tshorts_hi))

		# --- sample slrat (int) ---
		slrat = int(rng.integers(slrat_min, slrat_max + 1))

		# --- sample derivative (bool) ---
		derivative = bool(
			derivative_choices[int(rng.integers(0, len(derivative_choices)))]
		)

		tag = (
			f'rs_{i:04d}'
			f'_bp_lo{_num_tag(float(lo))}_hi{_num_tag(float(hi))}'
			f'_tp{_num_tag(tp)}_ts{_num_tag(ts)}'
			f'_sl{slrat}'
			f'_der{int(derivative)}'
		)
		run_out = run_root / tag
		run_data = run_out / 'loki_data'

		cfg = replace(cfg0, loki_output_path=run_out, loki_data_path=run_data)

		# MADスケーリングは探索から外して固定（あなたの方針通り）
		inp = replace(
			inputs0,
			# ---- preprocess (bandpass) ----
			pre_enable=True,
			pre_fstop_lo=float(fstop_lo),
			pre_fpass_lo=float(lo),
			pre_fpass_hi=float(hi),
			pre_fstop_hi=float(fstop_hi),
			# ---- STALTA-ish knobs ----
			tshortp_min=float(tp),
			tshortp_max=float(tp),
			tshorts_min=float(ts),
			tshorts_max=float(ts),
			slrat=int(slrat),
			# ---- derivative ----
			derivative=bool(derivative),
			# ---- keep MAD scaling enabled (fixed) ----
			pre_mad_scale=True,
		)

		save_many_yaml_and_effective(
			out_dir=cfg.loki_output_path,
			items=[
				('pipeline', pipeline_yaml, pipeline_preset, cfg),
				('inputs', cfg.inputs_yaml, cfg.inputs_preset, inp),
			],
		)

		# ---- LOKI run ----
		pipeline_loki_waveform_stacking(cfg, inp)

		# ---- QC ----
		header_path = Path(cfg.loki_db_path) / Path(cfg.loki_hdr_filename)
		if not header_path.is_file():
			raise FileNotFoundError(f'header not found: {header_path}')

		compare_csv = Path(cfg.loki_output_path) / 'compare_jma_vs_loki.csv'
		run_loki_vs_jma_qc(
			base_input_dir=Path(cfg.base_input_dir),
			loki_output_dir=Path(cfg.loki_output_path),
			header_path=header_path,
			event_glob=cfg.event_glob,
			plot_cfg=plot_cfg,
			use_build_compare_df=True,
			compare_csv_out=compare_csv,
			allowed_event_ids=allowed_event_ids,
			out_png=Path(cfg.loki_output_path) / 'loki_vs_jma.png',
		)

		m = _metrics_from_compare_csv(compare_csv)
		(Path(cfg.loki_output_path) / 'metrics.json').write_text(
			json.dumps(
				{
					**m,
					'params': {
						'bp_lo': float(lo),
						'bp_hi': float(hi),
						'pre_fstop_lo': float(fstop_lo),
						'pre_fstop_hi': float(fstop_hi),
						'tshortp': float(tp),
						'tshorts': float(ts),
						'slrat': int(slrat),
						'derivative': bool(derivative),
					},
				},
				indent=2,
				ensure_ascii=False,
			)
			+ '\n'
		)

		rows.append(
			{
				'tag': tag,
				'bp_lo': float(lo),
				'bp_hi': float(hi),
				'pre_fstop_lo': float(fstop_lo),
				'pre_fstop_hi': float(fstop_hi),
				'tshortp': float(tp),
				'tshorts': float(ts),
				'slrat': int(slrat),
				'derivative': int(derivative),
				**m,
				'out_dir': str(cfg.loki_output_path),
			}
		)
		cur_key = (float(m['median_e_w3d_km']), float(m['median_abs_dt_sec']))
		if cur_key < best_key:
			best_key = cur_key
			best_tag = tag
			print(
				f'[BEST] i={i + 1}/{n_runs} tag={best_tag} '
				f'median_e_w3d_km={best_key[0]:.4g} median_abs_dt_sec={best_key[1]:.4g}',
				flush=True,
			)
		else:
			print(
				f'[RUN] i={i + 1}/{n_runs} tag={tag} '
				f'median_e_w3d_km={cur_key[0]:.4g} median_abs_dt_sec={cur_key[1]:.4g}',
				flush=True,
			)

		if ((i + 1) % int(save_every) == 0) or (i + 1 == int(n_runs)):
			df_partial = pd.DataFrame(rows).sort_values(
				['median_e_w3d_km', 'median_abs_dt_sec'], ascending=[True, True]
			)
			df_partial.to_csv(partial_csv, index=False)
			print(f'[PROGRESS] wrote: {partial_csv}', flush=True)

		_safe_cleanup_loki_data(
			loki_output_path=Path(cfg.loki_output_path),
			loki_data_path=Path(cfg.loki_data_path),
		)

	df = pd.DataFrame(rows)
	df = df.sort_values(
		['median_e_w3d_km', 'median_abs_dt_sec'], ascending=[True, True]
	)

	summary_csv = run_root / 'random_search_summary.csv'
	df.to_csv(summary_csv, index=False)

	elapsed = (dt.datetime.now() - start).total_seconds()
	print(f'[OK] random search done: n_runs={len(df)} elapsed_sec={elapsed:.1f}')
	print(f'[OK] summary: {summary_csv}')
	print('[OK] best (top 10):')
	print(df.head(10).to_string(index=False))


if __name__ == '__main__':
	main()
