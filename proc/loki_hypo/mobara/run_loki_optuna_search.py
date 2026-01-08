# %%
# file: proc/loki_hypo/run_loki_optuna_search_continuous_bandpass.py
from __future__ import annotations

import datetime as dt
import json
import re
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import optuna
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

	# ===== search space (continuous bandpass) =====
	bp_lo_min, bp_lo_max = 0.5, 3.0
	bp_hi_min, bp_hi_max = 10.0, 30.0

	# STALTA-ish + derivative
	tshortp_lo, tshortp_hi = 0.10, 0.40
	tshorts_lo, tshorts_hi = 0.15, 0.90
	slrat_min, slrat_max = 2, 10

	fs = float(inputs0.base_sampling_rate_hz)
	nyq = fs / 2.0

	# fpass_hi は stopband 設計の都合で nyq*0.98 未満に制限
	bp_hi_max_eff = min(float(bp_hi_max), float(nyq * 0.98 - 1e-6))
	if not (bp_hi_min < bp_hi_max_eff):
		raise ValueError(
			f'invalid bandpass hi range under Nyquist: hi_min={bp_hi_min} hi_max_eff={bp_hi_max_eff} nyq={nyq}'
		)

	# 公平性：イベント集合固定
	event_dirs0 = list_event_dirs_filtered(cfg0)
	allowed_event_ids = {p.name for p in event_dirs0}

	base_out = Path(cfg0.loki_output_path)
	base_out.mkdir(parents=True, exist_ok=True)
	run_root = base_out / 'optuna_bp_stalta_derivative_contbp'
	run_root.mkdir(parents=True, exist_ok=True)

	storage_path = run_root / 'optuna.db'
	storage = f'sqlite:///{storage_path}'

	trials_csv = run_root / 'optuna_trials.csv'

	def objective(trial: optuna.Trial) -> tuple[float, float]:
		# ---- bandpass (continuous) ----
		lo = float(trial.suggest_float('bp_lo', bp_lo_min, bp_lo_max))
		hi_min_eff = max(float(bp_hi_min), float(lo + 0.5))
		if not (hi_min_eff < bp_hi_max_eff):
			raise ValueError(
				f'invalid hi range after lo: lo={lo} hi_min_eff={hi_min_eff} hi_max_eff={bp_hi_max_eff}'
			)
		hi = float(trial.suggest_float('bp_hi', hi_min_eff, bp_hi_max_eff))

		fstop_lo, fstop_hi = _design_stopbands(
			fpass_lo=float(lo), fpass_hi=float(hi), nyquist=nyq
		)

		# ---- tshortp/tshorts (continuous with constraint) ----
		tp = float(trial.suggest_float('tshortp', tshortp_lo, tshortp_hi))
		ts_min = max(float(tshorts_lo), float(tp + 0.05))
		if not (ts_min < tshorts_hi):
			raise ValueError(
				f'invalid tshort range: tp={tp} -> ts_min={ts_min} >= {tshorts_hi}'
			)
		ts = float(trial.suggest_float('tshorts', ts_min, tshorts_hi))

		# ---- slrat (int), derivative (bool) ----
		slrat = int(trial.suggest_int('slrat', slrat_min, slrat_max))
		derivative = bool(trial.suggest_categorical('derivative', [0, 1]))

		tag = (
			f'trial_{trial.number:04d}'
			f'_bp_lo{_num_tag(lo)}_hi{_num_tag(hi)}'
			f'_tp{_num_tag(tp)}_ts{_num_tag(ts)}'
			f'_sl{slrat}_der{int(derivative)}'
		)

		run_out = run_root / tag
		run_data = run_out / 'loki_data'

		cfg = replace(cfg0, loki_output_path=run_out, loki_data_path=run_data)
		inp = replace(
			inputs0,
			# preprocess: bandpass
			pre_enable=True,
			pre_fstop_lo=float(fstop_lo),
			pre_fpass_lo=float(lo),
			pre_fpass_hi=float(hi),
			pre_fstop_hi=float(fstop_hi),
			# keep MAD scaling enabled (fixed)
			pre_mad_scale=True,
			# STALTA-ish
			tshortp_min=float(tp),
			tshortp_max=float(tp),
			tshorts_min=float(ts),
			tshorts_max=float(ts),
			slrat=int(slrat),
			# derivative
			derivative=bool(derivative),
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
					**m,
				},
				indent=2,
				ensure_ascii=False,
			)
			+ '\n'
		)

		_safe_cleanup_loki_data(
			loki_output_path=Path(cfg.loki_output_path),
			loki_data_path=Path(cfg.loki_data_path),
		)

		# multi-objective: (location, time) を両方 minimize
		return float(m['median_e_w3d_km']), float(m['median_abs_dt_sec'])

	def on_trial_end(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
		df = study.trials_dataframe(attrs=('number', 'state', 'values', 'params'))
		df.to_csv(trials_csv, index=False)

		ok = df[df['state'] == 'COMPLETE'].copy()
		if ok.empty:
			return
		ok['score'] = ok['values_0'].astype(float) + 0.1 * ok['values_1'].astype(float)
		ok = ok.sort_values('score')
		head = ok.head(5)[['number', 'values_0', 'values_1', 'score']]
		print('[PROGRESS] top5 by score=loc+0.1*time')
		print(head.to_string(index=False), flush=True)

	# ===== create/resume study =====
	study = optuna.create_study(
		study_name='bp_stalta_derivative_contbp',
		directions=['minimize', 'minimize'],
		storage=storage,
		load_if_exists=True,
	)

	# sampler seed（再現性）
	sampler = optuna.samplers.TPESampler(seed=20251219, multivariate=True, group=True)
	study.sampler = sampler

	n_trials = 120
	start = dt.datetime.now()
	study.optimize(objective, n_trials=int(n_trials), callbacks=[on_trial_end])
	elapsed = (dt.datetime.now() - start).total_seconds()
	print(f'[OK] optuna done: elapsed_sec={elapsed:.1f}')
	print(f'[OK] wrote: {trials_csv}')
	print(f'[OK] storage: {storage_path}')


if __name__ == '__main__':
	main()
