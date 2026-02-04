from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from obspy import Stream, Trace

from common.config import (
	JmaDtPickErrorBandpassConfig,
	JmaDtPickErrorConfigV1,
	JmaDtPickErrorEvalConfig,
	JmaDtPickErrorInputsConfig,
	JmaDtPickErrorOutputConfig,
	JmaDtPickErrorPickerConfig,
	JmaDtPickErrorPickExtractConfig,
	JmaDtPickErrorPreprocessConfig,
	JmaDtPickErrorRunConfig,
	JmaDtPickErrorStaltaConfig,
)
from pipelines.jma_dt_pick_error_table import (
	_DT_TABLE_COLUMNS,
	run_jma_dt_pick_error_table,
)


@dataclass(frozen=True)
class _Meta:
	lat: float
	lon: float
	event_month: str


class _Inv:
	def __init__(self, station_meta: dict[str, dict[str, object]]):
		self.station_meta = station_meta


def _make_cfg(tmp_path, *, tol_sec: list[float]) -> JmaDtPickErrorConfigV1:
	run_id = 'unit_run'
	out_dir = tmp_path / 'runs' / run_id

	return JmaDtPickErrorConfigV1(
		version=1,
		run=JmaDtPickErrorRunConfig(
			run_id=run_id,
			out_dir=out_dir,
			overwrite=True,
			notes='',
		),
		inputs=JmaDtPickErrorInputsConfig(
			event_root=tmp_path / 'events',
			epicenters_csv=tmp_path / 'epicenters.csv',
			measurements_csv=tmp_path / 'measurements.csv',
			mapping_report_csv=tmp_path / 'mapping_report.csv',
			near0_csv=tmp_path / 'near0.csv',
			monthly_presence_csv=tmp_path / 'monthly_presence.csv',
			mag1_types_allowed=['v', 'V'],
			distance='hypocentral',
			phase_defs={'P': ['P', 'EP', 'IP'], 'S': ['S', 'ES', 'IS']},
			stations_allowlist=None,
			event_id_allowlist=None,
		),
		preprocess=JmaDtPickErrorPreprocessConfig(
			preprocess_preset='jma_snr_pick_table_v1',
			fs_target_hz=100.0,
			detrend='linear',
			bandpass=JmaDtPickErrorBandpassConfig(
				fstop_lo=0.5,
				fpass_lo=1.0,
				fpass_hi=20.0,
				fstop_hi=25.0,
				gpass=1.0,
				gstop=40.0,
			),
		),
		picker=JmaDtPickErrorPickerConfig(
			picker_name='stalta',
			picker_preset='stalta_p_v1',
			phase='P',
			component='U',
			stalta=JmaDtPickErrorStaltaConfig(
				transform='raw', sta_sec=0.2, lta_sec=2.0
			),
		),
		pick_extract=JmaDtPickErrorPickExtractConfig(
			search_pre_sec=1.0,
			search_post_sec=3.0,
			clip_search_window=True,
			choose='max',
			tie_break='min_t',
			thr=0.20,
			min_sep_sec=0.20,
			search_i1_inclusive=True,
		),
		eval=JmaDtPickErrorEvalConfig(tol_sec=tol_sec, keep_missing_rows=True),
		output=JmaDtPickErrorOutputConfig(
			dt_table_csv='dt_table.csv',
			skips_csv='skips.csv',
			save_config_snapshot=False,
		),
		experiments=[],
	)


def _write_min_inputs(tmp_path) -> tuple[str, int]:
	origin_iso = '2026-02-03T00:00:00+09:00'
	event_id = 123

	# event_root / one event dir containing exactly one .evt + matching .txt + .ch
	event_root = tmp_path / 'events'
	ev_dir = event_root / 'E0001'
	ev_dir.mkdir(parents=True, exist_ok=True)
	(evt_path := ev_dir / 'E0001.evt').write_text('dummy', encoding='utf-8')
	(evt_path.with_suffix('.txt')).write_text('dummy', encoding='utf-8')
	(ev_dir / 'E0001.ch').write_text('dummy', encoding='utf-8')

	# epicenters csv must have required columns; only 1 row
	pd.DataFrame(
		[
			{
				'event_id': event_id,
				'origin_time': origin_iso,
				'latitude_deg': 35.0,
				'longitude_deg': 139.0,
				'depth_km': 10.0,
				'mag1': 1.2,
				'mag1_type': 'V',
			}
		]
	).to_csv(tmp_path / 'epicenters.csv', index=False)

	# measurements csv content is not used (we stub pick-table builder), but must exist/readable
	pd.DataFrame([{'dummy': 0}]).to_csv(tmp_path / 'measurements.csv', index=False)

	# these must exist; readers are stubbed
	(tmp_path / 'mapping_report.csv').write_text('dummy\n', encoding='utf-8')
	(tmp_path / 'near0.csv').write_text('dummy\n', encoding='utf-8')
	(tmp_path / 'monthly_presence.csv').write_text('dummy\n', encoding='utf-8')

	return origin_iso, event_id


def test_run_jma_dt_pick_error_table_writes_csv_with_v1_columns(
	monkeypatch, tmp_path
) -> None:
	origin_iso, event_id = _write_min_inputs(tmp_path)
	cfg = _make_cfg(tmp_path, tol_sec=[0.05, 0.10, 0.20])

	import pipelines.jma_dt_pick_error_table as pipe

	# disable snapshot writing in tests
	monkeypatch.setattr(pipe, 'save_yaml_and_effective', lambda **_kw: None)

	# mapping/presence DB loaders (not under test here)
	monkeypatch.setattr(pipe, 'load_mapping_db', lambda *_a, **_kw: object())
	monkeypatch.setattr(pipe, 'load_presence_db', lambda *_a, **_kw: object())

	# event txt readers
	monkeypatch.setattr(pipe, 'read_origin_jst_iso', lambda _p: origin_iso)
	monkeypatch.setattr(
		pipe,
		'read_event_txt_meta',
		lambda _p: _Meta(lat=35.0, lon=139.0, event_month='2026-02'),
	)

	# pick table builder: 2 stations with p_time at +1.0 sec (JST)
	p_time = '2026-02-03T00:00:01.000+09:00'
	pick_df = pd.DataFrame({'p_time': [p_time, p_time]}, index=['STA1', 'STA2'])
	monkeypatch.setattr(
		pipe, 'build_pick_table_for_event', lambda *_a, **_kw: (pick_df, [])
	)

	# inventory for distance calculation: keep same lat/lon as event => epi distance 0
	inv = _Inv(
		station_meta={
			'STA1': {'lat': 35.0, 'lon': 139.0},
			'STA2': {'lat': 35.0, 'lon': 139.0},
		}
	)
	monkeypatch.setattr(pipe, 'build_inventory', lambda _p: inv)

	# waveform loader: return fixed t0 and fs + dummy stream
	t0 = datetime.fromisoformat('2026-02-03T00:00:00')
	st = Stream(
		traces=[
			Trace(data=np.zeros(1000, dtype=np.float32)),
			Trace(data=np.zeros(1000, dtype=np.float32)),
		]
	)

	@dataclass(frozen=True)
	class _LoadRes:
		t0: datetime
		fs_hz: float
		stream: Stream
		stations_used: list[str]
		skips: list[dict[str, object]]

	monkeypatch.setattr(
		pipe,
		'load_u_stream_for_event',
		lambda *_a, **_kw: _LoadRes(
			t0=t0, fs_hz=100.0, stream=st, stations_used=['STA1', 'STA2'], skips=[]
		),
	)

	# prob builder: provide a score array long enough (extraction is stubbed below)
	score = np.zeros(1000, dtype=float)
	monkeypatch.setattr(
		pipe,
		'build_probs_by_station',
		lambda *_a, **_kw: {'STA1': {'P': score}, 'STA2': {'P': score}},
	)

	# extractor: deterministic pick at 110 (=> dt = +0.10 sec)
	def _fake_extract(score_1d, ref_pick_idx, *, fs_hz, **_kw):
		assert int(ref_pick_idx) == 100  # +1.0 sec at 100Hz
		assert float(fs_hz) == 100.0
		return {
			'found_peak': True,
			'est_pick_idx': 110.0,
			'score_at_pick': 0.9,
			'n_peaks': 1,
			'search_i0': 0,
			'search_i1': 200,
			'fail_reason': '',
		}

	monkeypatch.setattr(pipe, 'extract_pick_near_ref', _fake_extract)

	yaml_path = tmp_path / 'pipeline.yaml'
	yaml_path.write_text('dummy', encoding='utf-8')

	dt_df, skip_df = run_jma_dt_pick_error_table(cfg, yaml_path=yaml_path, preset='v1')

	# strict v1 column order
	assert list(dt_df.columns) == _DT_TABLE_COLUMNS
	assert len(dt_df) == 2
	assert set(dt_df['station'].astype(str)) == {'STA1', 'STA2'}
	assert set(dt_df['event_id'].astype(int)) == {event_id}

	# key computed fields
	assert set(dt_df['found_peak'].astype(int)) == {1}
	assert set(dt_df['fail_reason'].astype(str)) == {''}
	assert set(dt_df['good_0p10'].astype(int)) == {1}
	assert set(dt_df['good_0p05'].astype(int)) == {0}
	assert set(dt_df['good_0p20'].astype(int)) == {1}
	vals = dt_df['dt_sec'].astype(float).to_list()
	assert len(vals) == 2
	for v in vals:
		assert v == pytest.approx(0.10)
	# ensure output files exist and header matches v1 order
	out_dir = cfg.run.out_dir
	dt_csv = out_dir / 'dt_table.csv'
	sk_csv = out_dir / 'skips.csv'
	assert dt_csv.is_file()
	assert sk_csv.is_file()

	dt_df2 = pd.read_csv(dt_csv, low_memory=False)
	assert list(dt_df2.columns) == _DT_TABLE_COLUMNS

	assert list(skip_df.columns) == ['event_dir', 'event_id', 'station', 'reason']


def test_run_jma_dt_pick_error_table_rejects_wrong_tol_sec(
	monkeypatch, tmp_path
) -> None:
	_write_min_inputs(tmp_path)
	cfg = _make_cfg(tmp_path, tol_sec=[0.05, 0.10])  # wrong (must be 3 values)

	import pipelines.jma_dt_pick_error_table as pipe

	monkeypatch.setattr(pipe, 'save_yaml_and_effective', lambda **_kw: None)

	yaml_path = tmp_path / 'pipeline.yaml'
	yaml_path.write_text('dummy', encoding='utf-8')

	with pytest.raises(ValueError, match=r'eval\.tol_sec must be'):
		run_jma_dt_pick_error_table(cfg, yaml_path=yaml_path, preset='v1')
