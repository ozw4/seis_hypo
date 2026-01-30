from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg', force=True)

import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import LineCollection

# -----------------------------
# viz.hypo.synth_eval 側のテスト
# -----------------------------


def _patch_save_figure_capture(monkeypatch: pytest.MonkeyPatch):
	"""viz.hypo.synth_eval.save_figure を差し替え、closeせずに fig を捕捉してPNGも出す。"""
	import viz.hypo.synth_eval as m

	cap: dict[str, object] = {}

	def _save_figure(
		fig,
		out_png,
		*,
		dpi=200,
		bbox_inches=None,
		pad_inches=None,
		tight_layout=False,
		close=True,
	):
		out_png = Path(out_png)
		out_png.parent.mkdir(parents=True, exist_ok=True)
		if tight_layout:
			fig.tight_layout()
		fig.savefig(out_png, dpi=int(dpi))
		cap['fig'] = fig
		cap['out_png'] = out_png
		return out_png

	monkeypatch.setattr(m, 'save_figure', _save_figure)
	return cap


def test_save_true_pred_xyz_3view_still_saves_png(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	import viz.hypo.synth_eval as m

	_patch_save_figure_capture(monkeypatch)

	true_xyz_m = np.array(
		[
			[0.0, 0.0, 1000.0],
			[1000.0, 1000.0, 2000.0],
		],
		dtype=float,
	)
	pred_xyz_m = np.array(
		[
			[100.0, -50.0, 1100.0],
			[900.0, 1100.0, 1900.0],
		],
		dtype=float,
	)

	out_png = tmp_path / 'xyz.png'

	m.save_true_pred_xyz_3view(true_xyz_m, pred_xyz_m, out_png, title='t')

	assert out_png.is_file()
	assert out_png.stat().st_size > 0


def test_ellipse_polyline_min_points_and_shape() -> None:
	import viz.hypo.synth_eval as m

	with pytest.raises(ValueError, match='n_points must be >= 20'):
		m._ellipse_polyline(
			cx=0.0, cy=0.0, a_km=1.0, b_km=0.5, theta_rad=0.0, n_points=19
		)

	xy = m._ellipse_polyline(
		cx=0.0, cy=0.0, a_km=1.0, b_km=0.5, theta_rad=0.0, n_points=50
	)
	assert xy.shape == (50, 2)
	assert np.isfinite(xy).all()


def test_save_true_pred_xyz_3view_with_uncertainty_validates_inputs(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	import viz.hypo.synth_eval as m

	_patch_save_figure_capture(monkeypatch)

	true_xyz_m = np.array([[0.0, 0.0, 1000.0]], dtype=float)
	pred_xyz_m = np.array([[0.0, 0.0, 1000.0]], dtype=float)

	# len(df_eval) mismatch
	df_bad_len = pd.DataFrame(
		{
			'ell_s1_km': [0.1, 0.2],
			'ell_az1_deg': [90, 90],
			'ell_dip1_deg': [0, 0],
			'ell_s2_km': [0.1, 0.2],
			'ell_az2_deg': [0, 0],
			'ell_dip2_deg': [0, 0],
			'ell_s3_km': [0.1, 0.2],
			'ell_az3_deg': [0, 0],
			'ell_dip3_deg': [90, 90],
		}
	)
	with pytest.raises(ValueError, match='len\\(df_eval\\) must match'):
		m.save_true_pred_xyz_3view_with_uncertainty(
			true_xyz_m,
			pred_xyz_m,
			df_bad_len,
			tmp_path / 'u.png',
		)

	# missing uncertainty columns
	df_missing = pd.DataFrame({'ell_s1_km': [0.1]})
	with pytest.raises(KeyError, match='missing uncertainty columns'):
		m.save_true_pred_xyz_3view_with_uncertainty(
			true_xyz_m,
			pred_xyz_m,
			df_missing,
			tmp_path / 'u2.png',
		)

	# invalid sigma_scale_sec
	df_ok = pd.DataFrame(
		{
			'ell_s1_km': [0.1],
			'ell_az1_deg': [90],
			'ell_dip1_deg': [0],
			'ell_s2_km': [0.1],
			'ell_az2_deg': [0],
			'ell_dip2_deg': [0],
			'ell_s3_km': [0.1],
			'ell_az3_deg': [0],
			'ell_dip3_deg': [90],
		}
	)
	with pytest.raises(ValueError, match='invalid sigma_scale_sec'):
		m.save_true_pred_xyz_3view_with_uncertainty(
			true_xyz_m,
			pred_xyz_m,
			df_ok,
			tmp_path / 'u3.png',
			sigma_scale_sec=0.0,
		)


def test_save_true_pred_xyz_3view_with_uncertainty_adds_ellipses_and_legend(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	import viz.hypo.synth_eval as m

	cap = _patch_save_figure_capture(monkeypatch)

	true_xyz_m = np.array(
		[
			[0.0, 0.0, 1000.0],
			[1000.0, 1000.0, 2000.0],
		],
		dtype=float,
	)
	pred_xyz_m = np.array(
		[
			[100.0, -50.0, 1100.0],
			[900.0, 1100.0, 1900.0],
		],
		dtype=float,
	)

	# Event0: ok (max SERR=0.2km)
	# Event1: poor (max SERR=100km -> clip to 10km, poor_thresh default=5km)
	df_eval = pd.DataFrame(
		{
			'ell_s1_km': [0.2, 100.0],
			'ell_az1_deg': [90, 90],  # E
			'ell_dip1_deg': [0, 0],
			'ell_s2_km': [0.1, 1.0],
			'ell_az2_deg': [0, 0],  # N
			'ell_dip2_deg': [0, 0],
			'ell_s3_km': [0.05, 0.5],
			'ell_az3_deg': [0, 0],  # Down
			'ell_dip3_deg': [90, 90],
		}
	)

	out_png = tmp_path / 'xyz_unc.png'

	m.save_true_pred_xyz_3view_with_uncertainty(
		true_xyz_m, pred_xyz_m, df_eval, out_png
	)

	assert out_png.is_file()
	assert out_png.stat().st_size > 0

	fig = cap['fig']
	assert fig is not None

	axes = list(fig.axes)

	# 3つの断面Axesには LineCollection が (ok/poor) の2つずつ乗る
	line_axes = []
	for ax in axes:
		n_lc = sum(isinstance(c, LineCollection) for c in ax.collections)
		if n_lc > 0:
			line_axes.append((ax, n_lc))

	assert len(line_axes) == 3
	for _, n_lc in line_axes:
		assert n_lc == 2

	# legend は空Axesにのみ載る想定（labelsに楕円2種類が含まれる）
	leg_ax = next((ax for ax in axes if ax.get_legend() is not None), None)
	assert leg_ax is not None

	labels = [t.get_text() for t in leg_ax.get_legend().get_texts()]
	assert '1σ ellipse' in labels
	assert '1σ ellipse (poor)' in labels


# -----------------------------
# qc.hypo.synth_eval 側のテスト
# -----------------------------


def _write_qc_fixture(
	tmp_path: Path,
	*,
	eval_df: pd.DataFrame,
	outputs_dir: str = 'run1',
	receiver_geometry: str = 'geom.npy',
) -> Path:
	# dataset
	dataset_dir = (tmp_path / 'dataset').resolve()
	(dataset_dir / 'geometry').mkdir(parents=True, exist_ok=True)

	# index.csv (内容は件数だけ使う)
	(dataset_dir / 'index.csv').write_text('event_id\nE1\nE2\n', encoding='utf-8')

	# geometry npy
	recv_xyz_m = np.array(
		[
			[0.0, 0.0, 0.0],
			[100.0, 0.0, 0.0],
			[0.0, 100.0, 0.0],
		],
		dtype=float,
	)
	np.save(dataset_dir / 'geometry' / receiver_geometry, recv_xyz_m)

	# proc layout: config_path.parent.parent / 'runs' が runs_root
	proc_dir = tmp_path / 'proc'
	qc_dir = proc_dir / 'qc'
	qc_dir.mkdir(parents=True, exist_ok=True)
	runs_root = proc_dir / 'runs'
	run_dir = runs_root / outputs_dir
	run_dir.mkdir(parents=True, exist_ok=True)

	# station_synth.csv (receiver_index必須)
	(run_dir / 'station_synth.csv').write_text(
		'station_code,receiver_index\nSTA001,0\nDSTA02,2\n', encoding='utf-8'
	)

	# eval_metrics.csv
	eval_df.to_csv(run_dir / 'eval_metrics.csv', index=False)

	# config yaml
	cfg = (
		f'dataset_dir: {dataset_dir}\n'
		f'outputs_dir: {outputs_dir}\n'
		f'receiver_geometry: {receiver_geometry}\n'
	)
	config_path = qc_dir / 'qc.yml'
	config_path.write_text(cfg, encoding='utf-8')

	return config_path


def test_run_qc_skips_uncertainty_when_missing_columns(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	import qc.hypo.synth_eval as q

	# ell_* 列は入れない（skip期待）
	df = pd.DataFrame(
		{
			'horiz_m': [1.0, 2.0],
			'dz_m': [1.0, 2.0],
			'err3d_m': [1.5, 2.8],
			'x_m_true': [0.0, 1000.0],
			'y_m_true': [0.0, 1000.0],
			'z_m_true': [1000.0, 2000.0],
			'x_m_hyp': [0.0, 1000.0],
			'y_m_hyp': [0.0, 1000.0],
			'z_m_hyp': [1000.0, 2000.0],
		}
	)
	config_path = _write_qc_fixture(tmp_path, eval_df=df)

	# plot関数は軽量化（PNGはtouchするだけ）
	def _touch_png(*args, **kwargs):
		out = Path(args[1] if len(args) >= 2 else kwargs['out_png'])
		out.parent.mkdir(parents=True, exist_ok=True)
		out.write_bytes(b'x')

	monkeypatch.setattr(q, 'save_hist', lambda *a, **k: _touch_png(a[0], a[1]))
	monkeypatch.setattr(q, 'save_dxdy_scatter', lambda *a, **k: _touch_png(a[0], a[2]))
	monkeypatch.setattr(
		q, 'save_true_pred_xyz_3view', lambda *a, **k: _touch_png(a[0], a[2])
	)

	# uncertainty 側は呼ばれたら失敗
	monkeypatch.setattr(
		q,
		'save_true_pred_xyz_3view_with_uncertainty',
		lambda *a, **k: (_ for _ in ()).throw(AssertionError('must not be called')),
	)

	q.run_qc(config_path)

	run_dir = config_path.resolve().parent.parent / 'runs' / 'run1'
	assert (run_dir / 'xy_true_vs_hyp.png').is_file()


def test_run_qc_passes_masked_df_to_uncertainty(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	import qc.hypo.synth_eval as q

	# 先頭行を NaN にして mask で落とし、残るのは2行目のみ
	df = pd.DataFrame(
		{
			'horiz_m': [1.0, 2.0],
			'dz_m': [1.0, 2.0],
			'err3d_m': [1.5, 2.8],
			'x_m_true': [0.0, 1000.0],
			'y_m_true': [0.0, 1000.0],
			'z_m_true': [1000.0, 2000.0],
			'x_m_hyp': [np.nan, 1000.0],
			'y_m_hyp': [0.0, 1000.0],
			'z_m_hyp': [1000.0, 2000.0],
			# ell_* を揃える（呼び出しされる）
			'ell_s1_km': [0.2, 0.2],
			'ell_az1_deg': [90, 90],
			'ell_dip1_deg': [0, 0],
			'ell_s2_km': [0.1, 0.1],
			'ell_az2_deg': [0, 0],
			'ell_dip2_deg': [0, 0],
			'ell_s3_km': [0.05, 0.05],
			'ell_az3_deg': [0, 0],
			'ell_dip3_deg': [90, 90],
		}
	)
	config_path = _write_qc_fixture(tmp_path, eval_df=df)

	def _touch_png(path: Path) -> None:
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_bytes(b'x')

	monkeypatch.setattr(q, 'save_hist', lambda *a, **k: _touch_png(Path(a[1])))
	monkeypatch.setattr(q, 'save_dxdy_scatter', lambda *a, **k: _touch_png(Path(a[2])))
	monkeypatch.setattr(
		q, 'save_true_pred_xyz_3view', lambda *a, **k: _touch_png(Path(a[2]))
	)

	called: dict[str, object] = {}

	def _unc(true_xyz_m, pred_xyz_m, df_plot, out_png, **kwargs):
		# mask適用後の1イベントだけになる（len一致必須）
		true_xyz_m = np.asarray(true_xyz_m, float).reshape(-1, 3)
		pred_xyz_m = np.asarray(pred_xyz_m, float).reshape(-1, 3)
		assert true_xyz_m.shape[0] == 1
		assert pred_xyz_m.shape[0] == 1
		assert len(df_plot) == 1

		# reset_index(drop=True) の確認：残る行が元index=1なので、resetされて0になるはず
		assert list(df_plot.index) == [0]

		_touch_png(Path(out_png))
		called['ok'] = True

	monkeypatch.setattr(q, 'save_true_pred_xyz_3view_with_uncertainty', _unc)

	q.run_qc(config_path)
	assert called.get('ok', False) is True

	run_dir = config_path.resolve().parent.parent / 'runs' / 'run1'
	assert (run_dir / 'xy_true_vs_hyp.png').is_file()
	assert (run_dir / 'xy_true_vs_hyp_uncertainty.png').is_file()
