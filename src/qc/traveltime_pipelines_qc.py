# %%
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 既存の import パスは適当に直してOK
from common.config import QcConfig
from common.geo import latlon_to_local_xy_km


# ----------------------------
# Path resolution (unified)
# ----------------------------
def resolve_qc_and_tt_paths(
	cfg: QcConfig,
	preset: str,
) -> tuple[QcConfig, Path, Path, Path]:
	"""QC系の派生パスをまとめて解決する。

	解決ルール:
	- QCベース:
		<run_dirの親>/qc/<preset名>
	- fig_dir:
		<QCベース>
	- control(P/S):
		<run_dir>/<model_label>_P.in
		<run_dir>/<model_label>_S.in
	- Grid2Time出力QCの out_dir:
		<QCベース>/traveltime_tables
	"""
	run_dir = Path(cfg.nll_run_dir)
	qc_base = run_dir.parent / 'qc' / preset

	new_cfg = replace(cfg, fig_dir=qc_base)

	model = cfg.model_label
	control_p = run_dir / f'{model}_P.in'
	control_s = run_dir / f'{model}_S.in'

	tt_out_dir = qc_base / 'traveltime_tables'

	return new_cfg, control_p, control_s, tt_out_dir


def parse_nll_control(control_path: str | Path) -> dict[str, Any]:
	"""NonLinLoc の control から走時QCに必要な情報を抽出する。

	抽出:
	- phase: GTFILES の phase
	- gt_root: GTFILES の gtroot
	- sources: GTSRCE の stationコード一覧
	- sources_latlon:
		GTSRCE <STA> LATLON <lat> <lon> <elev_km> ...
		の <lat, lon, elev_km> を station ごとに保持
	- vggrid:
		VGGRID nx ny nz x0 y0 z0 dx dy dz <quantity>
		の主要数値
	- trans_lat0/trans_lon0:
		TRANS SIMPLE <lat0> <lon0> <zref>
	"""
	p = Path(control_path)
	lines = p.read_text().splitlines()

	phase: str | None = None
	gt_root: Path | None = None
	sources: list[str] = []
	sources_latlon: dict[str, tuple[float, float, float]] = {}
	vggrid: dict[str, float] | None = None
	trans_lat0: float | None = None
	trans_lon0: float | None = None

	for ln in lines:
		ln = ln.strip()
		if not ln or ln.startswith('#'):
			continue

		parts = ln.split()
		key = parts[0].upper()

		if key == 'TRANS' and len(parts) >= 4 and parts[1].upper() == 'SIMPLE':
			trans_lat0 = float(parts[2])
			trans_lon0 = float(parts[3])
			continue

		if key == 'GTFILES' and len(parts) >= 4:
			# GTFILES <vgroot> <gtroot> <phase>
			gt_root = Path(parts[2])
			phase = parts[3].upper()
			continue

		if key == 'GTSRCE' and len(parts) >= 2:
			# GTSRCE <STA> LATLON <lat> <lon> <elev_km> ...
			sta = parts[1]
			sources.append(sta)

			if len(parts) >= 6 and parts[2].upper() == 'LATLON':
				lat = float(parts[3])
				lon = float(parts[4])
				elev_km = float(parts[5])
				sources_latlon[sta] = (lat, lon, elev_km)
			continue

		if key == 'VGGRID' and len(parts) >= 10:
			# VGGRID nx ny nz x0 y0 z0 dx dy dz <quantity>
			vggrid = {
				'nx': float(parts[1]),
				'ny': float(parts[2]),
				'nz': float(parts[3]),
				'x0': float(parts[4]),
				'y0': float(parts[5]),
				'z0': float(parts[6]),
				'dx': float(parts[7]),
				'dy': float(parts[8]),
				'dz': float(parts[9]),
			}
			continue

	if gt_root is None or phase is None:
		raise ValueError(f'GTFILES not found or invalid in {p}')
	if trans_lat0 is None or trans_lon0 is None:
		raise ValueError(f'TRANS SIMPLE not found or invalid in {p}')
	if vggrid is None:
		raise ValueError(f'VGGRID not found or invalid in {p}')

	return {
		'control_path': p,
		'phase': phase,
		'gt_root': gt_root,
		'sources': sources,
		'sources_latlon': sources_latlon,
		'vggrid': vggrid,
		'trans_lat0': trans_lat0,
		'trans_lon0': trans_lon0,
	}


def _annotate_source_location_xy(
	ax: Any,
	info: dict[str, Any],
	*,
	source: str,
) -> tuple[float, float]:
	"""TT マップ上に元のステーション位置を重ね描きする。

	Parameters
	----------
	ax:
		matplotlib Axes
	info:
		parse_nll_control() の戻り値
	source:
		GTSRCE に書き出された station コード

	Returns
	-------
	(x_km, y_km)
		TRANS SIMPLE 基準のローカルXY[km]

	"""
	ll = info['sources_latlon'].get(source)
	if ll is None:
		raise ValueError(f'source LATLON not found in control: {source}')

	lat, lon, _elev_km = ll
	lat0 = float(info['trans_lat0'])
	lon0 = float(info['trans_lon0'])

	x_km_arr, y_km_arr = latlon_to_local_xy_km(
		np.array([lat], dtype=float),
		np.array([lon], dtype=float),
		lat0_deg=lat0,
		lon0_deg=lon0,
	)

	x_km = float(x_km_arr[0])
	y_km = float(y_km_arr[0])

	ax.scatter([x_km], [y_km], marker='*', s=160)
	ax.text(
		x_km,
		y_km,
		source,
		ha='left',
		va='bottom',
		fontsize=9,
	)

	return x_km, y_km


# ----------------------------
# File matching helpers
# ----------------------------
def _candidate_time_files(
	time_dir: Path, root_name: str, phase: str, source: str
) -> dict[str, Path | None]:
	"""命名が完全固定なら strict 版に置換して削ってOK。
	今は移行期の保険として最小限の揺れ吸収をする。
	"""
	phase_u = phase.upper()

	exact_hdr = time_dir / f'{root_name}.{phase_u}.time.{source}.hdr'
	exact_buf = time_dir / f'{root_name}.{phase_u}.time.{source}.buf'

	hdr = exact_hdr if exact_hdr.is_file() else None
	buf = exact_buf if exact_buf.is_file() else None

	if hdr is None:
		cands = list(time_dir.glob(f'{root_name}.{phase_u}.time*{source}*.hdr'))
		if cands:
			hdr = sorted(cands)[0]

	if buf is None:
		cands = list(time_dir.glob(f'{root_name}.{phase_u}.time*{source}*.buf'))
		if cands:
			buf = sorted(cands)[0]

	if hdr is None:
		cands = list(time_dir.glob(f'{root_name}*{phase_u}*{source}*.hdr'))
		if cands:
			hdr = sorted(cands)[0]

	if buf is None:
		cands = list(time_dir.glob(f'{root_name}*{phase_u}*{source}*.buf'))
		if cands:
			buf = sorted(cands)[0]

	return {'hdr': hdr, 'buf': buf}


def _file_size_or_none(p: Path | None) -> int | None:
	if p is None:
		return None
	return p.stat().st_size


# ----------------------------
# CSV QC (existing style)
# ----------------------------
def qc_grid2time_outputs_from_control(
	control_path: str | Path,
	*,
	out_dir: Path,
) -> pd.DataFrame:
	"""1つの control(P or S) に対して
	Grid2Time 出力が局数分揃っているか QC する。
	"""
	info = parse_nll_control(control_path)
	phase = info['phase']
	gt_root: Path = info['gt_root']
	sources: list[str] = info['sources']

	time_dir = gt_root.parent
	root_name = gt_root.name

	if not time_dir.is_dir():
		raise FileNotFoundError(f'time_dir not found: {time_dir}')

	rows: list[dict[str, Any]] = []
	for src in sources:
		found = _candidate_time_files(time_dir, root_name, phase, src)
		hdr = found['hdr']
		buf = found['buf']

		rows.append(
			{
				'phase': phase,
				'source': src,
				'time_dir': str(time_dir),
				'root_name': root_name,
				'hdr_path': str(hdr) if hdr else None,
				'buf_path': str(buf) if buf else None,
				'hdr_size': _file_size_or_none(hdr),
				'buf_size': _file_size_or_none(buf),
				'hdr_exists': bool(hdr and hdr.is_file()),
				'buf_exists': bool(buf and buf.is_file()),
			}
		)

	df = pd.DataFrame(rows).sort_values(['phase', 'source']).reset_index(drop=True)

	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	df.to_csv(out_dir / f'tt_files_{phase}.csv', index=False)

	return df


def qc_grid2time_outputs_ps(
	control_p_path: str | Path,
	control_s_path: str | Path,
	*,
	out_dir: str | Path,
) -> dict[str, Path]:
	"""P/S 両方の Grid2Time 出力 QC（ファイル有無の棚卸）。"""
	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	df_p = qc_grid2time_outputs_from_control(control_p_path, out_dir=out_dir)
	df_s = qc_grid2time_outputs_from_control(control_s_path, out_dir=out_dir)

	def _summ(df: pd.DataFrame) -> dict[str, int]:
		n = len(df)
		n_hdr = int(df['hdr_exists'].sum())
		n_buf = int(df['buf_exists'].sum())
		n_ok = int((df['hdr_exists'] & df['buf_exists']).sum())
		return {'total': n, 'hdr': n_hdr, 'buf': n_buf, 'ok': n_ok}

	sp = _summ(df_p)
	ss = _summ(df_s)

	summary_txt = out_dir / 'tt_files_summary.txt'
	summary_txt.write_text(
		'\n'.join(
			[
				'Grid2Time output QC summary',
				'',
				f'P: total={sp["total"]} hdr={sp["hdr"]} buf={sp["buf"]} ok(hdr&buf)={sp["ok"]}',
				f'S: total={ss["total"]} hdr={ss["hdr"]} buf={ss["buf"]} ok(hdr&buf)={ss["ok"]}',
				'',
				f'P csv: {out_dir / "tt_files_P.csv"}',
				f'S csv: {out_dir / "tt_files_S.csv"}',
			]
		)
		+ '\n'
	)

	return {
		'p_csv': out_dir / 'tt_files_P.csv',
		's_csv': out_dir / 'tt_files_S.csv',
		'summary': summary_txt,
	}


# ----------------------------
# Travel-time grid read
# ----------------------------
@dataclass(frozen=True)
class TimeGridMeta:
	nx: int
	ny: int
	nz: int
	x0_km: float
	y0_km: float
	z0_km: float
	dx_km: float
	dy_km: float
	dz_km: float

	@classmethod
	def from_vggrid(cls, vg: dict[str, float]) -> TimeGridMeta:
		return cls(
			nx=int(vg['nx']),
			ny=int(vg['ny']),
			nz=int(vg['nz']),
			x0_km=float(vg['x0']),
			y0_km=float(vg['y0']),
			z0_km=float(vg['z0']),
			dx_km=float(vg['dx']),
			dy_km=float(vg['dy']),
			dz_km=float(vg['dz']),
		)

	def x_axis_km(self) -> np.ndarray:
		return self.x0_km + self.dx_km * np.arange(self.nx, dtype=float)

	def y_axis_km(self) -> np.ndarray:
		return self.y0_km + self.dy_km * np.arange(self.ny, dtype=float)

	def z_axis_km(self) -> np.ndarray:
		return self.z0_km + self.dz_km * np.arange(self.nz, dtype=float)


def read_time_buf(
	buf_path: str | Path,
	meta: TimeGridMeta,
	*,
	dtype: np.dtype = np.dtype('<f4'),
) -> np.ndarray:
	"""Grid2Time の .buf を 3D 配列にする。

	前提:
	- 1セル=4byte float を想定（一般的な NonLinLoc 出力）
	- 並び順は x が最内側で増える前提で reshape

	もし環境で並びが違うことが判明したら、
	reshape と axis の扱いだけここで直せば全図が修正される。
	"""
	p = Path(buf_path)
	if not p.is_file():
		raise FileNotFoundError(f'buf not found: {p}')

	arr = np.fromfile(p, dtype=dtype)
	expect = meta.nx * meta.ny * meta.nz
	if arr.size != expect:
		raise ValueError(f'buf size mismatch: got {arr.size}, expected {expect} ({p})')

	grid = arr.reshape((meta.nz, meta.ny, meta.nx), order='F')
	return grid


def load_time_grid_for_station(
	control_path: str | Path,
	station: str,
) -> tuple[np.ndarray, TimeGridMeta, dict[str, Any], Path]:
	"""Control と station 名から .buf を特定して読み込み。

	Returns:
		(grid_3d, meta, control_info, buf_path)

	"""
	info = parse_nll_control(control_path)
	meta = TimeGridMeta.from_vggrid(info['vggrid'])

	gt_root: Path = info['gt_root']
	phase = info['phase']

	time_dir = gt_root.parent
	root_name = gt_root.name

	found = _candidate_time_files(time_dir, root_name, phase, station)
	buf = found['buf']
	if buf is None:
		raise FileNotFoundError(
			f'travel-time buf not found for station={station} in {time_dir}'
		)

	grid = read_time_buf(buf, meta)
	return grid, meta, info, buf


# ----------------------------
# Visualization
# ----------------------------
def plot_tt_horizontal_slice(
	grid_3d: np.ndarray,
	meta: TimeGridMeta,
	control_path: str | Path,
	*,
	source: str,
	iz: int,
	title: str,
	out_png: str | Path,
) -> Path:
	"""水平スライスの走時マップ。

	grid_3d shape:
		(nz, ny, nx)
	"""
	if not (0 <= iz < meta.nz):
		raise ValueError(f'iz out of range: {iz} (nz={meta.nz})')
	info = parse_nll_control(control_path)
	out_png = Path(out_png)
	out_png.parent.mkdir(parents=True, exist_ok=True)

	x = meta.x_axis_km()
	y = meta.y_axis_km()
	z_km = meta.z_axis_km()[iz]

	slice_2d = grid_3d[iz, :, :]

	fig, ax = plt.subplots(figsize=(7, 6))
	im = ax.imshow(
		slice_2d,
		origin='lower',
		extent=[x[0], x[-1], y[0], y[-1]],
		aspect='auto',
	)
	_annotate_source_location_xy(ax, info, source=source)
	fig.colorbar(im, ax=ax, label='Travel time (s)')

	ax.set_xlabel('x East (km)')
	ax.set_ylabel('y North (km)')
	ax.set_title(f'{title} | z={z_km:.2f} km')

	fig.tight_layout()
	fig.savefig(out_png, dpi=200)
	plt.close(fig)
	return out_png


def plot_tt_vertical_xz(
	grid_3d: np.ndarray,
	meta: TimeGridMeta,
	*,
	iy: int,
	title: str,
	out_png: str | Path,
) -> Path:
	"""X-Z断面（固定 y インデックス）。"""
	if not (0 <= iy < meta.ny):
		raise ValueError(f'iy out of range: {iy} (ny={meta.ny})')

	out_png = Path(out_png)
	out_png.parent.mkdir(parents=True, exist_ok=True)

	x = meta.x_axis_km()
	z = meta.z_axis_km()

	section = grid_3d[:, iy, :]

	fig, ax = plt.subplots(figsize=(7, 5.5))
	im = ax.imshow(
		section,
		origin='lower',
		extent=[x[0], x[-1], z[0], z[-1]],
		aspect='auto',
	)
	fig.colorbar(im, ax=ax, label='Travel time (s)')

	ax.set_xlabel('x East (km)')
	ax.set_ylabel('z (km)')
	ax.set_title(title)

	fig.tight_layout()
	fig.savefig(out_png, dpi=200)
	plt.close(fig)
	return out_png


def plot_tt_vertical_yz(
	grid_3d: np.ndarray,
	meta: TimeGridMeta,
	*,
	ix: int,
	title: str,
	out_png: str | Path,
) -> Path:
	"""Y-Z断面（固定 x インデックス）。"""
	if not (0 <= ix < meta.nx):
		raise ValueError(f'ix out of range: {ix} (nx={meta.nx})')

	out_png = Path(out_png)
	out_png.parent.mkdir(parents=True, exist_ok=True)

	y = meta.y_axis_km()
	z = meta.z_axis_km()

	section = grid_3d[:, :, ix]

	fig, ax = plt.subplots(figsize=(7, 5.5))
	im = ax.imshow(
		section,
		origin='lower',
		extent=[y[0], y[-1], z[0], z[-1]],
		aspect='auto',
	)
	fig.colorbar(im, ax=ax, label='Travel time (s)')

	ax.set_xlabel('y North (km)')
	ax.set_ylabel('z (km)')
	ax.set_title(title)

	fig.tight_layout()
	fig.savefig(out_png, dpi=200)
	plt.close(fig)
	return out_png


def plot_tt_ps_difference_slice(
	grid_p: np.ndarray,
	grid_s: np.ndarray,
	meta: TimeGridMeta,
	*,
	iz: int,
	title: str,
	out_png: str | Path,
) -> Path:
	"""S - P の水平スライス差分。"""
	if grid_p.shape != grid_s.shape:
		raise ValueError('P/S grid shape mismatch')
	if not (0 <= iz < meta.nz):
		raise ValueError(f'iz out of range: {iz} (nz={meta.nz})')

	out_png = Path(out_png)
	out_png.parent.mkdir(parents=True, exist_ok=True)

	x = meta.x_axis_km()
	y = meta.y_axis_km()
	z_km = meta.z_axis_km()[iz]

	diff = grid_s[iz, :, :] - grid_p[iz, :, :]

	fig, ax = plt.subplots(figsize=(7, 6))
	im = ax.imshow(
		diff,
		origin='lower',
		extent=[x[0], x[-1], y[0], y[-1]],
		aspect='auto',
	)
	fig.colorbar(im, ax=ax, label='S - P (s)')

	ax.set_xlabel('x East (km)')
	ax.set_ylabel('y North (km)')
	ax.set_title(f'{title} | z={z_km:.2f} km')

	fig.tight_layout()
	fig.savefig(out_png, dpi=200)
	plt.close(fig)
	return out_png


# ----------------------------
# High-level QC runner
# ----------------------------
def run_traveltime_tables_qc(
	cfg: QcConfig,
	*,
	preset: str,
	preview_stations: Iterable[str] | None = None,
	iz_preview: int | None = None,
) -> dict[str, Path]:
	"""Grid2Time で作成した走時テーブルの
	「棚卸 + 可視化」をまとめて実行。

	自動決定:
	- fig_dir
	- control_p/control_s
	- tt_out_dir

	可視化の方針:
	- preview_stations を指定しなければ
	  controlの先頭3局を使う
	- iz_preview が None なら
	  中央深さを使う
	"""
	cfg, control_p, control_s, tt_out_dir = resolve_qc_and_tt_paths(cfg, preset)

	tt_out_dir.mkdir(parents=True, exist_ok=True)
	cfg.fig_dir.mkdir(parents=True, exist_ok=True)

	art_files = qc_grid2time_outputs_ps(
		control_p,
		control_s,
		out_dir=tt_out_dir,
	)

	info_p = parse_nll_control(control_p)
	info_s = parse_nll_control(control_s)

	sources_p = info_p['sources']
	sources_s = info_s['sources']
	if not sources_p:
		raise ValueError(f'no GTSRCE in {control_p}')
	if not sources_s:
		raise ValueError(f'no GTSRCE in {control_s}')

	if preview_stations is None:
		preview = sources_p[:3]
	else:
		preview = list(preview_stations)

	meta = TimeGridMeta.from_vggrid(info_p['vggrid'])
	if iz_preview is None:
		iz_preview = max(0, meta.nz // 2)

	fig_dir_tt = cfg.fig_dir / 'traveltime_maps'
	fig_dir_tt.mkdir(parents=True, exist_ok=True)

	figs: dict[str, Path] = {}

	for sta in preview:
		grid_p, meta_p, _, _ = load_time_grid_for_station(control_p, sta)
		grid_s, meta_s, _, _ = load_time_grid_for_station(control_s, sta)

		if meta_p != meta_s:
			raise ValueError(f'P/S meta mismatch for station={sta}')

		title_p = f'TT P | {sta}'
		title_s = f'TT S | {sta}'
		title_d = f'TT (S-P) | {sta}'

		figs[f'{sta}_P_hslice'] = plot_tt_horizontal_slice(
			grid_p,
			meta_p,
			control_path=control_p,
			source=sta,
			iz=iz_preview,
			title=title_p,
			out_png=fig_dir_tt / f'tt_P_{sta}_z{iz_preview:03d}.png',
		)

		figs[f'{sta}_S_hslice'] = plot_tt_horizontal_slice(
			grid_s,
			meta_s,
			control_path=control_p,
			source=sta,
			iz=iz_preview,
			title=title_s,
			out_png=fig_dir_tt / f'tt_S_{sta}_z{iz_preview:03d}.png',
		)

		figs[f'{sta}_SP_diff_hslice'] = plot_tt_ps_difference_slice(
			grid_p,
			grid_s,
			meta_p,
			iz=iz_preview,
			title=title_d,
			out_png=fig_dir_tt / f'tt_SPdiff_{sta}_z{iz_preview:03d}.png',
		)

		iy_mid = meta_p.ny // 2
		ix_mid = meta_p.nx // 2

		figs[f'{sta}_P_xz'] = plot_tt_vertical_xz(
			grid_p,
			meta_p,
			iy=iy_mid,
			title=f'TT P X-Z | {sta} | iy={iy_mid}',
			out_png=fig_dir_tt / f'tt_P_{sta}_xz_iy{iy_mid:03d}.png',
		)

		figs[f'{sta}_P_yz'] = plot_tt_vertical_yz(
			grid_p,
			meta_p,
			ix=ix_mid,
			title=f'TT P Y-Z | {sta} | ix={ix_mid}',
			out_png=fig_dir_tt / f'tt_P_{sta}_yz_ix{ix_mid:03d}.png',
		)

	summary_png_list = cfg.fig_dir / 'traveltime_maps_list.txt'
	summary_png_list.write_text('\n'.join(str(p) for p in figs.values()) + '\n')

	return {
		'fig_dir': cfg.fig_dir,
		'control_p_path': control_p,
		'control_s_path': control_s,
		'tt_out_dir': tt_out_dir,
		'tt_maps_list': summary_png_list,
		**art_files,
		**figs,
	}
