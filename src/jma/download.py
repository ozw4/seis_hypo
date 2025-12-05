from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Sequence
from pathlib import Path

from HinetPy import Client


def _name_stem(
	network_code: str, t0: dt.datetime, stations: Sequence[str], span_min: int
) -> str:
	# 安定名: ネット+分時刻+span+stations の短いハッシュ
	key = f'{network_code}|{t0:%Y%m%d%H%M}|{span_min}|{",".join(sorted(stations))}'
	digest = hashlib.sha1(key.encode()).hexdigest()[:8]
	return f'win_{network_code}_{t0:%Y%m%d%H%M}_{span_min}m_{digest}'


def download_win_for_stations(
	client: Client,
	stations: str | Sequence[str],
	when: dt.datetime,
	*,
	network_code: str = '0101',
	span_min: int = 1,
	outdir: str | Path = '.',
	threads: int = 8,
	cleanup: bool = True,
	clear_selection: bool = False,
	skip_if_exists: bool = True,  # ★追加
	data_name: str | None = None,
	ctable_name: str | None = None,
) -> tuple[Path, Path]:
	"""指定ステーション群・指定時刻（JST）の1分 Win32 をダウンロード。既存ならスキップ。"""
	t0 = when.replace(second=0, microsecond=0)
	outdir = Path(outdir)
	outdir.mkdir(parents=True, exist_ok=True)
	station_list = [stations] if isinstance(stations, str) else list(stations)
	if not station_list:
		raise ValueError('stations が空です。')

	# ここで決定的なファイル名を作る（明示指定があればそれを優先）
	if data_name is None or ctable_name is None:
		stem = _name_stem(network_code, t0, station_list, span_min)
		data_name = data_name or f'{stem}.cnt'
		ctable_name = ctable_name or f'{stem}.ch'

	cnt_path = outdir / data_name
	ch_path = outdir / ctable_name

	# ★ 既存チェック（両方あるときのみスキップ）
	if skip_if_exists and cnt_path.exists() and ch_path.exists():
		return cnt_path, ch_path

	# サーバ側の局選択
	client.select_stations(network_code, station_list)

	# ダウンロード（明示ファイル名を指定）
	data, ctable = client.get_continuous_waveform(
		network_code,
		t0,
		span_min,
		outdir=str(outdir),
		data=data_name,
		ctable=ctable_name,
		threads=threads,
		cleanup=cleanup,
	)

	if clear_selection:
		client.select_stations(network_code)

	return outdir / data, outdir / ctable
