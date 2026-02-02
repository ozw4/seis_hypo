import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import jma.win32_reader as wr  # TODO: 実モジュール名に合わせて修正

DATA_DIR_NAME = 'D20230118000041_20'
EXPECTED_SHA256 = 'c12e20b86da189922af78c4a6e5eb56b2c8cdab28f38f59c23fbe18dc7c7ab3a'

# 5秒・3ch固定（N.KHTH U/N/E）
SELECT_CH_HEX = ['3553', '3554', '3555']
BASE_FS_HZ = 100
DURATION_SEC = 5


def _read_active_ch_table(active_ch_path: Path) -> pd.DataFrame:
	rows = []
	for raw in active_ch_path.read_text(encoding='utf-8').splitlines():
		line = raw.strip()
		if not line or line.startswith('#'):
			continue
		toks = line.split()
		if len(toks) < 13:
			continue

		ch_hex = toks[0].upper()
		ch_int = int(ch_hex, 16)
		station = toks[3]
		component = toks[4]
		conv_coeff = float(toks[12])

		rows.append(
			{
				'ch_hex': ch_hex,
				'ch_int': ch_int,
				'conv_coeff': np.float32(conv_coeff),
				'station': station,
				'component': component,
			}
		)

	df = pd.DataFrame(rows)
	if df.empty:
		raise ValueError(f'failed to parse channel table: {active_ch_path}')

	return df


def _build_mini_win32(evt_full_path: Path, out_path: Path, n_seconds: int) -> None:
	# 先頭4B予約領域 + N秒分(16B header + payload) + 終端(16Bゼロ)
	with evt_full_path.open('rb') as f:
		reserved = f.read(4)
		if len(reserved) != 4:
			raise ValueError(f'truncated reserved area: {evt_full_path.name}')

		out = bytearray(reserved)

		sec = 0
		while sec < n_seconds:
			hdr = f.read(16)
			if len(hdr) != 16:
				raise ValueError(
					f'truncated 16B header at sec={sec}: {evt_full_path.name}'
				)

			block_size = int.from_bytes(hdr[12:16], 'big')
			if block_size == 0:
				raise ValueError(f'terminated early at sec={sec}: {evt_full_path.name}')

			payload = f.read(block_size)
			if len(payload) != block_size:
				raise ValueError(
					f'truncated payload at sec={sec}: expected={block_size}, got={len(payload)}'
				)

			out += hdr
			out += payload
			sec += 1

		out += bytes(16)  # 終端block_size=0

	out_path.write_bytes(out)


def _sha256_float32(arr: np.ndarray) -> str:
	a = np.ascontiguousarray(arr.astype(np.float32, copy=False))
	return hashlib.sha256(a.tobytes()).hexdigest()


def _patch_numba_to_pyfunc(monkeypatch):
	# numba jitのコンパイル時間/環境差を避けてロジック回帰に寄せる
	def py(fn):
		return getattr(fn, 'py_func', fn)

	monkeypatch.setattr(wr, '_sampling_rate', py(wr._sampling_rate), raising=True)
	monkeypatch.setattr(wr, '_channel_no', py(wr._channel_no), raising=True)
	monkeypatch.setattr(wr, '_sample0', py(wr._sample0), raising=True)
	monkeypatch.setattr(wr, '_4bytes', py(wr._4bytes), raising=True)
	monkeypatch.setattr(wr, '_3bytes', py(wr._3bytes), raising=True)
	monkeypatch.setattr(wr, '_2bytes', py(wr._2bytes), raising=True)
	monkeypatch.setattr(wr, '_1byte', py(wr._1byte), raising=True)
	monkeypatch.setattr(wr, '_05byte', py(wr._05byte), raising=True)
	monkeypatch.setattr(
		wr, '_process_secondblock', py(wr._process_secondblock), raising=True
	)
	monkeypatch.setattr(wr, '_datetime', py(wr._datetime), raising=True)
	monkeypatch.setattr(
		wr, '_secondblock_BYTES', py(wr._secondblock_BYTES), raising=True
	)
	monkeypatch.setattr(wr, '_process_file', py(wr._process_file), raising=True)


def test_regression_real_win32_5s(monkeypatch, tmp_path: Path):
	_patch_numba_to_pyfunc(monkeypatch)

	repo_root = Path(__file__).resolve().parents[1]
	data_dir = repo_root / 'tests' / 'data' / DATA_DIR_NAME

	evt_full = data_dir / f'{DATA_DIR_NAME}.evt'
	active_ch = data_dir / f'{DATA_DIR_NAME}_active.ch'

	if not evt_full.is_file():
		raise AssertionError(f'evt not found: {evt_full}')
	if not active_ch.is_file():
		raise AssertionError(f'active ch not found: {active_ch}')

	mini_evt = tmp_path / 'mini_5s.evt'
	_build_mini_win32(evt_full, mini_evt, n_seconds=DURATION_SEC)

	df = _read_active_ch_table(active_ch)

	sel = df[df['ch_hex'].isin([x.upper() for x in SELECT_CH_HEX])].copy()
	if len(sel) != len(SELECT_CH_HEX):
		raise AssertionError(
			f'selected channels not found: want={SELECT_CH_HEX}, got={sel["ch_hex"].tolist()}'
		)

	sel['__order'] = sel['ch_hex'].apply(lambda x: SELECT_CH_HEX.index(x))
	sel = sel.sort_values('__order').drop(columns=['__order']).reset_index(drop=True)

	out = wr.read_win32(
		mini_evt,
		sel,
		base_sampling_rate_HZ=BASE_FS_HZ,
		duration_SECOND=DURATION_SEC,
	)

	assert out.shape == (len(SELECT_CH_HEX), DURATION_SEC * BASE_FS_HZ)
	assert out.dtype == np.float32
	assert np.isfinite(out).all()

	got = _sha256_float32(out)
	assert got == EXPECTED_SHA256
