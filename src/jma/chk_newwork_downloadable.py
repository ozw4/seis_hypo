# %%
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from jma.download import create_hinet_client


def _to_t0_str(when: dt.datetime) -> str:
	# HinetPyは 'YYYYMMDDHHMM' をよく使うので固定
	if when.tzinfo is not None:
		# ここは「probe用途」なので、tz-aware渡すなら先にJST/UTCの方針決めてからにして
		raise ValueError(
			'when must be timezone-naive JST datetime for this probe script'
		)
	return when.strftime('%Y%m%d%H%M')


def _normalize_return_path(outdir: Path, ret: object | None) -> str:
	"""get_continuous_waveformの戻り値(data/ctable)を実在パス優先で正規化して返す。"""
	if ret is None:
		return ''

	p = Path(str(ret))

	# 1) 絶対パスで実在
	if p.is_absolute() and p.exists():
		return str(p)

	# 2) 返り値が outdir 付き相対パスで実在（例: network_test_downloads/0101/xxx.ch）
	if p.exists():
		return str(p)

	# 3) ファイル名だけを返してくるケース
	cand = outdir / p
	if cand.exists():
		return str(cand)

	# 4) 実在しない（空扱い）
	return ''


def probe_networks_by_get_continuous_waveform(
	*,
	network_info: dict[str, str],
	when,
	base_outdir: str | Path = 'network_test_downloads',
	span_min: int = 1,
	threads: int = 4,
	cleanup: bool = True,
	keep_cnt: bool = True,
	skip_if_ch_exists: bool = True,
) -> pd.DataFrame:
	"""network_codeごとに1回だけ get_continuous_waveform を叩いて可否を調べる。

	success条件は「.ch が存在すること」。
	"""
	if not network_info:
		raise ValueError('network_info is empty')

	t0 = _to_t0_str(when)

	base_outdir = Path(base_outdir)
	base_outdir.mkdir(parents=True, exist_ok=True)

	client = create_hinet_client()

	rows: list[dict[str, str]] = []
	for code, name in network_info.items():
		code = str(code).strip()
		if not code:
			raise ValueError('empty network_code in network_info')

		outdir = base_outdir / code
		outdir.mkdir(parents=True, exist_ok=True)

		cnt_name = f'probe_{code}_{t0}_{int(span_min)}m.cnt'
		ch_name = f'probe_{code}_{t0}_{int(span_min)}m.ch'

		ok = False
		cnt_path = ''
		ch_path = ''
		error_type = ''
		error_msg = ''

		expected_cnt = outdir / cnt_name
		expected_ch = outdir / ch_name

		# 既に .ch があるなら通信しない（OK扱いで out_csv にも出す）
		if bool(skip_if_ch_exists) and expected_ch.is_file():
			ok = True
			ch_path = str(expected_ch)

			if bool(keep_cnt):
				if expected_cnt.is_file():
					cnt_path = str(expected_cnt)
			elif expected_cnt.exists():
				expected_cnt.unlink()

			error_type = 'SkippedExistingChannelTable'
			error_msg = f'used existing .ch: {ch_path}'

			rows.append(
				{
					'network_code': code,
					'network_name': str(name),
					'ok': str(ok),
					'outdir': str(outdir),
					'cnt_path': cnt_path,
					'ch_path': ch_path,
					'error_type': error_type,
					'error_msg': error_msg,
				}
			)
			continue

		try:
			data, ctable = client.get_continuous_waveform(
				code,
				t0,
				int(span_min),
				outdir=str(outdir),
				data=cnt_name,
				ctable=ch_name,
				threads=int(threads),
				cleanup=bool(cleanup),
			)

			# 戻り値パス正規化（outdir二重連結も潰す）
			cnt_path = _normalize_return_path(outdir, data)
			ch_path = _normalize_return_path(outdir, ctable)

			# .ch が取れないケースは「失敗」
			if not ch_path:
				error_type = 'MissingChannelTable'
				error_msg = 'ctable is None or .ch not found'
				ok = False
			else:
				ok = Path(ch_path).is_file()
				if not ok:
					error_type = 'MissingChannelTable'
					error_msg = f'.ch not found: {ch_path}'

			if (not keep_cnt) and cnt_path:
				p = Path(cnt_path)
				if p.exists():
					p.unlink()
				cnt_path = ''

		except Exception as e:
			error_type = type(e).__name__
			error_msg = str(e)
			print(
				f'[WARN] network_code={code} connection failed: {error_type}: {error_msg}'
			)

		rows.append(
			{
				'network_code': code,
				'network_name': str(name),
				'ok': str(ok),
				'outdir': str(outdir),
				'cnt_path': cnt_path,
				'ch_path': ch_path,
				'error_type': error_type,
				'error_msg': error_msg,
			}
		)

	df = pd.DataFrame(
		rows,
		columns=[
			'network_code',
			'network_name',
			'ok',
			'outdir',
			'cnt_path',
			'ch_path',
			'error_type',
			'error_msg',
		],
	)

	out_csv = base_outdir / 'network_probe_results.csv'
	df.to_csv(out_csv, index=False, encoding='utf-8')
	print(f'[INFO] wrote: {out_csv}')

	return df
