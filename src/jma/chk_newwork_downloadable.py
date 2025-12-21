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


def probe_networks_by_get_continuous_waveform(
	*,
	network_info: dict[str, str],
	when: dt.datetime,
	base_outdir: str | Path = 'network_test_downloads',
	span_min: int = 1,
	threads: int = 4,
	cleanup: bool = True,
	keep_cnt: bool = True,
) -> pd.DataFrame:
	"""network_codeごとに1回だけ get_continuous_waveform を叩いて可否を調べる。

	- 成功: outdir/<code>/probe_<code>_YYYYMMDDHHMM_1m.cnt + .ch
	- 失敗: error_type/error_msg に理由を保存
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

		try:
			# 互換性のため positional を使う（code,t0,span_min）
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
			ok = True

			# 戻り値が filename / Path / str のどれでも拾えるようにする
			cnt_path = str(outdir / str(data))
			ch_path = str(outdir / str(ctable))

			if not keep_cnt:
				p = Path(cnt_path)
				if p.exists():
					p.unlink()
				cnt_path = ''
		except Exception as e:
			# probe用途：失敗しても継続（例外的にtry/except許容）
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


if __name__ == '__main__':
	# pasted.txt のこれをそのままコピペでOK
	when = dt.datetime(2025, 1, 1, 0, 0, 0)

	NETWORK_INFO = {
		'0101': 'NIED Hi-net',
		'0103': 'NIED F-net (broadband)',
		'0103A': 'NIED F-net (strong motion)',
		'010501': 'NIED V-net (Tokachidake)',
		'010502': 'NIED V-net (Tarumaesan)',
		'010503': 'NIED V-net (Usuzan)',
		'010504': 'NIED V-net (Hokkaido-Komagatake)',
		'010505': 'NIED V-net (Iwatesan)',
		'010506': 'NIED V-net (Nasudake)',
		'010507': 'NIED V-net (Asamayama)',
		'010508': 'NIED V-net (Kusatsu-Shiranesan)',
		'010509': 'NIED V-net (Fujisan)',
		'010510': 'NIED V-net (Miyakejima)',
		'010511': 'NIED V-net (Izu-Oshima)',
		'010512': 'NIED V-net (Asosan)',
		'010513': 'NIED V-net (Unzendake)',
		'010514': 'NIED V-net (Kirishimayama)',
		'0106': 'NIED Temp. obs. in eastern Shikoku',
		'0120': 'NIED S-net (velocity)',
		'0120A': 'NIED S-net (acceleration)',
		'0120B': 'NIED S-net (acceleration 2LG)',
		'0120C': 'NIED S-net (acceleration 2HG)',
		'0131': 'NIED MeSO-net',
		'0201': 'Hokkaido University',
		'0202': 'Tohoku University',
		'0203': 'Tokyo University',
		'0204': 'Kyoto University',
		'0205': 'Kyushu University',
		'0206': 'Hirosaki University',
		'0207': 'Nagoya University',
		'0208': 'Kochi University',
		'0209': 'Kagoshima University',
		'0231': 'MeSO-net (~2017.03)',
		'0301': 'JMA Seismometer Network',
		'0401': 'JAMSTEC Realtime Data from the Deep Sea Floor Observatory',
		'0501': 'AIST',
		'0601': 'GSI',
		'0701': 'Tokyo Metropolitan Government',
		'0702': 'Hot Spring Research Institute of Kanagawa Prefecture',
		'0703': 'Aomori Prefectural Government',
		'0705': 'Shizuoka Prefectural Government',
		'0801': 'ADEP',
	}

	df = probe_networks_by_get_continuous_waveform(
		network_info=NETWORK_INFO,
		when=when,
		base_outdir='network_test_downloads',
		span_min=1,
		threads=4,
		cleanup=True,
		keep_cnt=False,  # .chだけ欲しいならFalseが軽い
	)
	print(df)
