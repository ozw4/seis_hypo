# %%
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from common.time_util import ceil_minutes, floor_minute
from jma.download import (
	_name_stem,
	_supports_station_selection,
	create_hinet_client,
	download_win_for_stations,
)
from jma.prepare.event_dirs import (
	event_dir_date_jst_from_name,
	in_date_range,
	list_event_dirs,
	parse_date_yyyy_mm_dd,
)
from jma.prepare.event_paths import resolve_missing_continuous, resolve_single_evt
from jma.prepare.missing_io import read_missing_by_network
from jma.win32_reader import get_evt_info

# =========================
# 設定（ここを直書きでOK）
# =========================

WIN_EVENT_DIR = Path('/workspace/data/waveform/jma/event').resolve()

TARGET_EVENT_DIR_NAMES: list[str] = []
# 例:
# TARGET_EVENT_DIR_NAMES = ["D20230118000041_20", "D20230119012345_00"]

OUT_SUBDIR = 'continuous'
THREADS = 8
CLEANUP = True
SKIP_IF_EXISTS = True

# ---- 期間フィルタ（ディレクトリ名 DYYYYMMDD... の YYYYMMDD で絞る）----
DATE_MIN: str | None = '2023-01-01'
DATE_MAX: str | None = '2023-01-31'

# ---- 処理済み skip（ネットワーク単位 done マーカー方式） ----
SKIP_IF_DONE = True
RUN_TAG = 'v1'

# ---- returned None のときだけ、最大リトライ回数 ----
MAX_RETRY_REQUEST_NONE = 5

# ---- returned None のときだけ、threadsを段階的に落とす ----
THREADS_LADDER = [8, 4, 2, 1]


# =========================
# 実装
# =========================


@dataclass(frozen=True)
class EventInputs:
	event_dir: Path
	evt_path: Path
	missing_path: Path | None


def _open_log_writer(log_path: Path) -> tuple[object, csv.DictWriter]:
	log_path.parent.mkdir(parents=True, exist_ok=True)
	f = log_path.open('w', newline='', encoding='utf-8')
	fields = [
		'event_dir',
		'evt_file',
		't0_jst',
		'span_min',
		'network_code',
		'n_stations_request',
		'select_used',
		'full_download',
		'threads_used',
		'try_idx',
		'status',
		'cnt_file',
		'ch_file',
		'message',
	]
	w = csv.DictWriter(f, fieldnames=fields)
	w.writeheader()
	return f, w


def _net_done_path(
	event_dir: Path, *, evt_stem: str, run_tag: str, network_code: str
) -> Path:
	# ネットワークごとに done を残す（再開可能にするため）
	return event_dir / f'{evt_stem}_continuous_done_{run_tag}_{network_code}.json'


def _should_skip_net_done(done_path: Path, *, run_tag: str) -> bool:
	if not done_path.is_file():
		return False
	try:
		obj = json.loads(done_path.read_text(encoding='utf-8'))
	except Exception:
		return False
	if str(obj.get('run_tag', '')) != str(run_tag):
		return False
	return str(obj.get('status', '')) in {'done', 'exists', 'no_missing_file'}


def _write_net_done(
	done_path: Path,
	*,
	evt_file: str,
	run_tag: str,
	network_code: str,
	status: str,
	cnt_file: str,
	ch_file: str,
	message: str,
	n_stations_request: int,
	threads_used: int,
	try_idx: int,
) -> None:
	obj = {
		'evt_file': str(evt_file),
		'run_tag': str(run_tag),
		'network_code': str(network_code),
		'status': str(status),
		'cnt_file': str(cnt_file),
		'ch_file': str(ch_file),
		'message': str(message),
		'n_stations_request': int(n_stations_request),
		'threads_used': int(threads_used),
		'try_idx': int(try_idx),
	}
	done_path.write_text(
		json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
	)


def main() -> None:
	if not WIN_EVENT_DIR.is_dir():
		raise FileNotFoundError(WIN_EVENT_DIR)

	run_tag2 = str(RUN_TAG).strip()
	if not run_tag2:
		raise ValueError('RUN_TAG must be non-empty')

	dmin = parse_date_yyyy_mm_dd(DATE_MIN)
	dmax = parse_date_yyyy_mm_dd(DATE_MAX)
	if dmin is not None and dmax is not None and dmax < dmin:
		raise ValueError(f'DATE_MAX < DATE_MIN: {dmax} < {dmin}')

	if int(MAX_RETRY_REQUEST_NONE) <= 0:
		raise ValueError('MAX_RETRY_REQUEST_NONE must be >= 1')

	ladder = [int(x) for x in THREADS_LADDER]
	if not ladder:
		raise ValueError('THREADS_LADDER must be non-empty')
	if any(x <= 0 for x in ladder):
		raise ValueError(f'invalid THREADS_LADDER: {THREADS_LADDER}')
	if ladder[0] != int(THREADS):
		print(
			f'[warn] THREADS={THREADS} but THREADS_LADDER[0]={ladder[0]} '
			'(keeping ladder as-is)',
			flush=True,
		)

	client = create_hinet_client()

	event_dirs = list_event_dirs(WIN_EVENT_DIR, target_names=TARGET_EVENT_DIR_NAMES)
	if not event_dirs:
		raise RuntimeError(f'no event dirs under: {WIN_EVENT_DIR}')

	for event_dir in event_dirs:
		# ---- 期間フィルタ（ディレクトリ名の日付）----
		if dmin is not None or dmax is not None:
			dd = event_dir_date_jst_from_name(event_dir.name)
			if not in_date_range(dd, date_min=dmin, date_max=dmax):
				continue

		try:
			evt_path = resolve_single_evt(event_dir)
		except ValueError as e:
			msg = str(e)
			if msg.startswith('.evt must be exactly 1 in ') and '(found 0)' in msg:
				print(f'[warn] skip event (no .evt): {event_dir}', flush=True)
				continue
			raise
		missing_path = resolve_missing_continuous(event_dir, stem=evt_path.stem)
		inp = EventInputs(
			event_dir=event_dir, evt_path=evt_path, missing_path=missing_path
		)

		# missing が無いイベントはネットワーク処理自体が無いので何もしない
		if inp.missing_path is None:
			continue

		log_path = inp.event_dir / f'{inp.evt_path.stem}_continuous_download_log.csv'
		log_f, writer = _open_log_writer(log_path)

		try:
			evt_info = get_evt_info(inp.evt_path, scan_rate_blocks=1)
			t_start = evt_info.start_time
			t_end = evt_info.end_time_exclusive
			t0 = floor_minute(t_start)
			span_min = min(
				ceil_minutes((t_end - t0).total_seconds()), 3
			)  # max time 3 min

			stations_by_network = read_missing_by_network(inp.missing_path)
			outdir = inp.event_dir / OUT_SUBDIR
			outdir.mkdir(parents=True, exist_ok=True)

			for network_code, stations in stations_by_network.items():
				net_done = _net_done_path(
					inp.event_dir,
					evt_stem=inp.evt_path.stem,
					run_tag=run_tag2,
					network_code=str(network_code),
				)
				if SKIP_IF_DONE and _should_skip_net_done(net_done, run_tag=run_tag2):
					continue

				select_supported = _supports_station_selection(network_code)
				full_download = not select_supported

				stations_for_name = stations if select_supported else ['ALL']
				stem = _name_stem(network_code, t0, stations_for_name, span_min)
				data_name = f'{stem}.cnt'
				ctable_name = f'{stem}.ch'
				cnt_path = outdir / data_name
				ch_path = outdir / ctable_name

				try:
					if SKIP_IF_EXISTS and cnt_path.exists() and ch_path.exists():
						writer.writerow(
							{
								'event_dir': str(inp.event_dir),
								'evt_file': inp.evt_path.name,
								't0_jst': f'{t0:%Y-%m-%d %H:%M:%S}',
								'span_min': span_min,
								'network_code': network_code,
								'n_stations_request': len(stations),
								'select_used': select_supported,
								'full_download': full_download,
								'threads_used': '',
								'try_idx': '',
								'status': 'exists',
								'cnt_file': cnt_path.name,
								'ch_file': ch_path.name,
								'message': '',
							}
						)
						_write_net_done(
							net_done,
							evt_file=inp.evt_path.name,
							run_tag=run_tag2,
							network_code=str(network_code),
							status='exists',
							cnt_file=cnt_path.name,
							ch_file=ch_path.name,
							message='',
							n_stations_request=len(stations),
							threads_used=0,
							try_idx=0,
						)
						continue

					cnt_out = None
					ch_out = None
					select_used = False
					last_msg = ''
					last_threads = 0
					last_try_idx = 0

					# None のときだけ threads を落とす：8→4→2→1
					for th in ladder:
						for k in range(1, int(MAX_RETRY_REQUEST_NONE) + 1):
							last_threads = int(th)
							last_try_idx = int(k)
							try:
								cnt_out, ch_out, select_used = (
									download_win_for_stations(
										client,
										stations=stations,
										when=t0,
										network_code=str(network_code),
										span_min=span_min,
										outdir=outdir,
										threads=int(th),
										cleanup=CLEANUP,
										clear_selection=False,
										skip_if_exists=False,
										use_select=select_supported,
										data_name=data_name,
										ctable_name=ctable_name,
									)
								)
								last_msg = ''
								break
							except ValueError as e:
								msg = str(e)
								last_msg = msg
								if msg.startswith(
									'Fail to request WIN32 (returned None).'
								):
									print(
										f'[warn] returned None -> retry {k}/{MAX_RETRY_REQUEST_NONE} '
										f'(threads={th}): code={network_code} start={t0} '
										f'span_min={span_min} n_stations={len(stations)}',
										flush=True,
									)
									continue
								raise
						if cnt_out is not None and ch_out is not None:
							break

					if cnt_out is None or ch_out is None:
						# 警告付きフォールバック：どうしても None が続くので skip
						print(
							f'[warn] skip network (returned None) after ladder+retries: '
							f'code={network_code} start={t0} span_min={span_min} '
							f'n_stations={len(stations)}',
							flush=True,
						)
						writer.writerow(
							{
								'event_dir': str(inp.event_dir),
								'evt_file': inp.evt_path.name,
								't0_jst': f'{t0:%Y-%m-%d %H:%M:%S}',
								'span_min': span_min,
								'network_code': network_code,
								'n_stations_request': len(stations),
								'select_used': select_supported,
								'full_download': full_download,
								'threads_used': last_threads,
								'try_idx': last_try_idx,
								'status': 'request_failed_skip',
								'cnt_file': '',
								'ch_file': '',
								'message': last_msg,
							}
						)
						# skip は done にしない（次回このネットから再挑戦できる）
						continue

					writer.writerow(
						{
							'event_dir': str(inp.event_dir),
							'evt_file': inp.evt_path.name,
							't0_jst': f'{t0:%Y-%m-%d %H:%M:%S}',
							'span_min': span_min,
							'network_code': network_code,
							'n_stations_request': len(stations),
							'select_used': bool(select_used),
							'full_download': full_download,
							'threads_used': last_threads,
							'try_idx': last_try_idx,
							'status': 'downloaded',
							'cnt_file': Path(cnt_out).name,
							'ch_file': Path(ch_out).name,
							'message': '',
						}
					)
					_write_net_done(
						net_done,
						evt_file=inp.evt_path.name,
						run_tag=run_tag2,
						network_code=str(network_code),
						status='done',
						cnt_file=Path(cnt_out).name,
						ch_file=Path(ch_out).name,
						message='',
						n_stations_request=len(stations),
						threads_used=last_threads,
						try_idx=last_try_idx,
					)

				finally:
					if select_supported:
						client.select_stations(network_code)

		finally:
			log_f.close()


if __name__ == '__main__':
	main()

# %%
