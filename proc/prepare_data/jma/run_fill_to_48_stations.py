# %%
# proc/prepare_data/jma/run_fill_to_48_stations.py （薄いラッパー / Step3）
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from common.done_marker import read_done_json, should_skip_done, write_done_json
from common.geo import haversine_distance_km
from common.time_util import ceil_minutes, floor_minute
from jma.download import _name_stem, create_hinet_client, download_win_for_stations
from jma.prepare.event_dirs import (
	event_dir_date_jst_from_name,
	in_date_range,
	list_event_dirs,
	parse_date_yyyy_mm_dd,
)
from jma.prepare.event_paths import (
	resolve_active_ch,
	resolve_missing_continuous,
	resolve_single_evt,
	resolve_txt_for_evt,
)
from jma.prepare.missing_io import read_missing_pairs
from jma.station_reader import read_hinet_channel_table
from jma.stationcode_common import normalize_code, normalize_network_code
from jma.stationcode_presence import PresenceDB, load_presence_db
from jma.win32_reader import get_evt_info, scan_channel_sampling_rate_map_win32

# =========================
# 設定（ここを直書きでOK）
# =========================

WIN_EVENT_DIR = Path('/workspace/data/waveform/jma/event').resolve()

TARGET_EVENT_DIR_NAMES: list[str] = []
# 例:
# TARGET_EVENT_DIR_NAMES = ["D20230118000041_20", "D20230119012345_00"]

# ---- 追加：期間フィルタ（ディレクトリ名 DYYYYMMDD... の YYYYMMDD で日付粒度）----
DATE_MIN: str | None = '2023-01-01'  # 例: "2023-01-01"
DATE_MAX: str | None = '2023-01-31'  # 例: "2023-01-31"

MIN_STATIONS = 48
FILL_NETWORK_CODE = '0101'
NEAR_KM = 200.0

PRES_CSV = Path(
	'/workspace/proc/prepare_data/jma/stationcode_match/v1/snapshots/monthly/monthly_presence.csv'
).resolve()

HINET_CHANNEL_TABLE_PATH = Path(
	'/workspace/data/station/jma/hinet_channelstbl_20251007'
).resolve()

OUT_SUBDIR = 'continuous'
THREADS = 8
CLEANUP = True
SKIP_IF_EXISTS = True

CONT_SCAN_SECOND_BLOCKS = 3

# ---- Step2 実行済みチェック（doneマーカー） ----
SKIP_IF_STEP2_NOT_DONE = True
STEP2_RUN_TAG = 'v1'  # Step2側の RUN_TAG と合わせる

# ---- Step3 実行済みチェック（doneマーカー） ----
SKIP_IF_STEP3_DONE = True
STEP3_RUN_TAG = 'v1'  # ルール更新でやり直したい時はここを変える

# ---- ダウンロードは最大3分（Step3のみ） ----
MAX_SPAN_MIN = 3

# ---- ダウンロード失敗時の再挑戦 ----
MAX_RETRY_DOWNLOAD = 5  # 5回
# threads は段階的に落とす（例: 8->4->2->1->1）

# ===== Debug print =====
DEBUG_PRINT = True
DEBUG_PRINT_CNT_SUMMARY = True
DEBUG_PRINT_SELECTED_HEAD = 10


def _p(msg: str) -> None:
	if DEBUG_PRINT:
		print(msg, flush=True)


# =========================
# 実装
# =========================


@dataclass(frozen=True)
class EventInputs:
	event_dir: Path
	evt_path: Path
	active_ch_path: Path
	event_txt_path: Path
	missing_path: Path | None


def _parse_latlon_with_hemisphere(s: str) -> float:
	ss = s.strip()
	if len(ss) < 2:
		raise ValueError(f'invalid lat/lon token: {s!r}')
	hem = ss[-1].upper()
	val = float(ss[:-1])
	if hem in {'N', 'E'}:
		return val
	if hem in {'S', 'W'}:
		return -val
	raise ValueError(f'invalid hemisphere in lat/lon: {s!r}')


def _parse_origin_jst(s: str) -> datetime:
	ss = s.strip()
	parts = ss.split()
	if len(parts) != 2:
		raise ValueError(f'invalid ORIGIN_JST: {s!r}')
	date_part, time_part = parts
	y_s, m_s, d_s = date_part.split('/')
	h_s, mi_s, sec_s = time_part.split(':')

	if '.' in sec_s:
		sec_int_s, frac_s = sec_s.split('.', 1)
		frac_s = ''.join([c for c in frac_s if c.isdigit()])
		frac_s = (frac_s + '000000')[:6]
		micro = int(frac_s)
	else:
		sec_int_s = sec_s
		micro = 0

	return datetime(
		int(y_s),
		int(m_s),
		int(d_s),
		int(h_s),
		int(mi_s),
		int(sec_int_s),
		micro,
	)


@dataclass(frozen=True)
class EventMeta:
	origin_jst: datetime
	event_month: str  # YYYY-MM
	lat: float
	lon: float


def _read_event_txt_meta(path: Path) -> EventMeta:
	lines = path.read_text(encoding='utf-8', errors='ignore').splitlines()

	kv: dict[str, str] = {}
	for raw in lines:
		line = raw.strip()
		if not line or ':' not in line:
			continue
		k, v = line.split(':', 1)
		key = k.strip()
		val = v.strip()
		if key and val:
			kv[key] = val

	req_keys = {'ORIGIN_JST', 'LATITUDE', 'LONGITUDE'}
	missing = sorted(req_keys - set(kv.keys()))
	if missing:
		raise ValueError(f'missing keys in {path.name}: {missing}')

	origin = _parse_origin_jst(kv['ORIGIN_JST'])
	month = f'{origin.year:04d}-{origin.month:02d}'
	lat = _parse_latlon_with_hemisphere(kv['LATITUDE'])
	lon = _parse_latlon_with_hemisphere(kv['LONGITUDE'])

	return EventMeta(origin_jst=origin, event_month=month, lat=lat, lon=lon)


def _open_log_writer(log_path: Path) -> tuple[object, csv.DictWriter]:
	log_path.parent.mkdir(parents=True, exist_ok=True)
	f = log_path.open('w', newline='', encoding='utf-8')
	fields = [
		'event_dir',
		'evt_file',
		'event_month',
		'event_lat',
		'event_lon',
		't0_jst',
		'span_min',
		'n_before',
		'n_need',
		'n_selected',
		'n_after',
		'network_code',
		'select_used',
		'status',
		'cnt_file',
		'ch_file',
		'message',
	]
	w = csv.DictWriter(f, fieldnames=fields)
	w.writeheader()
	return f, w


def _load_station_geo_0101(channel_table_path: Path) -> dict[str, tuple[float, float]]:
	df = read_hinet_channel_table(channel_table_path)
	if df.empty:
		raise ValueError(f'empty channel table: {channel_table_path}')

	df2 = (
		df[['station', 'lat', 'lon']]
		.groupby('station', as_index=False)
		.agg({'lat': 'first', 'lon': 'first'})
	)

	out: dict[str, tuple[float, float]] = {}
	for _, r in df2.iterrows():
		sta = normalize_code(r['station'])
		out[sta] = (float(r['lat']), float(r['lon']))
	if not out:
		raise ValueError('no station geo rows loaded from channel table')
	return out


def _stations_in_active_ch(active_ch_path: Path) -> set[str]:
	df = read_hinet_channel_table(active_ch_path)
	stas = df['station'].astype(str).map(normalize_code).tolist()
	return set([s for s in stas if s])


def _parse_network_code_from_cnt_name(cnt_path: Path) -> str:
	parts = cnt_path.stem.split('_')
	if len(parts) < 5 or parts[0] != 'win':
		raise ValueError(f'unexpected cnt filename: {cnt_path.name}')
	return parts[1]


def _scan_cnt_present_station_rows(
	cnt_path: Path, ch_path: Path
) -> tuple[set[str], dict[str, tuple[float, float]]]:
	fs_by_ch = scan_channel_sampling_rate_map_win32(
		cnt_path, max_second_blocks=CONT_SCAN_SECOND_BLOCKS
	)
	if not fs_by_ch:
		if DEBUG_PRINT_CNT_SUMMARY:
			_p(f'    [cnt-scan] {cnt_path.name}: no channels found -> 0 stations')
		return set(), {}

	present_ch = set(int(x) for x in fs_by_ch.keys())

	df = read_hinet_channel_table(ch_path)
	df2 = df[df['ch_int'].isin(present_ch)]
	if df2.empty:
		if DEBUG_PRINT_CNT_SUMMARY:
			_p(
				f'    [cnt-scan] {cnt_path.name}: present_ch={len(present_ch)} '
				f'but matched ch rows=0 -> 0 stations'
			)
		return set(), {}

	stations = df2['station'].astype(str).map(normalize_code).tolist()
	out_set = set([s for s in stations if s])

	coord: dict[str, tuple[float, float]] = {}
	for _, r in df2.iterrows():
		sta = normalize_code(str(r['station']))
		if not sta or sta in coord:
			continue
		coord[sta] = (float(r['lat']), float(r['lon']))

	if DEBUG_PRINT_CNT_SUMMARY:
		_p(
			f'    [cnt-scan] {cnt_path.name}: present_ch={len(present_ch)} '
			f'matched_rows={len(df2)} stations={len(out_set)}'
		)

	return out_set, coord


@dataclass(frozen=True)
class ContinuousInventory:
	present_by_net: dict[str, set[str]]
	coord_by_station: dict[str, tuple[float, float]]


def _scan_continuous_inventory(cont_dir: Path) -> ContinuousInventory:
	present_by_net: dict[str, set[str]] = {}
	coord_by_station: dict[str, tuple[float, float]] = {}

	if not cont_dir.is_dir():
		return ContinuousInventory(
			present_by_net=present_by_net, coord_by_station=coord_by_station
		)

	cnt_files = sorted(cont_dir.glob('*.cnt'))
	_p(f'  [count] scan continuous cnt files: {len(cnt_files)}')

	for cnt_path in cnt_files:
		ch_path = cnt_path.with_suffix('.ch')
		if not ch_path.is_file():
			raise FileNotFoundError(f'missing .ch for .cnt: {cnt_path} -> {ch_path}')

		net = normalize_network_code(_parse_network_code_from_cnt_name(cnt_path))
		stas, coord = _scan_cnt_present_station_rows(cnt_path, ch_path)

		if stas:
			present_by_net.setdefault(net, set()).update(stas)

		for sta, ll in coord.items():
			if sta not in coord_by_station:
				coord_by_station[sta] = ll

	return ContinuousInventory(
		present_by_net=present_by_net, coord_by_station=coord_by_station
	)


def _missing_recovered_set(
	missing_pairs: list[tuple[str, str]],
	present_by_net: dict[str, set[str]],
) -> set[str]:
	out: set[str] = set()
	for sta, net in missing_pairs:
		if sta in present_by_net.get(net, set()):
			out.add(sta)
	return out


def _near200_from_continuous(
	*,
	event_lat: float,
	event_lon: float,
	base: set[str],
	coord_by_station: dict[str, tuple[float, float]],
) -> set[str]:
	cands = [sta for sta in coord_by_station if sta not in base]
	if not cands:
		return set()

	lat_arr = np.asarray([coord_by_station[s][0] for s in cands], dtype=float)
	lon_arr = np.asarray([coord_by_station[s][1] for s in cands], dtype=float)

	dist_km = haversine_distance_km(
		lat0_deg=float(event_lat),
		lon0_deg=float(event_lon),
		lat_deg=lat_arr,
		lon_deg=lon_arr,
	)

	mask = dist_km <= float(NEAR_KM)
	if not np.any(mask):
		return set()

	return set([cands[int(i)] for i in np.where(mask)[0]])


def _presence_station_candidates_0101(
	pdb: PresenceDB, *, event_month: str
) -> list[str]:
	if event_month not in pdb.month_cols:
		raise ValueError(f'event_month={event_month} not in presence columns')

	net = normalize_network_code(FILL_NETWORK_CODE)
	df = pdb.pres
	hit = df[(df['network_code'] == net) & (df[event_month] == 1)]
	if hit.empty:
		return []
	cands = hit['ch_key'].astype(str).map(normalize_code).tolist()
	return sorted(set([c for c in cands if c]))


def _select_nearest_0101(
	*,
	event_lat: float,
	event_lon: float,
	candidate_stations: list[str],
	station_geo_0101: dict[str, tuple[float, float]],
	n_need: int,
) -> tuple[list[str], list[tuple[str, float]]]:
	if n_need <= 0:
		return [], []

	cands = [normalize_code(s) for s in candidate_stations]
	cands = [s for s in cands if s in station_geo_0101]
	if not cands:
		return [], []

	lat_arr = np.asarray([station_geo_0101[s][0] for s in cands], dtype=float)
	lon_arr = np.asarray([station_geo_0101[s][1] for s in cands], dtype=float)

	dist_km = haversine_distance_km(
		lat0_deg=float(event_lat),
		lon0_deg=float(event_lon),
		lat_deg=lat_arr,
		lon_deg=lon_arr,
	)

	order = np.argsort(dist_km)
	k = int(min(n_need, len(cands)))
	picked = [cands[int(i)] for i in order[:k]]
	picked_with_dist = [(cands[int(i)], float(dist_km[int(i)])) for i in order[:k]]
	return picked, picked_with_dist


def _step2_done_exists(event_dir: Path, *, evt_stem: str, run_tag: str) -> bool:
	p0 = event_dir / f'{evt_stem}_continuous_download_done_{run_tag}.json'
	if p0.is_file():
		return True
	pats = list(event_dir.glob(f'{evt_stem}_continuous_download_done_{run_tag}_*.json'))
	return len(pats) > 0


def _step3_done_path(event_dir: Path, *, evt_stem: str, run_tag: str) -> Path:
	return event_dir / f'{evt_stem}_fill_to_48_done_{run_tag}.json'


def _should_skip_step3_done(done_path: Path, *, run_tag: str) -> bool:
	obj = read_done_json(done_path, on_missing='empty', on_error='empty')
	return should_skip_done(obj, run_tag=run_tag, ok_statuses=None)


def _write_step3_done(done_path: Path, obj: dict[str, object]) -> None:
	write_done_json(done_path, obj)


def _threads_schedule(base_threads: int, n_try: int) -> list[int]:
	bt = int(base_threads)
	if bt <= 0:
		raise ValueError(f'invalid THREADS={base_threads}')
	if int(n_try) <= 0:
		raise ValueError(f'invalid MAX_RETRY_DOWNLOAD={n_try}')

	seq: list[int] = [bt]
	while len(seq) < 4 and seq[-1] > 1:
		seq.append(max(1, seq[-1] // 2))
	if seq[-1] != 1:
		seq.append(1)
	while len(seq) < int(n_try):
		seq.append(seq[-1])
	return seq[: int(n_try)]


def main() -> None:
	_p('[start] run_fill_to_48_stations.py')
	_p(f'  WIN_EVENT_DIR={WIN_EVENT_DIR}')
	_p(f'  DATE_MIN={DATE_MIN} DATE_MAX={DATE_MAX}')
	_p(
		f'  MIN_STATIONS={MIN_STATIONS} FILL_NETWORK_CODE={FILL_NETWORK_CODE} NEAR_KM={NEAR_KM}'
	)
	_p(f'  PRES_CSV={PRES_CSV}')
	_p(f'  HINET_CHANNEL_TABLE_PATH={HINET_CHANNEL_TABLE_PATH}')
	_p(f'  OUT_SUBDIR={OUT_SUBDIR} SKIP_IF_EXISTS={SKIP_IF_EXISTS}')
	_p(
		f'  THREADS={THREADS} CLEANUP={CLEANUP} CONT_SCAN_SECOND_BLOCKS={CONT_SCAN_SECOND_BLOCKS}'
	)
	_p(
		f'  STEP2_RUN_TAG={STEP2_RUN_TAG} SKIP_IF_STEP2_NOT_DONE={SKIP_IF_STEP2_NOT_DONE}'
	)
	_p(f'  STEP3_RUN_TAG={STEP3_RUN_TAG} SKIP_IF_STEP3_DONE={SKIP_IF_STEP3_DONE}')
	_p(f'  MAX_SPAN_MIN={MAX_SPAN_MIN} MAX_RETRY_DOWNLOAD={MAX_RETRY_DOWNLOAD}')

	if not WIN_EVENT_DIR.is_dir():
		raise FileNotFoundError(WIN_EVENT_DIR)
	if not PRES_CSV.is_file():
		raise FileNotFoundError(PRES_CSV)
	if not HINET_CHANNEL_TABLE_PATH.is_file():
		raise FileNotFoundError(HINET_CHANNEL_TABLE_PATH)
	if int(MAX_SPAN_MIN) <= 0:
		raise ValueError('MAX_SPAN_MIN must be >= 1')
	if int(MAX_RETRY_DOWNLOAD) <= 0:
		raise ValueError('MAX_RETRY_DOWNLOAD must be >= 1')

	dmin = parse_date_yyyy_mm_dd(DATE_MIN)
	dmax = parse_date_yyyy_mm_dd(DATE_MAX)
	if dmin is not None and dmax is not None and dmax < dmin:
		raise ValueError(f'DATE_MAX < DATE_MIN: {dmax} < {dmin}')

	_p('[load] presence db ...')
	pdb = load_presence_db(PRES_CSV)
	_p(f'  presence months={len(pdb.month_cols)} rows={len(pdb.pres)}')

	_p('[load] station geo (0101) ...')
	station_geo_0101 = _load_station_geo_0101(HINET_CHANNEL_TABLE_PATH)
	_p(f'  station_geo size={len(station_geo_0101)}')

	_p('[init] hinet client ...')
	client = create_hinet_client()

	event_dirs = list_event_dirs(WIN_EVENT_DIR, target_names=TARGET_EVENT_DIR_NAMES)
	if not event_dirs:
		raise RuntimeError(f'no event dirs under: {WIN_EVENT_DIR}')
	_p(f'[plan] event dirs={len(event_dirs)}')

	step2_tag = str(STEP2_RUN_TAG).strip()
	step3_tag = str(STEP3_RUN_TAG).strip()
	if not step2_tag:
		raise ValueError('STEP2_RUN_TAG must be non-empty')
	if not step3_tag:
		raise ValueError('STEP3_RUN_TAG must be non-empty')

	thread_seq = _threads_schedule(THREADS, int(MAX_RETRY_DOWNLOAD))
	_p(f'  retry threads schedule={thread_seq}')

	for i, event_dir in enumerate(event_dirs, 1):
		# ---- 期間フィルタ（ディレクトリ名の日付）----
		if dmin is not None or dmax is not None:
			dd = event_dir_date_jst_from_name(event_dir.name)
			if not in_date_range(dd, date_min=dmin, date_max=dmax):
				continue

		_p(f'\n[event {i}/{len(event_dirs)}] {event_dir.name}')

		evt_path = resolve_single_evt(event_dir, allow_none=True)
		if evt_path is None:
			_p(f'  [warn] skip (no .evt): {event_dir}')
			continue
		active_ch_path = resolve_active_ch(event_dir, mode='glob_single')
		event_txt_path = resolve_txt_for_evt(evt_path)
		missing_path = resolve_missing_continuous(event_dir, stem=evt_path.stem)
		inp = EventInputs(
			event_dir=event_dir,
			evt_path=evt_path,
			active_ch_path=active_ch_path,
			event_txt_path=event_txt_path,
			missing_path=missing_path,
		)

		if SKIP_IF_STEP2_NOT_DONE:
			if not _step2_done_exists(
				inp.event_dir, evt_stem=inp.evt_path.stem, run_tag=step2_tag
			):
				_p(f'  [warn] skip (Step2 not done): {inp.evt_path.stem}')
				continue

		done3_path = _step3_done_path(
			inp.event_dir, evt_stem=inp.evt_path.stem, run_tag=step3_tag
		)
		if SKIP_IF_STEP3_DONE and _should_skip_step3_done(
			done3_path, run_tag=step3_tag
		):
			_p(f'  -> Step3 done exists (skip): {done3_path.name}')
			continue

		_p(f'  evt={inp.evt_path.name}')
		_p(f'  active_ch={inp.active_ch_path.name}')
		_p(f'  txt={inp.event_txt_path.name}')
		_p(f'  missing={inp.missing_path.name if inp.missing_path else "(none)"}')

		meta = _read_event_txt_meta(inp.event_txt_path)
		_p(
			f'  meta month={meta.event_month} lat={meta.lat:.5f} lon={meta.lon:.5f} origin={meta.origin_jst}'
		)

		evt_info = get_evt_info(inp.evt_path, scan_rate_blocks=1)
		t_start = evt_info.start_time
		t_end = evt_info.end_time_exclusive
		t0 = floor_minute(t_start)
		span_min_raw = ceil_minutes((t_end - t0).total_seconds())
		span_min = int(min(int(MAX_SPAN_MIN), int(span_min_raw)))
		_p(
			f'  evt window start={t_start} end={t_end} -> t0={t0} span_min={span_min} (raw={span_min_raw})'
		)

		log_path = inp.event_dir / f'{inp.evt_path.stem}_fill_to_48_log.csv'
		log_f, writer = _open_log_writer(log_path)
		_p(f'  log={log_path.name}')

		try:
			cont_dir = inp.event_dir / OUT_SUBDIR
			inv = _scan_continuous_inventory(cont_dir)

			s_active = _stations_in_active_ch(inp.active_ch_path)
			s_cont_0101 = inv.present_by_net.get(
				normalize_network_code(FILL_NETWORK_CODE), set()
			)

			missing_pairs = (
				read_missing_pairs(
					inp.missing_path,
					normalize_station=normalize_code,
					normalize_network=normalize_network_code,
				)
				if inp.missing_path is not None
				else []
			)
			s_missing_ok = _missing_recovered_set(missing_pairs, inv.present_by_net)

			base = set(s_active) | set(s_cont_0101) | set(s_missing_ok)
			n_base = len(base)
			n_need_base = int(MIN_STATIONS) - int(n_base)

			_p(
				f'  [base] active={len(s_active)} cont0101={len(s_cont_0101)} missing_ok={len(s_missing_ok)} -> base={n_base} need={n_need_base}'
			)

			if n_need_base <= 0:
				writer.writerow(
					{
						'event_dir': str(inp.event_dir),
						'evt_file': inp.evt_path.name,
						'event_month': meta.event_month,
						'event_lat': meta.lat,
						'event_lon': meta.lon,
						't0_jst': f'{t0:%Y-%m-%d %H:%M:%S}',
						'span_min': span_min,
						'n_before': n_base,
						'n_need': 0,
						'n_selected': 0,
						'n_after': n_base,
						'network_code': FILL_NETWORK_CODE,
						'select_used': True,
						'status': 'already_satisfied',
						'cnt_file': '',
						'ch_file': '',
						'message': f'base(active+cont0101+missing_ok)={n_base}',
					}
				)
				_write_step3_done(
					done3_path,
					{
						'evt_file': inp.evt_path.name,
						'evt_stem': inp.evt_path.stem,
						'run_tag': step3_tag,
						'status': 'already_satisfied',
						't0_jst': f'{t0:%Y-%m-%d %H:%M:%S}',
						'span_min': span_min,
						'n_after': n_base,
					},
				)
				continue

			s_near200 = _near200_from_continuous(
				event_lat=meta.lat,
				event_lon=meta.lon,
				base=base,
				coord_by_station=inv.coord_by_station,
			)
			pre = set(base) | set(s_near200)
			n_pre = len(pre)
			n_need_pre = int(MIN_STATIONS) - int(n_pre)
			_p(
				f'  [near{int(NEAR_KM)}] near={len(s_near200)} -> pre={n_pre} need={n_need_pre}'
			)

			if n_need_pre <= 0:
				writer.writerow(
					{
						'event_dir': str(inp.event_dir),
						'evt_file': inp.evt_path.name,
						'event_month': meta.event_month,
						'event_lat': meta.lat,
						'event_lon': meta.lon,
						't0_jst': f'{t0:%Y-%m-%d %H:%M:%S}',
						'span_min': span_min,
						'n_before': n_base,
						'n_need': 0,
						'n_selected': 0,
						'n_after': n_pre,
						'network_code': FILL_NETWORK_CODE,
						'select_used': True,
						'status': 'satisfied_by_near200',
						'cnt_file': '',
						'ch_file': '',
						'message': f'base={n_base} near{int(NEAR_KM)}={len(s_near200)}',
					}
				)
				_write_step3_done(
					done3_path,
					{
						'evt_file': inp.evt_path.name,
						'evt_stem': inp.evt_path.stem,
						'run_tag': step3_tag,
						'status': 'satisfied_by_near200',
						't0_jst': f'{t0:%Y-%m-%d %H:%M:%S}',
						'span_min': span_min,
						'n_after': n_pre,
					},
				)
				continue

			_p('  [cands] presence=1 candidates (0101) ...')
			cands_all = _presence_station_candidates_0101(
				pdb, event_month=meta.event_month
			)
			_p(f'    candidates total={len(cands_all)}')

			cands_rest = [s for s in cands_all if normalize_code(s) not in pre]
			_p(f'    after removing pre={len(cands_rest)}')

			selected, selected_with_dist = _select_nearest_0101(
				event_lat=meta.lat,
				event_lon=meta.lon,
				candidate_stations=cands_rest,
				station_geo_0101=station_geo_0101,
				n_need=n_need_pre,
			)
			_p(f'  [select0101] selected={len(selected)} (need={n_need_pre})')

			if selected_with_dist:
				head = selected_with_dist[
					: int(min(DEBUG_PRINT_SELECTED_HEAD, len(selected_with_dist)))
				]
				for sta, dkm in head:
					_p(f'    + {sta}\t{dkm:.3f} km')
				if len(selected_with_dist) > len(head):
					_p(f'    ... ({len(selected_with_dist) - len(head)} more)')

			added_list_path = (
				inp.event_dir / f'{inp.evt_path.stem}_fill_to_48_added_stations.txt'
			)
			shortage_path = (
				inp.event_dir / f'{inp.evt_path.stem}_fill_to_48_shortage.txt'
			)

			added_list_path.write_text(
				'\n'.join([f'{sta}\t{dkm:.3f}' for sta, dkm in selected_with_dist])
				+ ('\n' if selected_with_dist else ''),
				encoding='utf-8',
			)

			if not selected:
				shortage_path.write_text(
					f'event={inp.event_dir.name}\n'
					f'base={n_base}\n'
					f'near{int(NEAR_KM)}={len(s_near200)}\n'
					f'pre={n_pre}\n'
					f'n_need_pre={n_need_pre}\n'
					f'n_selected=0\n'
					f'reason=no_candidates_after_presence_and_geo_filter\n',
					encoding='utf-8',
				)
				writer.writerow(
					{
						'event_dir': str(inp.event_dir),
						'evt_file': inp.evt_path.name,
						'event_month': meta.event_month,
						'event_lat': meta.lat,
						'event_lon': meta.lon,
						't0_jst': f'{t0:%Y-%m-%d %H:%M:%S}',
						'span_min': span_min,
						'n_before': n_base,
						'n_need': n_need_pre,
						'n_selected': 0,
						'n_after': n_pre,
						'network_code': FILL_NETWORK_CODE,
						'select_used': True,
						'status': 'shortage',
						'cnt_file': '',
						'ch_file': '',
						'message': f'base={n_base} near{int(NEAR_KM)}={len(s_near200)} no candidates',
					}
				)
				_write_step3_done(
					done3_path,
					{
						'evt_file': inp.evt_path.name,
						'evt_stem': inp.evt_path.stem,
						'run_tag': step3_tag,
						'status': 'shortage',
						't0_jst': f'{t0:%Y-%m-%d %H:%M:%S}',
						'span_min': span_min,
						'n_after': n_pre,
						'n_selected': 0,
					},
				)
				continue

			outdir = inp.event_dir / OUT_SUBDIR
			outdir.mkdir(parents=True, exist_ok=True)

			stem = _name_stem(FILL_NETWORK_CODE, t0, sorted(selected), span_min)
			data_name = f'{stem}.cnt'
			ctable_name = f'{stem}.ch'
			cnt_path = outdir / data_name
			ch_path = outdir / ctable_name

			_p(f'  [out] {OUT_SUBDIR}/{data_name}')
			_p(f'  [out] {OUT_SUBDIR}/{ctable_name}')

			if SKIP_IF_EXISTS and cnt_path.exists() and ch_path.exists():
				_p('  -> exists (skip download)')
				client.select_stations(FILL_NETWORK_CODE)

				inv2 = _scan_continuous_inventory(cont_dir)
				s_cont_0101_2 = inv2.present_by_net.get(
					normalize_network_code(FILL_NETWORK_CODE), set()
				)
				s_missing_ok_2 = _missing_recovered_set(
					missing_pairs, inv2.present_by_net
				)
				base2 = set(s_active) | set(s_cont_0101_2) | set(s_missing_ok_2)

				s_near200_2 = (
					_near200_from_continuous(
						event_lat=meta.lat,
						event_lon=meta.lon,
						base=base2,
						coord_by_station=inv2.coord_by_station,
					)
					if len(base2) < MIN_STATIONS
					else set()
				)

				final2 = set(base2) | set(s_near200_2)
				n_after = len(final2)

				writer.writerow(
					{
						'event_dir': str(inp.event_dir),
						'evt_file': inp.evt_path.name,
						'event_month': meta.event_month,
						'event_lat': meta.lat,
						'event_lon': meta.lon,
						't0_jst': f'{t0:%Y-%m-%d %H:%M:%S}',
						'span_min': span_min,
						'n_before': n_base,
						'n_need': n_need_pre,
						'n_selected': len(selected),
						'n_after': n_after,
						'network_code': FILL_NETWORK_CODE,
						'select_used': True,
						'status': 'exists',
						'cnt_file': cnt_path.name,
						'ch_file': ch_path.name,
						'message': f'base={len(base2)} near{int(NEAR_KM)}={len(s_near200_2)}',
					}
				)
				_write_step3_done(
					done3_path,
					{
						'evt_file': inp.evt_path.name,
						'evt_stem': inp.evt_path.stem,
						'run_tag': step3_tag,
						'status': 'exists',
						't0_jst': f'{t0:%Y-%m-%d %H:%M:%S}',
						'span_min': span_min,
						'n_after': n_after,
						'n_selected': len(selected),
					},
				)
				continue

			_p('  [download] start (0101 select_stations enabled) ...')

			cnt_out = ch_out = None
			last_err: Exception | None = None
			for attempt, th in enumerate(thread_seq, 1):
				if attempt > 1:
					_p(
						f'  [warn] retry {attempt}/{len(thread_seq)} with threads={th}: {last_err!r}'
					)
				try:
					cnt_out, ch_out, _select_used = download_win_for_stations(
						client,
						stations=sorted(selected),
						when=t0,
						network_code=FILL_NETWORK_CODE,
						span_min=span_min,
						outdir=outdir,
						threads=int(th),
						cleanup=CLEANUP,
						clear_selection=False,
						skip_if_exists=False,
						use_select=True,
						data_name=data_name,
						ctable_name=ctable_name,
					)
					last_err = None
					break
				except Exception as e:
					last_err = e
				finally:
					client.select_stations(FILL_NETWORK_CODE)

			if last_err is not None or cnt_out is None or ch_out is None:
				raise (
					last_err
					if last_err is not None
					else RuntimeError('download failed (no outputs)')
				)

			_p(f'  [download] done: {Path(cnt_out).name}, {Path(ch_out).name}')

			inv2 = _scan_continuous_inventory(cont_dir)
			s_cont_0101_2 = inv2.present_by_net.get(
				normalize_network_code(FILL_NETWORK_CODE), set()
			)
			s_missing_ok_2 = _missing_recovered_set(missing_pairs, inv2.present_by_net)
			base2 = set(s_active) | set(s_cont_0101_2) | set(s_missing_ok_2)

			s_near200_2 = (
				_near200_from_continuous(
					event_lat=meta.lat,
					event_lon=meta.lon,
					base=base2,
					coord_by_station=inv2.coord_by_station,
				)
				if len(base2) < MIN_STATIONS
				else set()
			)

			final2 = set(base2) | set(s_near200_2)
			n_after = len(final2)

			status = 'downloaded'
			if n_after < MIN_STATIONS:
				_p('  -> partial: still below MIN_STATIONS')
				shortage_path.write_text(
					f'event={inp.event_dir.name}\n'
					f'base={len(base2)}\n'
					f'near{int(NEAR_KM)}={len(s_near200_2)}\n'
					f'final={n_after}\n'
					f'reason=still_below_min_after_download\n',
					encoding='utf-8',
				)
				status = 'partial'

			writer.writerow(
				{
					'event_dir': str(inp.event_dir),
					'evt_file': inp.evt_path.name,
					'event_month': meta.event_month,
					'event_lat': meta.lat,
					'event_lon': meta.lon,
					't0_jst': f'{t0:%Y-%m-%d %H:%M:%S}',
					'span_min': span_min,
					'n_before': n_base,
					'n_need': n_need_pre,
					'n_selected': len(selected),
					'n_after': n_after,
					'network_code': FILL_NETWORK_CODE,
					'select_used': True,
					'status': status,
					'cnt_file': Path(cnt_out).name,
					'ch_file': Path(ch_out).name,
					'message': f'base={len(base2)} near{int(NEAR_KM)}={len(s_near200_2)}',
				}
			)
			_write_step3_done(
				done3_path,
				{
					'evt_file': inp.evt_path.name,
					'evt_stem': inp.evt_path.stem,
					'run_tag': step3_tag,
					'status': status,
					't0_jst': f'{t0:%Y-%m-%d %H:%M:%S}',
					'span_min': span_min,
					'n_after': n_after,
					'n_selected': len(selected),
					'retry_threads': thread_seq,
					'date_filter': {
						'date_min': DATE_MIN,
						'date_max': DATE_MAX,
					},
				},
			)

		finally:
			log_f.close()

	_p('\n[done]')


if __name__ == '__main__':
	main()

# %%
