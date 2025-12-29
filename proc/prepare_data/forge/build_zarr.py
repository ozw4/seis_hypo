# %%
import calendar
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import zarr
from nptdms import TdmsFile
from numcodecs import Blosc, VLenUTF8
from scipy.signal import butter, filtfilt, resample_poly
from tqdm import tqdm

logging.getLogger('nptdms').setLevel(logging.ERROR)
logging.getLogger('nptdms.tdms_segment').setLevel(logging.ERROR)

JST = timezone(timedelta(hours=9))

# ====== Parameters (edit here) ======
data_dir = Path('/home/dcuser/daseventnet/data/silixa/raw_tdms/')
out_zarr = Path('/home/dcuser/daseventnet/data/silixa/forge_dfit_block_78AB_250Hz.zarr')

fs_in_hz = 4000.0
fs_out_hz_target = 250.0
nominal_in_samples = 60000  # 15 sec * 4000 Hz
min_valid_samples = 60000  # 60000ない奴は処理しない

window = ('kaiser', 8.6)
dtype_out = np.float32

bp_low_hz = 25.0
bp_high_hz = 110.0
bp_order = 4

SLICE_A = slice(69, 1079)
SLICE_B = slice(1195, 2401)
n_ch_a = SLICE_A.stop - SLICE_A.start
n_ch_b = SLICE_B.stop - SLICE_B.start
n_ch_out = n_ch_a + n_ch_b

chunk_b = 1
chunk_t_out = 1024
compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)

tol_ms = 2
force_break_on_short = True

scan_interval_sec = 2.0
stable_sec = 8.0
min_file_age_sec = 2.0
# ====================================


def _validate_downsample_params() -> tuple[int, float]:
	if fs_out_hz_target <= 0:
		raise ValueError(f'fs_out_hz_target must be > 0: {fs_out_hz_target}')
	if fs_in_hz <= 0:
		raise ValueError(f'fs_in_hz must be > 0: {fs_in_hz}')

	down = int(round(fs_in_hz / fs_out_hz_target))
	if down <= 0:
		raise ValueError(f'Computed down factor invalid: down={down}')

	fs_out = fs_in_hz / down
	if abs(fs_out - fs_out_hz_target) > 1e-9:
		raise ValueError(
			f'fs_in_hz({fs_in_hz}) not divisible to reach fs_out_hz_target({fs_out_hz_target}). '
			f'Got fs_out={fs_out} with down={down}'
		)
	return down, fs_out


DOWN_FACTOR, fs_out_hz = _validate_downsample_params()

if nominal_in_samples % DOWN_FACTOR != 0:
	raise ValueError(
		f'nominal_in_samples must be divisible by DOWN_FACTOR. '
		f'nominal_in_samples={nominal_in_samples}, DOWN_FACTOR={DOWN_FACTOR}'
	)

nominal_out_samples = nominal_in_samples // DOWN_FACTOR

if not (0.0 < bp_low_hz < bp_high_hz < (0.5 * fs_out_hz)):
	raise ValueError(
		f'Bad bandpass for fs_out_hz={fs_out_hz}. Need 0 < low < high < Nyquist. '
		f'Got low={bp_low_hz}, high={bp_high_hz}'
	)


def bandpass_filter(
	data: np.ndarray, fs: float, lowcut: float, highcut: float, order: int
) -> np.ndarray:
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	return filtfilt(b, a, data, axis=1)


def parse_start_ms_from_filename(fp: Path) -> int:
	parts = fp.stem.split('_')
	if 'UTC' not in parts:
		raise ValueError(f"Cannot find 'UTC' token in filename: {fp.name}")
	i = parts.index('UTC')
	if i + 2 >= len(parts):
		raise ValueError(f"Cannot parse date/time after 'UTC' in filename: {fp.name}")

	date_s = parts[i + 1]
	time_s = parts[i + 2]

	if len(date_s) != 8 or (not date_s.isdigit()):
		raise ValueError(f"Bad date token '{date_s}' in filename: {fp.name}")

	if '.' in time_s:
		hhmmss, frac = time_s.split('.', 1)
		ms = int((frac + '000')[:3])
	else:
		hhmmss = time_s
		ms = 0

	if len(hhmmss) != 6 or (not hhmmss.isdigit()):
		raise ValueError(f"Bad time token '{time_s}' in filename: {fp.name}")

	year = int(date_s[0:4])
	month = int(date_s[4:6])
	day = int(date_s[6:8])
	hour = int(hhmmss[0:2])
	minute = int(hhmmss[2:4])
	second = int(hhmmss[4:6])

	dt = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
	sec = calendar.timegm(dt.utctimetuple())
	return int(sec * 1000 + ms)


def _ms_to_strings(utc_ms: int) -> tuple[str, str]:
	dt_utc = datetime.fromtimestamp(utc_ms / 1000.0, tz=timezone.utc)
	dt_jst = dt_utc.astimezone(JST)
	return dt_utc.strftime('%Y-%m-%d %H:%M:%S UTC'), dt_jst.strftime(
		'%Y-%m-%d %H:%M:%S JST'
	)


def _process_one_side(x: np.ndarray) -> np.ndarray:
	y = resample_poly(x, up=1, down=int(DOWN_FACTOR), axis=1, window=window).astype(
		np.float32, copy=False
	)
	y = np.asarray(
		bandpass_filter(
			y,
			fs=float(fs_out_hz),
			lowcut=float(bp_low_hz),
			highcut=float(bp_high_hz),
			order=int(bp_order),
		),
		dtype=np.float32,
	)
	med = np.median(y, axis=0).astype(np.float32, copy=False)
	return (y - med).astype(np.float32, copy=False)


processed_names: set[str] = set()


def _load_processed_names(root: zarr.hierarchy.Group) -> set[str]:
	if 'file_name' not in root:
		return set()
	ds = root['file_name']
	if ds.shape[0] == 0:
		return set()
	return set(str(x) for x in ds[:])


def open_or_create_zarr() -> zarr.hierarchy.Group:
	global processed_names

	if out_zarr.exists():
		root = zarr.open_group(str(out_zarr), mode='a')
		a = root.attrs
		if int(a.get('nominal_in_samples')) != int(nominal_in_samples):
			raise ValueError(
				'nominal_in_samples mismatch between existing Zarr and this script settings'
			)
		if int(a.get('nominal_out_samples')) != int(nominal_out_samples):
			raise ValueError(
				'nominal_out_samples mismatch between existing Zarr and this script settings'
			)
		if int(a.get('down_factor')) != int(DOWN_FACTOR):
			raise ValueError(
				'down_factor mismatch between existing Zarr and this script settings'
			)
		if int(a.get('n_channels')) != int(n_ch_out):
			raise ValueError(
				'n_channels mismatch between existing Zarr and this script settings'
			)

		processed_names = _load_processed_names(root)
		return root

	root = zarr.open_group(str(out_zarr), mode='w')

	chunk_t = int(min(int(chunk_t_out), int(nominal_out_samples)))
	if chunk_t <= 0:
		raise ValueError(
			f'chunk_t_out must be > 0: chunk_t_out={chunk_t_out}, nominal_out_samples={nominal_out_samples}'
		)

	root.create_dataset(
		'block',
		shape=(0, n_ch_out, nominal_out_samples),
		chunks=(int(chunk_b), int(n_ch_out), int(chunk_t)),
		dtype=dtype_out,
		compressor=compressor,
		overwrite=True,
	)

	root.create_dataset(
		'done', shape=(0,), chunks=(4096,), dtype=np.bool_, overwrite=True
	)
	root.create_dataset(
		'file_name',
		shape=(0,),
		chunks=(4096,),
		dtype=object,
		object_codec=VLenUTF8(),
		overwrite=True,
	)
	root.create_dataset(
		'starttime_utc_ms', shape=(0,), chunks=(4096,), dtype=np.int64, overwrite=True
	)
	root.create_dataset(
		'valid_in_samples', shape=(0,), chunks=(4096,), dtype=np.int32, overwrite=True
	)
	root.create_dataset(
		'valid_out_samples', shape=(0,), chunks=(4096,), dtype=np.int32, overwrite=True
	)
	root.create_dataset(
		'segment_id', shape=(0,), chunks=(4096,), dtype=np.int32, overwrite=True
	)

	root.attrs.update(
		{
			'source_dir': str(data_dir),
			'layout': 'B,C,Tb',
			'n_channels': int(n_ch_out),
			'fs_in_hz': float(fs_in_hz),
			'fs_out_hz': float(fs_out_hz),
			'down_factor': int(DOWN_FACTOR),
			'nominal_in_samples': int(nominal_in_samples),
			'nominal_out_samples': int(nominal_out_samples),
			'chunk_b': int(chunk_b),
			'chunk_t_out': int(chunk_t),
			'slices': {
				'78A': [int(SLICE_A.start), int(SLICE_A.stop)],
				'78B': [int(SLICE_B.start), int(SLICE_B.stop)],
			},
			'bandpass_hz': [float(bp_low_hz), float(bp_high_hz)],
			'segment_tol_ms': int(tol_ms),
			'force_break_on_short': bool(force_break_on_short),
			'note': 'Only TDMS with exactly nominal_in_samples are ingested; others are skipped.',
		}
	)

	processed_names = set()
	return root


def _read_matrix_from_channels(
	chans, slc: slice, fp_name: str
) -> tuple[np.ndarray, int]:
	rows: list[np.ndarray] = []
	n_in: int | None = None

	for ch in chans[slc]:
		x = np.asarray(ch.data, dtype=np.float32)
		n = int(x.shape[0])
		if n_in is None:
			n_in = n
		elif n != n_in:
			raise ValueError(f'Sample length mismatch within slice in {fp_name}')
		rows.append(x)

	if n_in is None:
		raise ValueError(f'No channels read for slice in {fp_name}')

	return np.vstack(rows), int(n_in)


def _append_block(root: zarr.hierarchy.Group, fp: Path, out_block: np.ndarray) -> None:
	arr = root['block']
	done = root['done']
	fname = root['file_name']
	start = root['starttime_utc_ms']
	vin = root['valid_in_samples']
	vout = root['valid_out_samples']
	segid = root['segment_id']

	i = int(arr.shape[0])
	new_len = i + 1

	arr.resize(new_len, n_ch_out, nominal_out_samples)
	done.resize(new_len)
	fname.resize(new_len)
	start.resize(new_len)
	vin.resize(new_len)
	vout.resize(new_len)
	segid.resize(new_len)

	this_start_ms = parse_start_ms_from_filename(fp)
	expected_ms = int(round((nominal_in_samples / fs_in_hz) * 1000.0))

	if i == 0:
		this_seg = 0
	else:
		prev_start_ms = int(start[i - 1])
		prev_vin = int(vin[i - 1])
		prev_seg = int(segid[i - 1])
		prev_short = bool(force_break_on_short) and (prev_vin < nominal_in_samples)
		dt_ms = int(this_start_ms - prev_start_ms)
		gap_ms = int(dt_ms - expected_ms)
		this_seg = prev_seg + 1 if (prev_short or (abs(gap_ms) > tol_ms)) else prev_seg

	arr[i, :, :] = out_block
	done[i] = True
	fname[i] = fp.name
	start[i] = this_start_ms
	vin[i] = int(nominal_in_samples)
	vout[i] = int(nominal_out_samples)
	segid[i] = int(this_seg)


def try_process_one_file(
	root: zarr.hierarchy.Group, fp: Path, stage_pb: tqdm | None
) -> bool:
	if fp.name in processed_names:
		return False

	# ダウンロード途中で壊れてる/読めない可能性は現実にあるので、ここだけは警告付きスキップを許可
	try:
		tdms = TdmsFile.read(fp)
	except (OSError, EOFError, ValueError) as e:
		logging.warning(
			f'Skip unreadable TDMS (likely downloading): {fp.name} ({type(e).__name__}: {e})'
		)
		return False

	group = tdms['Measurement']
	chans = group.channels()
	if len(chans) < SLICE_B.stop:
		raise ValueError(
			f'Not enough channels in {fp.name}: total={len(chans)}, need >= {SLICE_B.stop}'
		)

	if stage_pb is not None:
		stage_pb.set_postfix_str('read A')
		stage_pb.update(1)

	seis_a, n_in_a = _read_matrix_from_channels(chans, SLICE_A, fp.name)
	if n_in_a != int(min_valid_samples):
		logging.info(f'Skip (not {min_valid_samples} samples): {fp.name} n_in={n_in_a}')
		return False

	if stage_pb is not None:
		stage_pb.set_postfix_str('proc A')
		stage_pb.update(1)

	den_a = _process_one_side(seis_a)
	del seis_a

	if int(den_a.shape[1]) != int(nominal_out_samples):
		raise ValueError(
			f'Unexpected out length in A for {fp.name}: y_len={den_a.shape[1]}, expected={nominal_out_samples}'
		)

	out_block = np.empty((n_ch_out, nominal_out_samples), dtype=dtype_out)
	out_block[0:n_ch_a, :] = den_a.astype(dtype_out, copy=False)
	del den_a

	if stage_pb is not None:
		stage_pb.set_postfix_str('read+proc B')
		stage_pb.update(1)

	seis_b, n_in_b = _read_matrix_from_channels(chans, SLICE_B, fp.name)
	if n_in_b != int(n_in_a):
		raise ValueError(
			f'Sample length mismatch between A/B in {fp.name}: A={n_in_a}, B={n_in_b}'
		)

	den_b = _process_one_side(seis_b)
	del seis_b

	if int(den_b.shape[1]) != int(nominal_out_samples):
		raise ValueError(
			f'Unexpected out length in B for {fp.name}: y_len={den_b.shape[1]}, expected={nominal_out_samples}'
		)

	out_block[n_ch_a:n_ch_out, :] = den_b.astype(dtype_out, copy=False)
	del den_b

	if stage_pb is not None:
		stage_pb.set_postfix_str('write Zarr')
		stage_pb.update(1)

	_append_block(root, fp, out_block)
	processed_names.add(fp.name)
	return True


@dataclass
class FileState:
	size: int
	last_change_ts: float


_file_states: dict[str, FileState] = {}


def is_file_stable(fp: Path, now_ts: float) -> bool:
	st = fp.stat()
	key = str(fp)
	prev = _file_states.get(key)

	if prev is None:
		_file_states[key] = FileState(
			size=int(st.st_size), last_change_ts=float(now_ts)
		)
		return False

	if int(st.st_size) != int(prev.size):
		_file_states[key] = FileState(
			size=int(st.st_size), last_change_ts=float(now_ts)
		)
		return False

	if float(now_ts - prev.last_change_ts) < float(stable_sec):
		return False

	if float(now_ts - float(st.st_mtime)) < float(min_file_age_sec):
		return False

	return True


def ingest_loop() -> None:
	root = open_or_create_zarr()
	print(f'Zarr: {out_zarr}')
	print(f'Watching: {data_dir}')
	print(
		f'Rule: only process TDMS with n_in == {min_valid_samples}; NO deletion here.'
	)
	if len(processed_names) > 0:
		print(f'Already processed in Zarr: {len(processed_names)} files')

	while True:
		now_ts = datetime.now(timezone.utc).timestamp()
		tdms_files = sorted(data_dir.glob('*.tdms'))

		stable_files: list[Path] = []
		for fp in tdms_files:
			if fp.name in processed_names:
				continue
			if is_file_stable(fp, now_ts):
				stable_files.append(fp)

		if len(stable_files) == 0:
			time.sleep(float(scan_interval_sec))
			continue

		stable_files = sorted(stable_files, key=parse_start_ms_from_filename)

		files_pb = tqdm(
			total=len(stable_files), desc='Stable TDMS', unit='file', dynamic_ncols=True
		)
		for fp in stable_files:
			utc_ms = parse_start_ms_from_filename(fp)
			_, jst_s = _ms_to_strings(utc_ms)

			stage_pb = tqdm(
				total=4,
				desc=fp.name[-40:],
				unit='step',
				position=1,
				leave=False,
				dynamic_ncols=True,
				mininterval=0.1,
			)

			processed = try_process_one_file(root, fp, stage_pb=stage_pb)
			stage_pb.close()

			if processed:
				files_pb.set_postfix_str(f'processed | {jst_s}')
			else:
				files_pb.set_postfix_str(f'skipped | {jst_s}')

			files_pb.update(1)

		files_pb.close()
		time.sleep(float(scan_interval_sec))


# Run
ingest_loop()
