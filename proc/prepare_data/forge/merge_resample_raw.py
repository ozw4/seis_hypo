# %%
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc
from scipy.signal import resample_poly

# ====== Parameters (edit here) ======
data_dir = Path('/home/dcuser/daseventnet/data/silixa/raw_78B_npy/')
out_zarr = Path('/home/dcuser/daseventnet/data/silixa/raw_78B_block_ds10.zarr')

overwrite_zarr = True  # True: delete existing out_zarr directory

fs_in_hz = 1000.0
up = 1
down = 10
window = ('kaiser', 8.6)

dtype_out = np.float32

# chunk for (B, C, Tb_out). For sequential inference (all ch), 8 or 16 is a safe start.
chunk_b = 16

compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)

# Segment settings
tol_ms = 2  # time gap tolerance in milliseconds
force_break_on_short = True  # if previous file is short, break segment at boundary

print_every = 5
# ====================================


def parse_start_ms_from_filename(fp: Path) -> int:
	# Expected: ..._UTC_YYYYMMDD_HHMMSS(.mmm)_...
	parts = fp.stem.split('_')
	if 'UTC' not in parts:
		raise ValueError(f"Cannot find 'UTC' token in filename: {fp.name}")
	i = parts.index('UTC')
	if i + 2 >= len(parts):
		raise ValueError(f"Cannot parse date/time after 'UTC' in filename: {fp.name}")

	date_s = parts[i + 1]  # YYYYMMDD
	time_s = parts[i + 2]  # HHMMSS or HHMMSS.mmm

	if len(date_s) != 8:
		raise ValueError(f"Bad date token '{date_s}' in filename: {fp.name}")

	if '.' in time_s:
		hhmmss, frac = time_s.split('.', 1)
		ms = int((frac + '000')[:3])
	else:
		hhmmss = time_s
		ms = 0

	if len(hhmmss) != 6:
		raise ValueError(f"Bad time token '{time_s}' in filename: {fp.name}")

	year = int(date_s[0:4])
	month = int(date_s[4:6])
	day = int(date_s[6:8])
	hour = int(hhmmss[0:2])
	minute = int(hhmmss[2:4])
	second = int(hhmmss[4:6])

	dt = datetime(
		year, month, day, hour, minute, second, ms * 1000, tzinfo=timezone.utc
	)
	return int(dt.timestamp() * 1000)


def ceil_div(a: int, b: int) -> int:
	return (a + b - 1) // b


# ---- Gather files ----
files = sorted(data_dir.glob('*.npy'))
n_blocks = len(files)
if n_blocks == 0:
	raise ValueError(f'No .npy files found in: {data_dir}')

# ---- (Re)create Zarr ----
if out_zarr.exists():
	if not overwrite_zarr:
		raise ValueError(f'Output Zarr already exists: {out_zarr}')
	if out_zarr.is_file():
		raise ValueError(f'Output path is a file, not a directory: {out_zarr}')
	print(f'[WARN] Deleting existing Zarr directory: {out_zarr}')
	shutil.rmtree(out_zarr)

# ---- Prepass: channel count, sample lengths, start times ----
start_ms = np.empty(n_blocks, dtype=np.int64)
valid_in = np.empty(n_blocks, dtype=np.int32)

x0 = np.load(files[0], mmap_mode='r')
if x0.ndim != 2:
	raise ValueError(f'Expected 2D array, got ndim={x0.ndim} in {files[0]}')
n_ch = int(x0.shape[0])

for i, fp in enumerate(files):
	start_ms[i] = parse_start_ms_from_filename(fp)
	x = np.load(fp, mmap_mode='r')
	if x.ndim != 2:
		raise ValueError(f'Expected 2D array, got ndim={x.ndim} in {fp}')
	if int(x.shape[0]) != n_ch:
		raise ValueError(f'Channel mismatch in {fp}: {x.shape[0]} vs expected {n_ch}')
	valid_in[i] = int(x.shape[1])

# Nominal length = mode of lengths
uniq, cnt = np.unique(valid_in, return_counts=True)
nom_in = int(uniq[int(np.argmax(cnt))])
if nom_in % down != 0:
	raise ValueError(f'Nominal length {nom_in} is not divisible by down={down}')
nom_out = nom_in // down

print(f'Found files(blocks): {n_blocks}')
print(f'Channels: {n_ch}')
print(f'Nominal in-samples per file: {nom_in} (mode)')
print(f'Nominal out-samples per block: {nom_out} (down={down})')

# Segment id
expected_ms = int(round((nom_in / fs_in_hz) * 1000.0))
segment_id = np.empty(n_blocks, dtype=np.int32)
seg = 0
segment_id[0] = seg
for i in range(1, n_blocks):
	prev_short = force_break_on_short and (int(valid_in[i - 1]) < nom_in)
	dt_ms = int(start_ms[i] - start_ms[i - 1])
	gap_ms = dt_ms - expected_ms
	if prev_short or (abs(gap_ms) > tol_ms):
		seg += 1
	segment_id[i] = seg
n_segments = int(segment_id.max()) + 1
print(f'Segments: {n_segments}')

# ---- Create Zarr datasets ----
root = zarr.open_group(str(out_zarr), mode='w')

arr = root.create_dataset(
	'block',
	shape=(n_blocks, n_ch, nom_out),
	chunks=(chunk_b, n_ch, nom_out),
	dtype=dtype_out,
	compressor=compressor,
	overwrite=True,
)

done = root.create_dataset(
	'done',
	shape=(n_blocks,),
	chunks=(min(4096, n_blocks),),
	dtype=np.bool_,
	overwrite=True,
)
done[:] = False

names = [f.name for f in files]
maxlen = max(len(n.encode('utf-8')) for n in names)
names_b = np.array([n.encode('utf-8') for n in names], dtype=f'S{maxlen}')
root.create_dataset(
	'file_name',
	data=names_b,
	chunks=(min(4096, n_blocks),),
	overwrite=True,
)

root.create_dataset(
	'starttime_utc_ms',
	data=start_ms,
	chunks=(min(4096, n_blocks),),
	dtype=np.int64,
	overwrite=True,
)

root.create_dataset(
	'valid_in_samples',
	data=valid_in,
	chunks=(min(4096, n_blocks),),
	dtype=np.int32,
	overwrite=True,
)

valid_out_init = np.minimum(
	np.array([ceil_div(int(v), down) for v in valid_in], dtype=np.int32), nom_out
)
root.create_dataset(
	'valid_out_samples',
	data=valid_out_init,
	chunks=(min(4096, n_blocks),),
	dtype=np.int32,
	overwrite=True,
)

root.create_dataset(
	'segment_id',
	data=segment_id,
	chunks=(min(4096, n_blocks),),
	dtype=np.int32,
	overwrite=True,
)

root.attrs.update(
	{
		'source_dir': str(data_dir),
		'layout': 'B,C,Tb',
		'n_blocks': int(n_blocks),
		'n_channels': int(n_ch),
		'fs_in_hz': float(fs_in_hz),
		'fs_out_hz': float(fs_in_hz * up / down),
		'nominal_in_samples': int(nom_in),
		'nominal_out_samples': int(nom_out),
		'dtype_out': str(np.dtype(dtype_out)),
		'segment_tol_ms': int(tol_ms),
		'force_break_on_short': bool(force_break_on_short),
		'resample_poly': {
			'up': int(up),
			'down': int(down),
			'axis': 1,
			'window': [str(window[0]), float(window[1])],
		},
		'compressor': {
			'name': 'Blosc',
			'cname': 'zstd',
			'clevel': int(compressor.clevel),
			'shuffle': int(compressor.shuffle),
		},
		'note': "Short files are zero-padded in 'block'. Use valid_* and segment_id to avoid crossing gaps.",
	}
)

# ---- Convert ----
for i, fp in enumerate(files):
	x = np.load(fp, mmap_mode='r')
	n_in = int(x.shape[1])
	if n_in <= 0:
		raise ValueError(f'Invalid length in {fp}: {x.shape}')
	if n_in > nom_in:
		raise ValueError(
			f'Too-long file in {fp}: {x.shape} vs nominal {(n_ch, nom_in)}'
		)

	x32 = np.asarray(x, dtype=np.float32)
	y = resample_poly(x32, up=up, down=down, axis=1, window=window)
	y_len = int(y.shape[1])
	if y_len > nom_out:
		raise ValueError(
			f'Resample output longer than nominal in {fp}: {y.shape} vs nominal {(n_ch, nom_out)}'
		)

	out_block = np.zeros((n_ch, nom_out), dtype=dtype_out)
	out_block[:, :y_len] = y.astype(dtype_out, copy=False)

	arr[i, :, :] = out_block
	done[i] = True

	if (i % print_every) == 0 or (i + 1 == n_blocks):
		print(f'Wrote {i + 1}/{n_blocks}')

print('All done.')
print(f'Zarr store: {out_zarr}')
