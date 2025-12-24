# %%

from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc
from scipy.signal import resample_poly

# ====== Parameters (edit here) ======
data_dir = Path('/home/dcuser/daseventnet/data/silixa/raw_78B_npy/')
out_zarr = Path('/home/dcuser/daseventnet/data/silixa/raw_78B_block_ds10.zarr')

up = 1
down = 10
window = ('kaiser', 8.6)

dtype_out = np.float32

# Chunking for block layout: (B, C, Tb)
# If you want 1-block random access, set chunk_b=1 (more chunk files).
# If you want fewer chunk files & faster sequential reads, set chunk_b=8~64.
chunk_b = 16

compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)

print_every_blocks = 200
# ====================================

files = sorted(data_dir.glob('*.npy'))
n_blocks = len(files)
if n_blocks == 0:
	raise ValueError(f'No .npy files found in: {data_dir}')

x0 = np.load(files[0], mmap_mode='r')
if x0.ndim != 2:
	raise ValueError(f'Expected 2D array, got ndim={x0.ndim} in {files[0]}')
n_ch, n_samp_in = x0.shape

if n_samp_in % down != 0:
	raise ValueError(f'n_samp_in={n_samp_in} is not divisible by down={down}')
n_samp_out = n_samp_in // down

est_gib = (n_blocks * n_ch * n_samp_out * np.dtype(dtype_out).itemsize) / (1024**3)
print(f'Found blocks(files): {n_blocks}')
print(f'Input block shape: ({n_ch}, {n_samp_in}), input dtype: {x0.dtype}')
print(f'Output block shape: ({n_ch}, {n_samp_out}), output dtype: {dtype_out}')
print(f'Zarr layout: (B, C, Tb) = ({n_blocks}, {n_ch}, {n_samp_out})')
print(f'Estimated uncompressed size: {est_gib:.2f} GiB')

if out_zarr.exists():
	root = zarr.open_group(str(out_zarr), mode='a')
	if 'block' not in root or 'done' not in root:
		raise ValueError(f'Existing store missing required datasets: {out_zarr}')
	arr = root['block']
	done = root['done']

	if arr.shape != (n_blocks, n_ch, n_samp_out):
		raise ValueError(f"Existing 'block' shape mismatch: {arr.shape}")
	if done.shape != (n_blocks,):
		raise ValueError(f"Existing 'done' shape mismatch: {done.shape}")

	if 'file_name' in root:
		stored_names = root['file_name'][:]
		current_names = np.array(
			[f.name.encode('utf-8') for f in files], dtype=stored_names.dtype
		)
		if stored_names.shape != current_names.shape or np.any(
			stored_names != current_names
		):
			raise ValueError(
				'file list mismatch between existing Zarr and current directory listing'
			)

	print('Resuming existing Zarr store.')
else:
	root = zarr.open_group(str(out_zarr), mode='w')

	arr = root.create_dataset(
		'block',
		shape=(n_blocks, n_ch, n_samp_out),
		chunks=(chunk_b, n_ch, n_samp_out),
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

	root.attrs.update(
		{
			'source_dir': str(data_dir),
			'layout': 'B,C,Tb',
			'n_blocks': int(n_blocks),
			'n_channels': int(n_ch),
			'n_samples_in_per_block': int(n_samp_in),
			'n_samples_out_per_block': int(n_samp_out),
			'dtype_out': str(np.dtype(dtype_out)),
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
			'note': 'Each .npy file becomes one block along axis=0 after downsampling.',
		}
	)

	print('Created new Zarr store.')

blocks_total = (n_blocks + chunk_b - 1) // chunk_b
for b in range(blocks_total):
	start = b * chunk_b
	end = min(start + chunk_b, n_blocks)

	if np.all(done[start:end]):
		continue

	batch = np.empty((end - start, n_ch, n_samp_out), dtype=dtype_out)

	for i, fp in enumerate(files[start:end]):
		x = np.load(fp, mmap_mode='r')
		if x.shape != (n_ch, n_samp_in):
			raise ValueError(
				f'Shape mismatch in {fp}: {x.shape} vs expected {(n_ch, n_samp_in)}'
			)

		x32 = np.asarray(x, dtype=np.float32)
		y = resample_poly(x32, up=up, down=down, axis=1, window=window)

		if y.shape != (n_ch, n_samp_out):
			raise ValueError(f'Unexpected resample output shape in {fp}: {y.shape}')

		batch[i] = y.astype(dtype_out, copy=False)

	arr[start:end, :, :] = batch
	done[start:end] = True

	if (b % print_every_blocks) == 0 or (end == n_blocks):
		n_done = int(done[:].sum())
		print(
			f'Batch {b + 1}/{blocks_total}: wrote blocks [{start}:{end})  done={n_done}/{n_blocks}'
		)

print('All done.')
print(f'Zarr store: {out_zarr}')
