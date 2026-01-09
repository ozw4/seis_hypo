# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr

from common.time_util import utc_ms_to_iso

# ====== EDIT HERE ======
zarr_path = Path(
	'/home/dcuser/daseventnet/data/silixa/forge_dfit_block_78AB_250Hz.zarr'
)

# exact file name or substring (if exact not found)
target = 'FORGE_DFIT_UTC_20220417_105956.202.tdms'

# choose which occurrence if multiple exact matches (among done==True)
pick = 'last'  # "first" or "last"

# plot controls
imshow_v = None  # None -> percentile auto, else fixed float like 0.01
robust_pct = (2.0, 98.0)

# zoom window in seconds (None -> full)
zoom_sec0 = 5.0
zoom_sec1 = 10.0

# show a few traces
n_trace = 6
# =======================


def as_str(x) -> str:
	if isinstance(x, (bytes, np.bytes_)):
		return x.decode('utf-8')
	return str(x)


def auto_v_by_percentile(x: np.ndarray, pct: tuple[float, float]) -> float:
	lo, hi = np.percentile(x, [float(pct[0]), float(pct[1])])
	v = float(max(abs(lo), abs(hi)))
	return v if v > 0 else 1e-12


def heatmap(name: str, x: np.ndarray, fs: float, v_fixed: float | None) -> None:
	if x.size == 0 or x.shape[1] == 0:
		raise ValueError(f'Empty heatmap data: {name}')
	v = float(v_fixed) if v_fixed is not None else auto_v_by_percentile(x, robust_pct)

	C, T = x.shape
	extent = [0.0, T / fs, C, 0]
	plt.figure(figsize=(10, 5))
	plt.imshow(
		x,
		aspect='auto',
		cmap='seismic',
		vmin=-v,
		vmax=+v,
		extent=extent,
		origin='upper',
		interpolation='none',
	)
	plt.colorbar(label='amplitude')
	plt.title(name)
	plt.xlabel('time [s]')
	plt.ylabel('channel')


def plot_traces(name: str, x: np.ndarray, fs: float, n: int) -> None:
	if n <= 0:
		raise ValueError('n_trace must be > 0')
	C, T = x.shape
	idx = np.linspace(0, C - 1, num=min(int(n), C), dtype=int)
	t = np.arange(T, dtype=np.float32) / float(fs)

	plt.figure(figsize=(10, 4))
	for ch in idx:
		plt.plot(t, x[ch, :], label=f'ch{int(ch)}')
	plt.title(name)
	plt.xlabel('time [s]')
	plt.ylabel('amplitude')
	plt.legend(loc='upper right', fontsize=8)


def plot_common_mode(name: str, x: np.ndarray, fs: float) -> None:
	cm = np.median(x, axis=0).astype(np.float32, copy=False)
	t = np.arange(cm.size, dtype=np.float32) / float(fs)
	plt.figure(figsize=(10, 3))
	plt.plot(t, cm)
	plt.title(name)
	plt.xlabel('time [s]')
	plt.ylabel('median over channels')


def find_indices_by_filename(
	root: zarr.hierarchy.Group, target_name: str, only_done: bool
) -> np.ndarray:
	if 'file_name' not in root or 'done' not in root:
		raise ValueError("Zarr must contain 'file_name' and 'done' datasets.")

	ds_name = root['file_name']
	ds_done = root['done']
	n = int(ds_done.shape[0])
	if int(ds_name.shape[0]) != n:
		raise ValueError('file_name and done length mismatch.')

	target_name = str(target_name)
	chunk = int(ds_name.chunks[0]) if ds_name.chunks else 4096

	hits: list[int] = []
	for i0 in range(0, n, chunk):
		i1 = min(n, i0 + chunk)
		names = ds_name[i0:i1]
		done = np.asarray(ds_done[i0:i1], dtype=bool)
		for j, nm in enumerate(names):
			if as_str(nm) != target_name:
				continue
			if only_done and (not bool(done[j])):
				continue
			hits.append(i0 + j)

	return np.asarray(hits, dtype=np.int64)


def find_indices_by_substring(
	root: zarr.hierarchy.Group, sub: str, max_show: int = 50
) -> np.ndarray:
	ds_name = root['file_name']
	ds_done = root['done']
	n = int(ds_done.shape[0])
	chunk = int(ds_name.chunks[0]) if ds_name.chunks else 4096

	hits: list[int] = []
	sub = str(sub)
	for i0 in range(0, n, chunk):
		i1 = min(n, i0 + chunk)
		names = ds_name[i0:i1]
		for j, nm in enumerate(names):
			if sub in as_str(nm):
				hits.append(i0 + j)
				if len(hits) >= int(max_show):
					return np.asarray(hits, dtype=np.int64)
	return np.asarray(hits, dtype=np.int64)


# ---- Open Zarr ----
root = zarr.open_group(str(zarr_path), mode='r')
for required in ['block', 'done', 'file_name']:
	if required not in root:
		raise ValueError(f'Zarr missing dataset: {required}')

if 'fs_out_hz' not in root.attrs:
	raise ValueError(
		"Zarr attrs missing 'fs_out_hz' (needed for time axis in seconds)."
	)
fs_out = float(root.attrs['fs_out_hz'])

block = root['block']
done = np.asarray(root['done'][:], dtype=bool)
n_total = int(done.size)
n_done = int(done.sum())
print('=== Zarr summary ===')
print('zarr       :', zarr_path)
print('total      :', n_total)
print('done       :', n_done)
print('fs_out_hz  :', fs_out)
print()

if n_done == 0:
	raise ValueError('No completed blocks yet (done==True).')

if 'starttime_utc_ms' not in root:
	raise ValueError(
		'Zarr missing dataset: starttime_utc_ms (needed for first/last timestamps).'
	)
start_ms = np.asarray(root['starttime_utc_ms'][:], dtype=np.int64)

done_idx = np.flatnonzero(done)
first_i = int(done_idx[0])
last_i = int(done_idx[-1])

print('=== First / Last (append order among done==True) ===')
print('FIRST index:', first_i)
print('FIRST file :', as_str(root['file_name'][first_i]))
print('FIRST utc  :', utc_ms_to_iso(int(start_ms[first_i])))
print('LAST index :', last_i)
print('LAST file  :', as_str(root['file_name'][last_i]))
print('LAST utc   :', utc_ms_to_iso(int(start_ms[last_i])))
print()

start_done = start_ms[done]
i_earliest = int(done_idx[int(np.argmin(start_done))])
i_latest = int(done_idx[int(np.argmax(start_done))])

print('=== Earliest / Latest (starttime_utc_ms among done==True) ===')
print('EARLIEST index:', i_earliest)
print('EARLIEST file :', as_str(root['file_name'][i_earliest]))
print('EARLIEST utc  :', utc_ms_to_iso(int(start_ms[i_earliest])))
print('LATEST index  :', i_latest)
print('LATEST file   :', as_str(root['file_name'][i_latest]))
print('LATEST utc    :', utc_ms_to_iso(int(start_ms[i_latest])))
print()

# ---- Resolve A/B boundary if available ----
a_len = None
slices = root.attrs.get('slices', None)
if isinstance(slices, dict) and ('78A' in slices):
	a0, a1 = slices['78A']
	a_len = int(a1) - int(a0)

# ---- Find target index ----
idx = find_indices_by_filename(root, target, only_done=True)
if idx.size == 0:
	print(
		'No exact match among done==True. Showing substring candidates (done True/False).'
	)
	cand = find_indices_by_substring(root, target, max_show=50)
	for j in cand[:20].tolist():
		nm = as_str(root['file_name'][j])
		dn = bool(root['done'][j])
		print(f'  [i={j:6d}] done={dn} file={nm}')
	raise ValueError(
		'Pick an exact file name (or make target more specific) and rerun.'
	)

i = int(idx[0] if pick == 'first' else idx[-1])

# ---- Metadata ----
file_name = as_str(root['file_name'][i])
done_i = bool(root['done'][i])
segid = int(root['segment_id'][i]) if 'segment_id' in root else None
vin = int(root['valid_in_samples'][i]) if 'valid_in_samples' in root else None
vout = int(root['valid_out_samples'][i]) if 'valid_out_samples' in root else None

print('=== Selected block ===')
print('index       :', i)
print('file        :', file_name)
print('done        :', done_i)
print('utc         :', utc_ms_to_iso(int(start_ms[i])))
if segid is not None:
	print('segment_id  :', segid)
if vin is not None and vout is not None:
	print('valid_in/out:', vin, vout)
print()

# ---- Read block ----
x = np.asarray(block[i, :, :], dtype=np.float32)  # (C, T)
C, T = x.shape

absmean = float(np.mean(np.abs(x)))
rms = float(np.sqrt(np.mean(x * x)))
frac_zero = float(np.mean(x == 0.0))
rms_by_ch = np.sqrt(np.mean(x * x, axis=1))

print('=== Stats ===')
print('shape      :', (C, T))
print('absmean    :', absmean)
print('rms        :', rms)
print('frac_zero  :', frac_zero)
print(
	'rms_by_ch  : min/med/max =',
	float(rms_by_ch.min()),
	float(np.median(rms_by_ch)),
	float(rms_by_ch.max()),
)
print()

# ---- Zoom slicing ----
if zoom_sec0 is None or zoom_sec1 is None:
	t0, t1 = 0, T
else:
	t0 = int(round(float(zoom_sec0) * fs_out))
	t1 = int(round(float(zoom_sec1) * fs_out))
	t0 = max(0, min(t0, T - 1))
	t1 = max(t0 + 1, min(t1, T))
	print(f'[zoom] t0:t1={t0}:{t1} ({t0 / fs_out:.3f}-{t1 / fs_out:.3f}s)')

xz = x[:, t0:t1]

# ---- Visualize ----
heatmap(f'Zarr block[{i}] FULL  {file_name}', x, fs_out, imshow_v)
heatmap(f'Zarr block[{i}] ZOOM  {file_name}', xz, fs_out, imshow_v)

plot_traces(f'Zarr block[{i}] traces (ZOOM)', xz, fs_out, int(n_trace))
plot_common_mode(
	f'Zarr block[{i}] common-mode (median over channels, ZOOM)', xz, fs_out
)

# A/B split views (if boundary known)
if a_len is not None and 0 < a_len < C:
	heatmap(f'Zarr block[{i}] ZOOM A (0:{a_len})', xz[0:a_len, :], fs_out, imshow_v)
	heatmap(f'Zarr block[{i}] ZOOM B ({a_len}:{C})', xz[a_len:C, :], fs_out, imshow_v)

plt.show()
# %%
