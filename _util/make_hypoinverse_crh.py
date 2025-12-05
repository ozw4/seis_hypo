# %%
#!/usr/bin/env python3
import sys


def read_jma_velocity(path: str) -> tuple[list[float], list[float], list[float]]:
	depths: list[float] = []
	vp: list[float] = []
	vs: list[float] = []

	with open(path) as f:
		for lineno, line in enumerate(f, 1):
			line = line.strip()
			if not line:
				continue
			cols = line.split()
			if len(cols) != 3:
				raise ValueError(
					f'Line {lineno}: expected 3 columns (Vp Vs Depth), got {len(cols)}'
				)
			p_str, s_str, d_str = cols
			p = float(p_str)
			s = float(s_str)
			d = float(d_str)

			vp.append(p)
			vs.append(s)
			depths.append(d)

	if not depths:
		raise ValueError('No data found in JMA velocity file')

	for i in range(len(depths) - 1):
		if depths[i + 1] < depths[i]:
			raise ValueError('Depths must be sorted in ascending order')

	return depths, vp, vs


def compute_layer_means(
	depths: list[float],
	values: list[float],
	layer_tops: list[float],
	halfspace_bottom: float,
	label: str,
) -> list[tuple[float, float]]:
	if not layer_tops:
		raise ValueError('layer_tops must not be empty')

	if abs(layer_tops[0] - 0.0) > 1e-6:
		raise ValueError('First layer top must be 0.0 km for Hypoinverse CRH')

	for i in range(1, len(layer_tops)):
		if layer_tops[i] <= layer_tops[i - 1]:
			raise ValueError('layer_tops must be strictly increasing')

	max_depth = depths[-1]
	if halfspace_bottom <= layer_tops[-1]:
		raise ValueError('halfspace_bottom must be deeper than last layer top')

	halfspace_bottom = min(halfspace_bottom, max_depth)

	layers: list[tuple[float, float]] = []

	for i, top in enumerate(layer_tops):
		if i + 1 < len(layer_tops):
			bottom = layer_tops[i + 1]
		else:
			bottom = halfspace_bottom

		bottom = min(bottom, max_depth)

		idxs = [
			k
			for k, z in enumerate(depths)
			if (top <= z < bottom) or (i == len(layer_tops) - 1 and top <= z <= bottom)
		]

		if not idxs:
			raise ValueError(
				f'No samples found between {top} and {bottom} km for {label}'
			)

		mean_v = sum(values[k] for k in idxs) / float(len(idxs))
		layers.append((mean_v, top))

	for i in range(len(layers) - 1):
		v0, z0 = layers[i]
		v1, z1 = layers[i + 1]
		if v1 < v0:
			raise ValueError(
				f'{label} velocity must increase with depth for CRH: '
				f'{v1:.3f} at {z1} km < {v0:.3f} at {z0} km'
			)

	return layers


def compute_pos(
	depths: list[float],
	vp: list[float],
	vs: list[float],
	max_depth: float,
) -> float:
	idxs = [i for i, z in enumerate(depths) if z <= max_depth and vs[i] > 0.0]
	if not idxs:
		idxs = [i for i, v in enumerate(vs) if v > 0.0]

	if not idxs:
		raise ValueError('No valid Vp/Vs samples to compute POS')

	ratios = [vp[i] / vs[i] for i in idxs]
	return sum(ratios) / float(len(ratios))


def write_crh(path: str, model_name: str, layers: list[tuple[float, float]]) -> None:
	with open(path, 'w') as f:
		f.write(model_name[:30].ljust(30) + '\n')
		f.writelines(f'{v:5.2f}{top:5.2f}\n' for v, top in layers)


def main(in_path, out_p, out_s) -> None:
	depths, vp, vs = read_jma_velocity(in_path)

	# ★ここを好みに合わせて調整（層境界と半空間の下限深さ）★
	layer_tops = [0.0, 5.0, 10.0, 20.0, 35.0, 50.0, 70.0]  # km
	halfspace_bottom = 200.0  # km

	p_layers = compute_layer_means(depths, vp, layer_tops, halfspace_bottom, label='P')
	s_layers = compute_layer_means(depths, vs, layer_tops, halfspace_bottom, label='S')

	write_crh(out_p, 'JMA2001A_P', p_layers)
	write_crh(out_s, 'JMA2001A_S', s_layers)

	pos = compute_pos(depths, vp, vs, max_depth=60.0)

	sys.stdout.write(f'Wrote P-model CRH: {out_p}\n')
	sys.stdout.write(f'Wrote S-model CRH: {out_s}\n')
	sys.stdout.write('Suggested Hypoinverse commands:\n')
	sys.stdout.write(f"  CRH 1 '{out_p}'\n")
	sys.stdout.write(f"  CRH 2 '{out_s}'\n")
	sys.stdout.write('  SAL 1 2\n')
	sys.stdout.write(
		f'(If you prefer single-model scaling instead of SAL, you could use POS {pos:.2f})\n'
	)


if __name__ == '__main__':
	input_files = '/workspace/data/velocity/jma/vjma2001'
	output_p_crh = 'JMA2001A_P.crh'
	output_s_crh = 'JMA2001A_S.crh'

	main(input_files, output_p_crh, output_s_crh)
