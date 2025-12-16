# %%
from pathlib import Path

from jma.arrivetime_reader import convert_epicenter_to_csv, convert_measure_to_csv

if __name__ == '__main__':
	input_dir = Path('/workspace/data/arrivetime/JMA/')
	input_files = sorted(input_dir.glob('d*'))[:]
	print(input_files)

	output_csv = str(input_dir / 'arrivetime_epicenters.csv')
	convert_epicenter_to_csv(input_files, output_csv)
	output_meas = str(input_dir / 'arrivetime_measurements.csv')
	convert_measure_to_csv(input_files, output_meas)
# %%
