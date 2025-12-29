# proc/jma_arrivetime/run_jma_arrivetime_qc.py
# %%
from __future__ import annotations

from pathlib import Path

from qc.jma_qc_arrival import run_jma_arrivetime_qc

MEAS_CSV = Path('/workspace/data/arrivetime/JMA/arrivetime_measurements_1month.csv')
EPIC_CSV = Path('/workspace/data/arrivetime/JMA/arrivetime_epicenters_1month.csv')
STATION_CSV = Path('/workspace/data/station/jma/station.csv')
OUT_DIR = Path('/workspace/data/qc/jma_arrivetime_1month')


def main() -> None:
	art = run_jma_arrivetime_qc(
		meas_csv=MEAS_CSV,
		epic_csv=EPIC_CSV,
		station_csv=STATION_CSV,
		out_dir=OUT_DIR,
	)

	print('QC finished.')
	print('out_dir:', art.out_dir)
	print('fig_dir:', art.fig_dir)
	print('picks_csv:', art.picks_csv)
	print('event_summary_csv:', art.event_summary_csv)
	print('station_summary_csv:', art.station_summary_csv)
	print('station_sp_csv:', art.station_sp_csv)


if __name__ == '__main__':
	main()
