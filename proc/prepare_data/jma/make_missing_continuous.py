# %%
# proc/prepare_data/jma/make_missing_continuous.py  （薄いラッパー）
from __future__ import annotations

from pathlib import Path

from jma.missing_continuous import run_make_missing_continuous

WIN_EVENT_DIR = Path('/workspace/data/waveform/jma/event').resolve()

MEAS_CSV = Path(
	'/workspace/data/arrivetime/JMA/arrivetime_measurements_2023.0.csv'
).resolve()
EPI_CSV = Path(
	'/workspace/data/arrivetime/JMA/arrivetime_epicenters_2023.0.csv'
).resolve()

PRES_CSV = Path(
	'/workspace/proc/prepare_data/jma/stationcode_match/v1/snapshots/monthly/monthly_presence.csv'
).resolve()

MATCH_OUT_DIR = Path(
	'/workspace/proc/prepare_data/jma/stationcode_match/v1/match_out_final'
).resolve()
MAPPING_REPORT_CSV = (MATCH_OUT_DIR / 'mapping_report.csv').resolve()
NEAR0_SUGGEST_CSV = (MATCH_OUT_DIR / 'near0_suggestions.csv').resolve()

OUT_MISSING_CSV = Path(
	'/workspace/data/waveform/jma/event/missing_in_active_ch.csv'
).resolve()

SKIP_IF_NO_ACTIVE_CH = True


DATE_MIN = '2023-01-01'
DATE_MAX = '2023-01-31'


def main() -> None:
	run_make_missing_continuous(
		win_event_dir=WIN_EVENT_DIR,
		meas_csv=MEAS_CSV,
		epi_csv=EPI_CSV,
		pres_csv=PRES_CSV,
		mapping_report_csv=MAPPING_REPORT_CSV,
		near0_suggest_csv=NEAR0_SUGGEST_CSV,
		out_missing_csv=OUT_MISSING_CSV,
		skip_if_no_active_ch=SKIP_IF_NO_ACTIVE_CH,
		date_min=DATE_MIN,
		date_max=DATE_MAX,
	)


if __name__ == '__main__':
	main()
