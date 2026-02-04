# %%
# proc/prepare_data/jma/make_missing_continuous.py  （薄いラッパー）
from __future__ import annotations

from pathlib import Path

from common.load_config import load_config
from jma.missing_continuous import run_make_missing_continuous
from jma.prepare.config import JmaMissingContinuousConfig

YAML_PATH = Path(__file__).resolve().parent / 'config' / 'missing_continuous.yaml'
PRESET = 'sample'


def main() -> None:
	cfg = load_config(JmaMissingContinuousConfig, YAML_PATH, PRESET)
	run_make_missing_continuous(
		win_event_dir=cfg.win_event_dir,
		meas_csv=cfg.meas_csv,
		epi_csv=cfg.epi_csv,
		pres_csv=cfg.pres_csv,
		mapping_report_csv=cfg.mapping_report_csv,
		near0_suggest_csv=cfg.near0_suggest_csv,
		out_missing_csv=cfg.out_missing_csv,
		skip_if_no_active_ch=cfg.skip_if_no_active_ch,
		date_min=cfg.date_min,
		date_max=cfg.date_max,
		skip_if_done=cfg.skip_if_done,
	)


if __name__ == '__main__':
	main()
