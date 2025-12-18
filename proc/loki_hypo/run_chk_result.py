# %%
#!/usr/bin/env python3
# proc/loki_hypo/run_plot_compare_jma_loki.py
#
# JMA(event.json) と LOKI(.loc) を比較して、
# - LOKI点群（cmaxで色/サイズ）
# - JMA点群（extras_lld）
# - 対応線（links_lld）
# を plot_events_map_and_sections で1枚に描く（直書き運用版）

from __future__ import annotations

from pathlib import Path

from common.load_config import load_config
from qc.loki_compare_qc import run_loki_vs_jma_qc
from viz.plot_config import PlotConfig


def main() -> None:
	# =========================
	# ここを環境に合わせて直書き
	# =========================
	USE_BUILD_COMPARE_DF = True

	base_input_dir = Path('/workspace/data/waveform')
	loki_output_dir = Path('/workspace/proc/loki_hypo/loki_output_mobara_w_preprocess')
	header_path = Path('/workspace/proc/loki_hypo/mobara_traveltime/db/header.hdr')
	event_glob = '[0-9]*'

	compare_csv_in = loki_output_dir / 'compare_jma_vs_loki.csv'
	out_png = loki_output_dir / 'loki_vs_jma.png'

	plot_setting = 'mobara_default'
	plot_config_yaml = Path('/workspace/data/config/plot_config.yaml')

	# =========================

	params = load_config(PlotConfig, plot_config_yaml, plot_setting)

	run_loki_vs_jma_qc(
		base_input_dir=base_input_dir,
		loki_output_dir=loki_output_dir,
		header_path=header_path,
		event_glob=event_glob,
		plot_cfg=params,
		use_build_compare_df=USE_BUILD_COMPARE_DF,
		compare_csv_out=compare_csv_in if USE_BUILD_COMPARE_DF else None,
		compare_csv_in=None if USE_BUILD_COMPARE_DF else compare_csv_in,
		out_png=out_png,
	)


if __name__ == '__main__':
	main()
