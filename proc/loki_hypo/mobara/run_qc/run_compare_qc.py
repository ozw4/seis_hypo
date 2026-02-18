# %%
from pathlib import Path

from qc.loki.compare import compare_error_hists_from_compare_csvs

compare_error_hists_from_compare_csvs(
	[
		Path(
			'/workspace/proc/loki_hypo/mobara/loki_output_mobara_optuna/compare_jma_vs_loki.csv'
		),
		Path(
			'/workspace/proc/loki_hypo/mobara/loki_output_mobara_eqt/compare_jma_vs_loki.csv'
		),
		Path(
			'/workspace/proc/loki_hypo/mobara/loki_output_mobara_eqt_trainmiyagi/compare_jma_vs_loki.csv'
		),
	],
	labels=['STALTA', 'EqT(WithoutFT)', 'EqT(FT)'],
	bins_dh=50,
	bins_dz=50,
	out_dir=Path('/workspace/proc/loki_hypo/mobara/compare_plots'),
	density=True,
	min_cmax=0,
	max_cmax=80,
)
