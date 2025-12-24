# %%
from pathlib import Path

from qc.loki_compare_qc import compare_error_hists_from_compare_csvs

compare_error_hists_from_compare_csvs(
	[
		Path(
			'/workspace/proc/loki_hypo/loki_output_mobara_optuna/compare_jma_vs_loki.csv'
		),
		Path(
			'/workspace/proc/loki_hypo/loki_output_mobara_eqt/compare_jma_vs_loki.csv'
		),
		Path(
			'/workspace/proc/loki_hypo/loki_output_mobara_eqt_trainmiyagi/compare_jma_vs_loki.csv'
		),
	],
	labels=['STALTA', 'EqT(Original)', 'EqT(TrainMiyagi)'],
	out_dir=Path('/workspace/proc/loki_hypo/compare_plots'),
	density=False,
	min_cmax=0,
)
