from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _find_src_dir(start: Path) -> Path:
	p = start.resolve()
	for d in [p] + list(p.parents):
		src = d / 'src'
		if (src / 'hypo' / 'synth_eval' / 'pipeline.py').is_file():
			return src
	raise FileNotFoundError(
		'could not locate repo src/ (expected src/hypo/synth_eval/pipeline.py)'
	)


def parse_args(argv: list[str]) -> argparse.Namespace:
	p = argparse.ArgumentParser()
	p.add_argument(
		'--config', required=True, type=Path, help='config yaml path (final config)'
	)
	p.add_argument(
		'--runs-root',
		type=Path,
		default=None,
		help='override runs root directory (default: <this_dir>/runs)',
	)
	p.add_argument('--no-qc', action='store_true', help='skip QC')
	return p.parse_args(argv)


def main(argv: list[str]) -> None:
	args = parse_args(argv)

	this_dir = Path(__file__).resolve().parent
	runs_root = args.runs_root.resolve() if args.runs_root else (this_dir / 'runs')

	src = _find_src_dir(this_dir)
	if str(src) not in sys.path:
		sys.path.insert(0, str(src))

	from hypo.synth_eval.pipeline import run_synth_eval
	from qc.hypo.synth_eval import run_qc

	cfg_path = args.config.expanduser().resolve()
	if not cfg_path.is_file():
		raise FileNotFoundError(f'config not found: {cfg_path}')

	run_dir, df_eval, stats = run_synth_eval(cfg_path, runs_root=runs_root)
	print(stats)
	print(f'[OK] run_dir: {run_dir}')

	if not args.no_qc:
		run_qc(cfg_path)


if __name__ == '__main__':
	main(sys.argv[1:])
