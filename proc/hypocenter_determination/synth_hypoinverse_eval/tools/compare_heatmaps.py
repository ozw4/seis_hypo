from __future__ import annotations

import sys
from pathlib import Path


def _find_src_dir(start: Path) -> Path:
	path = start.resolve()
	for directory in [path] + list(path.parents):
		src = directory / 'src'
		if (src / 'hypo' / 'synth_eval' / 'heatmap_compare.py').is_file():
			return src
	raise FileNotFoundError(
		'could not locate repo src/ (expected src/hypo/synth_eval/heatmap_compare.py)'
	)


def main(argv: list[str] | None = None) -> None:
	this_dir = Path(__file__).resolve().parent
	src = _find_src_dir(this_dir)
	if str(src) not in sys.path:
		sys.path.insert(0, str(src))

	from hypo.synth_eval.heatmap_compare import main as run_main

	run_main(argv)


if __name__ == '__main__':
	main(sys.argv[1:])
