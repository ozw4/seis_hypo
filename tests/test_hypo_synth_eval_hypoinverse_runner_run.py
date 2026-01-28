from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hypo.synth_eval import hypoinverse_runner as hr


def test_run_hypoinverse_returns_completedprocess_and_calls_subprocess_run(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	exe_path = tmp_path / 'hyp1'
	cmd_path = tmp_path / 'run.cmd'
	run_dir = tmp_path / 'run'

	exe_path.write_text('', encoding='utf-8')
	cmd_path.write_text("STA 'x'\n", encoding='utf-8', newline='\n')
	run_dir.mkdir(parents=True, exist_ok=True)

	expected = subprocess.CompletedProcess(args=[str(exe_path)], returncode=0)

	seen: dict[str, object] = {}

	def _fake_run(
		args: list[str], *, stdin, cwd: Path, check: bool
	) -> subprocess.CompletedProcess:
		seen['args'] = args
		seen['cwd'] = cwd
		seen['check'] = check
		seen['stdin_readable'] = hasattr(stdin, 'read')
		return expected

	monkeypatch.setattr(hr.subprocess, 'run', _fake_run)

	ret = hr.run_hypoinverse(exe_path, cmd_path, run_dir)

	assert ret is expected
	assert seen['args'] == [str(exe_path)]
	assert seen['cwd'] == run_dir
	assert seen['check'] is True
	assert seen['stdin_readable'] is True
