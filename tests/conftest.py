"""Pytest configuration.

This repo is not installed as a package.
Modules under ./src are imported as top-level (e.g. `import common`).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest


def _ensure_src_on_syspath() -> None:
	repo_root = Path(__file__).resolve().parents[1]
	src = repo_root / 'src'
	src_str = str(src)
	if src_str not in sys.path:
		sys.path.insert(0, src_str)


_ensure_src_on_syspath()


@pytest.fixture
def sample_event_dict() -> dict[str, Any]:
	"""A minimal event.json-like payload."""
	return {
		'event_id': 'E000001',
		'origin_time': '2020-01-01T00:00:00+00:00',
		'lat': 35.0,
		'lon': 140.0,
		'depth_km': 10.0,
	}


@pytest.fixture
def write_text(tmp_path: Path) -> Callable[[str, str], Path]:
	"""Return a helper to write UTF-8 text under tmp_path."""

	def _write(rel: str, text: str) -> Path:
		p = tmp_path / rel
		p.parent.mkdir(parents=True, exist_ok=True)
		p.write_text(text, encoding='utf-8')
		return p

	return _write
