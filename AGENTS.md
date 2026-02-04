# Repository Guidelines

## Project Structure & Module Organization

- `src/`: reusable library code (pipelines, IO, hypo/CRE utilities, QC/viz).
- `proc/`: runnable entry scripts (thin wrappers around `src/`; keep new logic out of here when possible).
- `tests/`: pytest suite (`tests/data/` for small fixtures).
- `docs/`: design notes and test specs (especially HypoInverse/CRE/synthetic eval).
- `external_source/`: vendored/required external tools (e.g., HypoInverse, NonLinLoc, loki).

This repo is not packaged; most scripts expect `src/` on `PYTHONPATH`.

## Build, Test, and Development Commands

Local (no Docker):

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r .devcontainer/requirements-dev.txt
# Or (matches GitHub Actions): python3 -m pip install -r requirements-ci.txt
export PYTHONPATH="$PWD/src"
pytest -q
```

Devcontainer/Docker (recommended for full toolchain):

```bash
docker compose -f .devcontainer/compose-dev.yaml up -d --build
docker compose -f .devcontainer/compose-dev.yaml exec hinet bash
```

Run a pipeline/script (example):

```bash
export PYTHONPATH="$PWD/src"
python3 proc/hypocenter_determination/jma_mobara_hypoinverse/pipeline.py
```

## Coding Style & Naming Conventions

- Formatter/linter: Ruff (`ruff.toml`). Use `ruff format .` and `ruff check .`.
- Indentation: tabs (repo-wide Ruff setting); strings prefer single quotes.
- Names: modules/functions `snake_case`; classes `PascalCase`; tests `tests/test_*.py`.

## Testing Guidelines

- Framework: `pytest` (CI runs `pytest -q` with `PYTHONPATH=src` on Python 3.11).
- Add/adjust tests for bug fixes and behavior changes; keep fixtures small and checked into `tests/data/`.
- Optional coverage locally: `pytest --cov=src --cov-report=term-missing`.

## Commit & Pull Request Guidelines

- Commit subjects commonly use Conventional-Commit-style prefixes (`feat:`, `fix:`, `refactor:`) or clear imperatives (`Add ...`, `Refactor ...`). Prefer one-line, scoped changes.
- PRs: describe intent + risk, link issues, note how you tested (`pytest`, scripts run), and include before/after plots or screenshots when changing QC/viz output.

## Data, Outputs, and Configuration

- Large data/configs are typically mounted externally (devcontainer expects `/workspace/data/`); avoid committing generated artifacts.
- Many file types/outputs are gitignored (see `.gitignore`); write new outputs under `outputs/`/`result/` or your mounted data dir, not under `src/`.
- For deeper workflow orientation (Mobara/JMA, Forge DAS, LOKI, HypoInverse), see `agents.md`.
