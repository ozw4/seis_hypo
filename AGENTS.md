# Repository Guidelines (for contributors and automation)

This repository is **not packaged** (no `pip install .`). Most entry points are under `proc/`, and most reusable code lives under `src/`. When running anything locally, you almost always want:

```bash
export PYTHONPATH="$PWD/src"
```

## Project structure

- `src/` — reusable library code (pipelines, IO, HypoInverse/CRE utilities, QC/viz).
- `proc/` — runnable scripts (thin wrappers around `src/`; keep new logic out of here when possible).
- `tests/` — pytest suite (`tests/data/` for small fixtures).
- `docs/` — notes/specs (especially HypoInverse / CRE / synthetic eval).
- `external_source/` — vendored external tools.
  - **Present in this snapshot:** `external_source/hyp1.40/` (HypoInverse).
  - **Expected by `Dockerfile` but NOT included in this snapshot:**
    - `external_source/NonLinLoc/` (to build NonLinLoc executables)
    - `external_source/loki/` (to install the `loki` Python package)

## Build, test, and dev commands

### Local (no Docker)

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r .devcontainer/requirements-dev.txt

export PYTHONPATH="$PWD/src"
pytest -q
```

Notes:
- Host installs of `geopandas` often require system packages (GDAL/GEOS/PROJ). If `pip install geopandas` fails, use the devcontainer.
- Some pipelines also need external binaries (see **External dependencies** below).

### CI-parity install

GitHub Actions runs on **Python 3.11** and uses `requirements-ci.txt`.

```bash
python -m pip install -U pip
python -m pip install -r requirements-ci.txt

export PYTHONPATH="$PWD/src"
export MPLBACKEND=Agg
pytest -q
```

If dependency installation fails due to a git/private dependency, check the workflow at `.github/workflows/pytest.yml` and mirror its git-auth configuration.

### Devcontainer / Docker (recommended for full toolchain)

```bash
docker compose -f .devcontainer/compose-dev.yaml up -d --build
docker compose -f .devcontainer/compose-dev.yaml exec hinet bash
```

Important:
- `.devcontainer/compose-dev.yaml` contains **author-machine-specific bind mounts** (absolute paths). You will need to edit `volumes:` for your environment.
- The compose file is configured for **NVIDIA GPUs**. If you do not have a compatible GPU/driver stack, remove the GPU reservation and `NVIDIA_VISIBLE_DEVICES` (or use a CPU-only base image).

#### Dockerfile pitfalls in this snapshot

`Dockerfile` currently does:
- `COPY external_source/NonLinLoc ...` then builds NonLinLoc
- `COPY external_source/loki ...` then `pip install /opt/loki`

Those directories are **not included** here, so `docker build` will fail unless you provide them. If you only need HypoInverse workflows, the simplest fix is to comment out the NonLinLoc/LOKI sections in `Dockerfile`.

## Running scripts and pipelines

Most runnable entry points are scripts under `proc/`. Example:

```bash
export PYTHONPATH="$PWD/src"
python proc/hypocenter_determination/jma_mobara_hypoinverse/pipeline.py
```

Many `proc/` scripts are intentionally configured via **top-of-file constants** (YAML path + preset name) instead of full CLI flags.

Most scripts assume external data/configs are mounted at:

- `/workspace/data/`

When not using the devcontainer, either:
- keep the same directory convention, or
- edit the top-of-file path constants.

## External dependencies

Depending on the workflow you run, you may need:

- **HypoInverse**: vendored under `external_source/hyp1.40/`.
- **NonLinLoc executables** (e.g., `Vel2Grid`, `Grid2Time`): required by some travel-time pipelines.
  - The `Dockerfile` expects a local `external_source/NonLinLoc/` clone to build these.
- **LOKI Python package**: required by LOKI stacking pipelines.
  - The `Dockerfile` expects a local `external_source/loki/` clone.

If you are working only on HypoInverse/synthetic-eval utilities, you can generally avoid NonLinLoc and LOKI.

## Coding style

- Formatter/linter: Ruff (`ruff.toml`).
  - `ruff format src proc`
  - `ruff check src proc`
- Formatting conventions (repo-wide): **tabs for indentation**, and **single quotes**.
- Naming:
  - modules/functions: `snake_case`
  - classes: `PascalCase`
  - tests: `tests/test_*.py`

## Testing guidelines

- Framework: `pytest` (CI runs `pytest -q` with `PYTHONPATH=src` on Python 3.11).
- Add/adjust tests for bug fixes and behavior changes.
- Keep fixtures small and commit them under `tests/data/`.

## Data, outputs, and configuration

- Large data/configs are expected outside the repo (commonly mounted under `/workspace/data/`); avoid committing generated artifacts.
- Prefer writing new outputs under a dedicated output directory (e.g., `outputs/`, `result/`) or the mounted data directory.

## Where to look next

- High-level workflow overview: `README.md`
- Detailed notes/specs: `docs/`
