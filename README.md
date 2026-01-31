# seis_hypo-main — Seismic hypocenter / travel-time / LOKI pipelines

This repository contains Python pipelines and utilities for seismic workflows, centered on:

- **Hi-net / JMA (WIN32)**: event-folder preparation (download → per-event folders), station/channel handling, arrival-time products, and QC/plots
- **Forge DAS**: TDMS/Zarr utilities, per-event window cutting, pick extraction helpers (STALTA / PhaseNet / EQTransformer via SeisBench), and GaMMA association helpers
- **LOKI waveform stacking**: Mobara (Hi-net style) and Forge DAS (direct_input / pass1 STALTA) workflows
- **Hypocenter determination (HypoInverse)**: build ARC from arrival-time tables, run HypoInverse, merge results back to a catalog, and visualize
- **Synthetic evaluation utilities**: evaluate pre-generated synthetic datasets with HypoInverse, including **CRE**-based elevation handling

Most runnable entry points are scripts under **`proc/`**. Reusable library code lives under **`src/`**.

> Note: Many pipelines expect YAML presets and large input data to live outside the repo and be mounted under `/workspace/data/` (see below). This repo snapshot does not include those datasets.

---

## TL;DR (how most people run this)

### Option A: VS Code Dev Containers (recommended)

1. Put your external configs/data on your host (e.g. `~/Desktop/data/seis_hypo`).
2. Edit `.devcontainer/compose-dev.yaml` and fix the `volumes:` bind mounts for your machine.
3. In VS Code: **Dev Containers → Reopen in Container**.

### Option B: Run scripts on your host (no Docker)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r .devcontainer/requirements-dev.txt

export PYTHONPATH="$PWD/src"
python proc/hypocenter_determination/jma_mobara_hypoinverse/pipeline.py
```

> Heads up: host installs of `geopandas` often require system packages (e.g., GDAL/GEOS/PROJ). If `pip install geopandas` fails, Docker is usually the quickest path.

---

## Repository layout

- `proc/` — runnable scripts (entry points)
  - `proc/prepare_data/jma/` — Hi-net/JMA download + WIN32 helpers
  - `proc/prepare_data/forge/` — Forge DAS utilities (TDMS→Zarr, station meta, event cutting, GaMMA inputs)
  - `proc/loki_hypo/mobara/` — Mobara (Hi-net style) LOKI workflow (+ Optuna tuning)
  - `proc/loki_hypo/forge/` — Forge DAS travel-time + STALTA-pass1 LOKI stacking
  - `proc/hypocenter_determination/jma_mobara_hypoinverse/` — HypoInverse workflow for Mobara/JMA (+ templates)
  - `proc/hypocenter_determination/synth_hypoinverse_eval/` — synthetic evaluation driver for HypoInverse (CRH / CRE)
  - `proc/run_continuous/` — continuous picking + association (Forge DAS)
  - `proc/data_profiling/` — lightweight profiling / visualization scripts

- `src/` — library code (imported by `proc/`)
  - `src/common/` — config dataclasses, YAML loader, time/json helpers
  - `src/catalog/` — event table selection and catalog helpers
  - `src/io_util/` — stream builders (WIN32 / DAS), TDMS/Zarr helpers, trace utilities
  - `src/jma/` — Hi-net/JMA download + WIN32 parsing
  - `src/das/` — DAS-specific helpers (matching, post-processing)
  - `src/pick/` — probability stream / picking utilities (SeisBench, etc.)
  - `src/pipelines/` — orchestration (travel-time pipelines, LOKI stacking pipelines, STALTA pipelines)
  - `src/nonlinloc/` — NonLinLoc control / 1D velocity helpers (for travel-time tables)
  - `src/loki_tools/` — header / `.phs` parsing, plotting, comparisons (requires the `loki` Python package)
  - `src/hypo/` — HypoInverse utilities (ARC/PRT, station `.sta`, CRH/CRE helpers, synthetic evaluation)
  - `src/waveform/` — preprocessing, filters, transforms
  - `src/qc/`, `src/viz/` — QC and plotting helpers

- `docs/` — test specs and notes (especially for HypoInverse / CRE / synthetic evaluation)
- `tests/` — unit tests

- `external_source/`
  - `external_source/hyp1.40/` — **HypoInverse** binary + upstream docs/source

---

## Environment

### Recommended: devcontainer / Docker

The repo includes:

- `Dockerfile` (target: `develop`)
- `.devcontainer/compose-dev.yaml`

If you prefer to run it without VS Code, you can also use Docker Compose directly:

```bash
docker compose -f .devcontainer/compose-dev.yaml up -d --build
docker compose -f .devcontainer/compose-dev.yaml exec hinet bash
```

> Note: `.devcontainer/compose-dev.yaml` is configured for NVIDIA GPUs. If you do not have a compatible GPU / driver stack, remove the `deploy:` GPU reservation and the `NVIDIA_VISIBLE_DEVICES` setting (or use a CPU-only base image).

Inside the container, the repo is mounted at:

- `/workspace` (this repository)

External configs/data are expected at:

- `/workspace/data/` (mounted from your host)

#### About `.devcontainer/compose-dev.yaml`

`compose-dev.yaml` contains several bind-mounts that are specific to the original author’s machine (for example, absolute paths under `/home/youruser/Desktop/data`). You will almost certainly need to edit those `volumes:` entries for your environment.

A typical mount that many scripts rely on is:

- `${HOME}/Desktop/data/seis_hypo  ->  /workspace/data`

The compose file also mounts a few optional WIN32 helper binaries (`catwin32`, `win2sac_32`) into the container. They are not required for the Python WIN32 reader in `src/jma/win32_reader.py`, but they can be handy for manual inspection/conversion.

### Running without Docker

This repo is not packaged as an installable Python package. If you run scripts on your host, make sure `src/` is on `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/src"
python proc/loki_hypo/mobara/run_prepare_event.py
```

Python dependencies used by the devcontainer are listed in `.devcontainer/requirements-dev.txt`.

---

## External dependencies

### NonLinLoc executables

Some travel-time pipelines require NonLinLoc executables (for example `Vel2Grid`, `Grid2Time`) available on `PATH`.

The `Dockerfile` is written to build NonLinLoc from a local folder:

- `external_source/NonLinLoc`

This folder is not included in the repo snapshot. If you want to build the container as-is, provide that directory before building. Otherwise, comment out the NonLinLoc build section and install NonLinLoc another way.

If you only need HypoInverse workflows (and not travel-time pipelines), the simplest approach is:

1) comment out the `COPY external_source/NonLinLoc /opt/NonLinLoc` line and the subsequent `cmake/make/install` lines in `Dockerfile`
2) remove any `Vel2Grid` / `Grid2Time` usage from your run scripts

Quick workaround if you do *not* need NonLinLoc right now:

- Comment out the "Build NonLinLoc" section in `Dockerfile`.
- Or add a separate Docker target that skips it.

### LOKI Python package

LOKI waveform stacking pipelines require the `loki` Python package. The `Dockerfile` expects a local clone at:

- `external_source/loki`

That directory is not included in the repo snapshot. Provide it (or change the Dockerfile to install `loki` from another source) if you use the LOKI workflows.

If you do not use LOKI, you can comment out the `COPY external_source/loki /opt/loki` and `pip install /opt/loki` lines in `Dockerfile`.

Quick workaround if you do *not* need LOKI right now:

- Comment out the "Install LOKI" section in `Dockerfile`.

### HypoInverse

HypoInverse is vendored under:

- `external_source/hyp1.40/hypoinverse.exe`

Despite the `.exe` suffix, this is intended to be run on Linux in the devcontainer.

---

## Configuration (YAML presets)

Many pipelines use `src/common/load_config.py::load_config()` and read presets from YAML.

A common pattern in `proc/` scripts is:

- a top-of-file `YAML_PATH = Path('/workspace/data/config/traveltime_config.yaml')`
- a `PRESET = 'preset_name'`

In other words, configuration files are typically expected under:

- `/workspace/data/config/*.yaml`

This repo includes some self-contained example configs for synthetic evaluation under:

- `proc/hypocenter_determination/synth_hypoinverse_eval/configs/`

---

## Data directory convention (`/workspace/data`)

Most scripts assume an external data/config tree mounted at `/workspace/data`. Below is a *typical* layout (file names vary by project):

```text
/workspace/data/
  config/
    prepare_events.yaml
    traveltime_config.yaml
    loki_waveform_pipeline.yaml
    plot_config.yaml
  station/
    jma/
      station.csv
      stations_hypoinverse.sta
    forge/
      forge_das_station_metadata.csv
  arrivetime/
    JMA/
    NIED/
  velocity/
    jma_crh/
      JMA2001A_P.crh
      JMA2001A_S.crh
    forge/
  qc/
  synthe/
  N03-20240101_GML/
    N03-20240101_prefecture.shp
```

Many `proc/` scripts hard-code absolute paths like `Path('/workspace/data/config/plot_config.yaml')`. When running on your host, either:

- keep using the container and mount your data to `/workspace/data`, or
- edit the top-of-file constants in the runner scripts to match your local paths

---

## Quickstart: Mobara (Hi-net style) → travel-times → LOKI → QC

Run these from the repository root.

### 1) Prepare per-event folders (download WIN32)

Uses Hi-net credentials (commonly via a mounted `~/.netrc`).

```bash
python proc/loki_hypo/mobara/run_prepare_event.py
```

### 2) Build travel-time DB (+ `header.hdr`)

Requires NonLinLoc executables `Vel2Grid` and `Grid2Time` in your `PATH`.

```bash
python proc/loki_hypo/mobara/run_traveltime_pipelines.py
```

### 3) Run LOKI waveform stacking + plots + LOKI↔JMA QC

Requires the `loki` Python package.

```bash
python proc/loki_hypo/mobara/run_loki_waveform_stacking_pipelines.py
```

Optional:

- EQTransformer-assisted variant: `proc/loki_hypo/mobara/run_loki_waveform_stacking_pipelines_eqt.py`
- Parameter search (Optuna): `proc/loki_hypo/mobara/run_loki_optuna_search.py`

---

## Quickstart: Forge DAS → travel-times → STALTA pass1 LOKI

Typical flow:

1) Build/inspect Zarr (if starting from TDMS):

- `proc/prepare_data/forge/build_zarr.py`
- `proc/prepare_data/forge/chk_tdms.py`
- `proc/prepare_data/forge/chk_zarr.py`

2) Cut per-event windows for LOKI input:

- `proc/prepare_data/forge/cut_events_fromzarr_for_loki.py`

3) Build travel-time DB + `header.hdr` (NonLinLoc tools required):

- `proc/loki_hypo/forge/run_traveltime_pipeline.py`
- QC for travel-time tables: `proc/loki_hypo/forge/run_traveltime_qc.py`

4) Run STALTA-pass1 stacking (LOKI direct_input-style):

- `proc/loki_hypo/forge/run_loki_waveform_stacking_pipelines_stalta_das_pass1.py`

---

## Hypocenter determination (HypoInverse)

Example JMA Mobara pipeline:

- `proc/hypocenter_determination/jma_mobara_hypoinverse/pipeline.py`

Command templates used by that workflow live under:

- `proc/hypocenter_determination/jma_mobara_hypoinverse/template/`

This workflow typically:

- reads arrival-time tables under `/workspace/data/arrivetime/`
- writes HypoInverse input/output (ARC/PRT/etc.) under a local `run_dir`
- generates QC plots/maps (often requiring shapefiles and plotting configs under `/workspace/data/`)

---

## Synthetic HypoInverse evaluation (CRH / CRE)

Synthetic evaluation entry point:

```bash
python proc/hypocenter_determination/synth_hypoinverse_eval/run_synth_eval.py
```

Configs live under:

- `proc/hypocenter_determination/synth_hypoinverse_eval/configs/`

Key features:

- reads a synthetic dataset (geometry + event catalogs) from `dataset_dir`
- runs HypoInverse with a template command file
- evaluates errors and produces QC plots
- supports **CRH** and **CRE** modes
  - CRE mode can delegate station elevation handling to HypoInverse (see `src/hypo/cre.py`)
  - `station_synth.csv` includes a `receiver_index` column for stable receiver mapping

Most configs assume the synthetic dataset directory is under `/workspace/data/`.

---

## Development

- Lint: `ruff check src proc`
- Format: `ruff format src proc`
- Tests: `pytest`

Many `proc/` scripts are intentionally parameterized by constants near the top of the file (instead of argparse CLIs).
If you want to change I/O paths or presets, start by editing those top-of-file parameters.

---

## License / attribution

This repository does not currently include a `LICENSE` file. If you plan to redistribute or publish results based on this code, confirm the intended licensing with the repository owner.

The following components are external projects with their own licenses/citation requirements:

- HypoInverse (vendored under `external_source/hyp1.40/`)
- NonLinLoc (expected under `external_source/NonLinLoc` when building the devcontainer)
- LOKI (expected under `external_source/loki` when building the devcontainer)
- GaMMA and SeisBench (installed via pip)
