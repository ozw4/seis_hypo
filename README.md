# seis_hypo-main — Seismic hypocenter / travel-time / LOKI pipelines

This repository contains Python pipelines and utilities for seismic workflows including:

- **JMA / Hi-net WIN32**: event preparation (download → per-event folders), travel-time DB building, LOKI waveform stacking, and QC/plots
- **Forge DAS**: TDMS/Zarr utilities, per-event cutting, STALTA / probability-stream helpers, and LOKI “direct_input”-style stacking
- **Hypocenter determination (HypoInverse)**: build ARC from arrival-time CSVs, run HypoInverse, merge back to catalog, and visualize
- **Continuous pick + association (Forge)**: continuous picking (e.g., EQTransformer / PhaseNet via SeisBench) and association (GaMMA)

Most runnable entry points are scripts under **`proc/`**. Reusable library code lives under **`src/`**.

> Note: YAML presets and large input data are expected to live outside the repo and be mounted under `/workspace/data/` (see below). This zip snapshot does **not** include those datasets/configs.

---

## Repository layout

- `proc/` — runnable scripts (entry points)
  - `proc/prepare_data/jma/` — Hi-net/JMA download + WIN32 helpers
  - `proc/prepare_data/forge/` — Forge DAS utilities (TDMS→Zarr, station meta, event cutting)
  - `proc/loki_hypo/mobara/` — example Mobara (Hi-net style) LOKI workflow (+ Optuna tuning)
  - `proc/loki_hypo/forge/` — Forge DAS travel-time + STALTA-pass1 LOKI stacking
  - `proc/hypocenter_determination/` — HypoInverse workflows (JMA Mobara example, synth eval)
  - `proc/run_continuous/` — continuous picking + association (Forge)

- `src/` — library code (imported by `proc/`)
  - `src/common/` — config dataclasses, YAML loader, time/json helpers
  - `src/jma/` — Hi-net/JMA download + WIN32 parsing
  - `src/nonlinloc/` — NonLinLoc control/layer helpers (for travel-time tables)
  - `src/loki_tools/` — header / .phs parsing, plotting, comparisons (LOKI Python package required)
  - `src/pipelines/` — orchestration (travel-time pipeline, LOKI stacking pipelines, STALTA pipelines)
  - `src/pick/` — probability-stream / picking utilities (SeisBench, etc.)
  - `src/qc/`, `src/viz/` — QC and plotting helpers

- `external_source/`
  - `external_source/hyp1.40/` — **HypoInverse** binary (Linux ELF) + docs/source

---

## Environment

### Recommended: devcontainer / Docker

The repo includes:

- `Dockerfile` (target: `develop`)
- `.devcontainer/compose-dev.yaml` (bind-mounts repo and external data)

Inside the container, the repo is mounted at:

- `/workspace`  (this repository)

External configs/data are expected at:

- `/workspace/data/`  (mounted from your host)

Typical mount in `.devcontainer/compose-dev.yaml`:

- `${HOME}/Desktop/data/seis_hypo  ->  /workspace/data`

#### Important: external_source/NonLinLoc and external_source/loki

The `Dockerfile` expects these two directories when building:

- `external_source/NonLinLoc` (to build/install `Vel2Grid` / `Grid2Time`)
- `external_source/loki` (pip-installed as the `loki` Python package)

They are **not included** in this zip snapshot. Either:
- provide them in `external_source/` before building, or
- edit the `Dockerfile` to skip those steps and install them another way.

---

## Configuration (YAML presets)

Many pipelines use `src/common/load_config.py::load_config()` and read presets from YAML.

In this repo snapshot, scripts typically reference YAMLs under:

- `/workspace/data/config/*.yaml`

Examples:
- `proc/loki_hypo/mobara/run_prepare_event.py` → `/workspace/data/config/prepare_events.yaml` preset `mobara`
- `proc/loki_hypo/mobara/run_traveltime_pipelines.py` → `/workspace/data/config/traveltime_config.yaml` preset `mobara`
- `proc/loki_hypo/mobara/run_loki_waveform_stacking_pipelines.py` → `/workspace/data/config/loki_waveform_pipeline.yaml` preset `mobara`

`load_config()` also supports simple template expansion like `{some_key}` inside YAML string values.

---

## Quickstart: Mobara (Hi-net style) → travel-times → LOKI → QC

Run these from the repository root (with `PYTHONPATH` including `src/` — the devcontainer sets this):

### 1) Prepare per-event folders (download WIN32)

Uses Hi-net credentials (typically via a mounted `~/.netrc`).

```bash
python proc/loki_hypo/mobara/run_prepare_event.py
```

### 2) Build travel-time DB (+ `header.hdr`)

Requires NonLinLoc executables `Vel2Grid` and `Grid2Time` in your `PATH`.

```bash
python proc/loki_hypo/mobara/run_traveltime_pipelines.py
```

### 3) Run LOKI waveform stacking + plots + LOKI↔JMA QC

Requires the **`loki` Python package** to be installed.

```bash
python proc/loki_hypo/mobara/run_loki_waveform_stacking_pipelines.py
```

Optional:
- EQT-assisted variant: `proc/loki_hypo/mobara/run_loki_waveform_stacking_pipelines_eqt.py`
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
- QC for travel-time tables:
  - `proc/loki_hypo/forge/run_traveltime_qc.py`

4) Run STALTA-pass1 stacking (LOKI direct_input-style):
- `proc/loki_hypo/forge/run_loki_waveform_stacking_pipelines_stalta_das_pass1.py`

---

## Hypocenter determination (HypoInverse)

HypoInverse binary is included at:
- `external_source/hyp1.40/hypoinverse.exe`

Example JMA Mobara pipeline:
- `proc/hypocenter_determination/jma_mobara_hypoinverse/pipeline.py`

This workflow typically:
- reads arrival-time CSVs under `/workspace/data/arrivetime/`
- writes HypoInverse input/output (ARC/PRT/etc.) under a local `run_dir`
- generates QC plots/maps (requires shapefiles and plotting configs under `/workspace/data/`)

---

## Development

- Lint: `ruff check src proc`
- Format: `ruff format src proc`

Many `proc/` scripts are intentionally **parameterized by constants near the top** (instead of argparse CLIs).
If you want to change I/O paths or presets, start by editing those top-of-file parameters.
