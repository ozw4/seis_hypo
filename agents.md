# agent.md — seis_hypo-main (Codex / Code Agent Guide)

This repository contains end-to-end pipelines for **seismic hypocenter / location workflows** and **LOKI waveform stacking**, covering both:

- **JMA / Hi-net WIN32** event-based workflows (download → event folder → travel-times → LOKI → comparison/QC)
- **Forge DAS** workflows (TDMS/Zarr → event window cutting → STALTA-based prob streams → LOKI direct_input)

It is organized with **thin runnable entry scripts** under `proc/` and reusable **core logic** under `src/`.

---

## Where to start (fast orientation)

1. `src/common/config.py` + `src/common/load_config.py`
   Configuration dataclasses and YAML loading helpers (note: YAML presets are **not included** in this zip snapshot).

2. Choose your workflow:
   - **JMA / Hi-net**: start at `proc/prepare_data/jma/` and `proc/loki_hypo/mobara/`
   - **Forge DAS + LOKI (STALTA pass1)**: start at `proc/prepare_data/forge/` and `proc/loki_hypo/forge/`

---

## Repository layout

- `proc/` — runnable scripts (entry points)
  - `proc/prepare_data/jma/` — Hi-net WIN32 download, station matching, event window preparation
  - `proc/prepare_data/forge/` — Forge DAS utilities (TDMS→Zarr, station meta, event cutting for LOKI)
  - `proc/loki_hypo/mobara/` — LOKI waveform stacking pipeline for the Mobara example dataset
  - `proc/loki_hypo/forge/` — Forge DAS travel-time pipeline, QC, STALTA pass1 LOKI stacking
  - `proc/hypocenter_determination/` — hypocenter determination pipelines (incl. variants with DAS)
  - `proc/run_continuous/` — continuous picking / association (Forge)

- `src/` — library code (core logic)
  - `src/io_util/stream.py` — build ObsPy `Stream` from WIN32 or Forge DAS event dirs
  - `src/pipelines/` — orchestration logic:
    - `traveltime_pipelines.py` — NonLinLoc travel-time table building
    - `loki_waveform_stacking_pipelines.py` — generic LOKI waveform stacking pipeline + event listing utilities
    - `loki_stalta_pipelines.py` — Forge DAS STALTA→prob-stream→LOKI direct_input (pass1)
  - `src/loki_tools/` — LOKI parsing, plotting, comparisons
  - `src/nonlinloc/` — NonLinLoc control / velocity model helpers
  - `src/jma/` — Hi-net WIN32 download + event directory creation
  - `src/pick/` — picking / probability stream tools (STALTA, etc.)
  - `src/viz/`, `src/qc/` — plotting and QC helpers

- `external_source/`
  - `external_source/hyp1.40/` — HypoInverse (binary + source + docs).
  - **LOKI is NOT bundled** in this zip snapshot (you must supply it separately in your environment).

- `.devcontainer/`, `Dockerfile` — development container setup; dependencies are listed in `.devcontainer/requirements-dev.txt`.

---

## Data formats (important)

### 1) Hi-net / JMA event directory (WIN32-based)
The JMA workflow creates per-event folders containing:
- `event.json` (origin time, meta, window definitions, etc.)
- WIN32 waveform files (e.g., `.cnt` / `.ch`) downloaded via HinetPy
- Optionally derived products used by downstream QC / hypocenter steps

Core conversions live in `src/jma/` and `src/io_util/stream.py`.

### 2) Forge DAS event directory (LOKI direct_input)
Forge DAS event cutting produces:
- `waveform.npy` — shape `(C, T)` float array
- `meta.json` — includes at least `fs_hz` and `window_start_utc`
- `stations.csv` — includes at least `station_id` and `index`
  Convention: channel order is expected to align with ascending `index`.

`src/io_util/stream.py::build_stream_from_forge_event_npy()` converts this into an ObsPy `Stream`.
**Channel code must end with `Z`** (1 component DAS treated as Z).

### 3) LOKI outputs
Under `cfg.loki_output_path`, outputs are typically structured as:
- one subdirectory per event
- `.phs` files for each trial
- optional merged comparison/QC products (CSV/PNGs) depending on the pipeline

LOKI `.phs` parsing and plotting utilities live in `src/loki_tools/`.

---

## Main workflows (entry points)

### A) Forge DAS → event windows → STALTA pass1 → LOKI
1. (Optional) Inspect/convert raw data:
   - `proc/prepare_data/forge/chk_tdms.py`
   - `proc/prepare_data/forge/build_zarr.py`
   - `proc/prepare_data/forge/chk_zarr.py`

2. Build station metadata (Forge-specific):
   - `proc/prepare_data/forge/build_well78_station_meta_table.py`

3. Cut event windows for LOKI (creates `waveform.npy/meta.json/stations.csv`):
   - `proc/prepare_data/forge/cut_events_fromzarr_for_loki.py`

4. Build travel-time tables + LOKI header (NonLinLoc-based):
   - `proc/loki_hypo/forge/run_traveltime_pipeline.py`
   - `proc/loki_hypo/forge/run_traveltime_qc.py`

5. Run STALTA pass1 waveform stacking via LOKI direct_input:
   - `proc/loki_hypo/forge/run_loki_waveform_stacking_pipelines_stalta_das_pass1.py`
   - Debug single event STALTA:
     - `proc/loki_hypo/forge/plot_stalta_on_forge_event.py`

Core logic is in `src/pipelines/loki_stalta_pipelines.py`.

### B) Mobara (Hi-net style) example → LOKI waveform stacking
- Entry scripts under `proc/loki_hypo/mobara/`
- Core LOKI pipeline utilities under `src/pipelines/loki_waveform_stacking_pipelines.py`

### C) Hypocenter determination (HypoInverse / DAS variants)
- Entry scripts:
  - `proc/hypocenter_determination/pipeline.py`
  - `proc/hypocenter_determination/pipeline_with_das.py`
  - `proc/hypocenter_determination/pipeline_with_jma_fpevent.py`
- HypoInverse binary and docs are bundled in `external_source/hyp1.40/`.

---

## External dependencies (what is / is not included)

Included:
- HypoInverse (external_source/hyp1.40)
- Python source for the pipelines

Not included in this zip snapshot (expected to exist in your environment):
- **LOKI** (and any required binaries/resources)
- **NonLinLoc** executables (Vel2Grid/Grid2Time), if used externally
- Forge DAS raw data files (TDMS) and any local data directories referenced in `proc/`

---

## Conventions for code changes (important for agents)

- **Do not add argparse-based CLIs.** Entry scripts typically expose “user edit” constants near the top.
- Prefer changes in `src/` and keep `proc/` scripts thin wrappers.
- Keep behavior identical when new options are disabled (default-off feature flags).
- Favor fail-fast validation (explicit `ValueError` / `FileNotFoundError`) over silent fallbacks.
- Formatting/linting: `ruff` (`ruff.toml` exists at repo root).
- Avoid introducing helper functions like `_setup_syspath()`; keep imports package-based.

---

## Minimal sanity checks after edits

- `ruff check src proc` (and optionally `ruff format src proc`)
- Run a small Forge DAS pass1 subset (e.g., 1–3 events) using:
  - `proc/loki_hypo/forge/run_loki_waveform_stacking_pipelines_stalta_das_pass1.py`
- If you changed stream/metadata logic, verify:
  - number of traces == number of station IDs used for probs
  - LOKI produces a `.phs` for each processed event
