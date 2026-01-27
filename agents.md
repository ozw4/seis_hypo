# agents.md — seis_hypo-main (Codex / Code Agent Guide)

This repository contains end-to-end pipelines around **seismic event processing**, centered on:

- **Hi-net / JMA (WIN32)**: event-folder preparation, station/channel handling, arrival-time products
- **Forge DAS**: Zarr preparation, window cutting, pick extraction (PhaseNet/EqT/STALTA), GaMMA association
- **LOKI** waveform stacking / locator workflows (Mobara example + Forge DAS pass1)
- **HypoInverse**-based hypocenter determination (Mobara/JMA) + synthetic evaluation utilities

The code is organized as:

- `proc/` = runnable entry scripts (thin wrappers)
- `src/` = reusable core logic (imported via `PYTHONPATH=/workspace/src` in the devcontainer)

---

## Where to start (fast orientation)

1. **Configs & IO utilities**
   - `src/common/config.py` — dataclass configs used by YAML-driven pipelines
   - `src/common/load_config.py` — YAML preset loader
   - `src/io_util/stream.py` — build ObsPy `Stream` from WIN32 or Forge event dirs

2. **Pick one workflow**
   - **Mobara (Hi-net style) LOKI**: `proc/loki_hypo/mobara/`
   - **Forge DAS LOKI (STALTA pass1)**: `proc/loki_hypo/forge/`
   - **Continuous picking (Forge DAS)**: `proc/run_continuous/pick/forge/`
   - **Association (GaMMA)**: `proc/run_continuous/association/forge/`
   - **HypoInverse hypocenter determination**: `proc/hypocenter_determination/jma_mobara_hypoinverse/`

---

## Repository layout

### `proc/` — runnable scripts (entry points)

- `proc/prepare_data/jma/`
  - WIN32 download helpers, station list handling, and arrival-time CSV builders.

- `proc/prepare_data/forge/`
  - Forge DAS utilities: TDMS/Zarr inspection, station metadata tables, event-window cutting for LOKI,
    and helpers to build GaMMA inputs (picks/velocity).

- `proc/loki_hypo/mobara/`
  - Mobara example runners: travel-time pipeline, LOKI waveform stacking, Optuna search, EQT variant.

- `proc/loki_hypo/forge/`
  - Forge DAS travel-time + QC and STALTA pass1 LOKI stacking.

- `proc/run_continuous/pick/forge/`
  - Continuous DAS picking to CSV (EqT / PhaseNet) and debug plotting on TDMS/Zarr.

- `proc/run_continuous/association/forge/`
  - GaMMA association from pick CSV + station metadata + velocity JSON.

- `proc/hypocenter_determination/jma_mobara_hypoinverse/`
  - HypoInverse runners (+ templates) for Mobara/JMA, plus join/QC/plots.

- `proc/hypocenter_determination/synth_hypoinverse_eval/`
  - Synthetic evaluation driver(s) for HypoInverse.

- `proc/data_profiling/`
  - Lightweight profiling / visualization scripts for stations/events/arrivaltime tables.

### `src/` — library code (core logic)

- `src/common/`
  - Config dataclasses, YAML/JSON helpers, time utilities, station helpers, snapshots.

- `src/io_util/`
  - WIN32/DAS stream building, TDMS→Zarr helpers, and trace utilities.

- `src/jma/`
  - WIN32 reading, downloads, station code mapping/presence, and pick/arrivaltime helpers.

- `src/das/`
  - DAS↔JMA matching helpers and pick post-processing utilities.

- `src/pick/` + `src/pipelines/`
  - Probability generation and pick extraction (STALTA/PhaseNet/EqT) and end-to-end pipelines.
  - Notable pipeline modules:
    - `src/pipelines/das_eqt_pipelines.py`
    - `src/pipelines/das_phasenet_pipelines.py`
    - `src/pipelines/loki_stalta_pipelines.py`
    - `src/pipelines/loki_waveform_stacking_pipelines.py`
    - `src/pipelines/traveltime_pipelines.py`

- `src/nonlinloc/`
  - NonLinLoc control file and 1D velocity helpers.

- `src/loki_tools/`
  - LOKI header building, `.phs` parsing, comparisons, plotting utilities.

- `src/hypo/`
  - HypoInverse ARC/PRT handling, phase/station metadata, joining JMA↔HypoInverse outputs.

- `src/waveform/`
  - Preprocessing, filters, transforms.

- `src/qc/` + `src/viz/`
  - QC summaries and plotting utilities (LOKI, JMA arrival-times, NonLinLoc travel-time tables, etc.).

---

## Configuration & paths (important)

### YAML presets are typically *outside* the repo

Many Mobara/LOKI scripts load YAML presets from paths like:

- `/workspace/data/config/loki_waveform_pipeline.yaml`
- `/workspace/data/config/plot_config.yaml`

In the provided devcontainer setup, a host directory is mounted to `/workspace/data/`
(see `.devcontainer/compose-dev.yaml`). These YAML files are therefore **not expected to live in the git repo**.

### Import style

Modules are imported directly from `src/` (for example, `from common.config import LokiWaveformStackingPipelineConfig` or
`from pipelines.loki_waveform_stacking_pipelines import pipeline_loki_waveform_stacking`).
This relies on `PYTHONPATH=/workspace/src` (set in `Dockerfile`).
Do **not** add ad-hoc `sys.path` manipulation.

---

## Data formats (quick reference)

### 1) Hi-net / JMA event directory (WIN32)

Typical event folder contains:

- `event.json` (origin time + metadata)
- WIN32 waveform files (e.g., `.cnt` / `.ch`) referenced by `event.json`

Core conversions live in `src/jma/` and `src/io_util/stream.py`.

### 2) Forge DAS event directory (LOKI direct_input)

Event cutting for LOKI commonly produces:

- `waveform.npy` — shape `(C, T)` float array
- `meta.json` — includes at least sampling rate and window start time
- `stations.csv` — includes at least `station_id` and `index` (channel order aligns with ascending `index`)

`src/io_util/stream.py::build_stream_from_forge_event_npy()` converts this into an ObsPy `Stream`.

### 3) Pick CSV (continuous pipelines)

Continuous pick runners typically output a CSV containing at least:

- `station_id` (or `id`)
- `phase_time` (or `timestamp`)
- `phase_type` (P/S)
- optional `phase_score` / `prob`

`proc/run_continuous/association/forge/run_gamma_forge.py` normalizes column names before GaMMA.

---

## Main workflows (entry points)

### A) Forge DAS → event windows → STALTA pass1 → LOKI

1. (Optional) Inspect/convert raw data:
   - `proc/prepare_data/forge/chk_tdms.py`
   - `proc/prepare_data/forge/build_zarr.py`
   - `proc/prepare_data/forge/chk_zarr.py`

2. Build station metadata:
   - `proc/prepare_data/forge/build_well78_station_meta_table.py`

3. Cut event windows for LOKI:
   - `proc/prepare_data/forge/cut_events_fromzarr_for_loki.py`

4. Build travel-time tables + LOKI header:
   - `proc/loki_hypo/forge/run_traveltime_pipeline.py`
   - `proc/loki_hypo/forge/run_traveltime_qc.py`

5. Run STALTA pass1 LOKI stacking:
   - `proc/loki_hypo/forge/run_loki_waveform_stacking_pipelines_stalta_das_pass1.py`
   - Debug a single event:
     - `proc/loki_hypo/forge/plot_stalta_on_forge_event.py`

Core logic: `src/pipelines/loki_stalta_pipelines.py`.

### B) Mobara (Hi-net style) → LOKI waveform stacking

- Travel-time pipeline:
  - `proc/loki_hypo/mobara/run_traveltime_pipelines.py`
  - `proc/loki_hypo/mobara/run_traveltime_qc.py`
- LOKI waveform stacking + plots + QC:
  - `proc/loki_hypo/mobara/run_loki_waveform_stacking_pipelines.py`
  - EQT variant:
    - `proc/loki_hypo/mobara/run_loki_waveform_stacking_pipelines_eqt.py`

Core logic: `src/pipelines/loki_waveform_stacking_pipelines.py`.

### C) Continuous picking (Forge DAS) → CSV

- EqT:
  - `proc/run_continuous/pick/forge/run_eqt_continuous.py`
- PhaseNet debug plotting:
  - `proc/run_continuous/pick/forge/debug_plot_phasenet_on_tdms.py`
- EqT debug plotting:
  - `proc/run_continuous/pick/forge/debug_plot_eqt_on_tdms.py`

Core logic: `src/pipelines/das_eqt_pipelines.py`, `src/pipelines/das_phasenet_pipelines.py`.

### D) Association (GaMMA)

- `proc/run_continuous/association/forge/run_gamma_forge.py`

Core dependency: `GaMMA` (installed in the devcontainer via a git requirement).

### E) Hypocenter determination (HypoInverse)

- `proc/hypocenter_determination/jma_mobara_hypoinverse/pipeline.py`
- `proc/hypocenter_determination/jma_mobara_hypoinverse/pipeline_with_das.py`
- `proc/hypocenter_determination/jma_mobara_hypoinverse/pipeline_with_jma_fpevent.py`

HypoInverse bundle:

- `external_source/hyp1.40/hypoinverse.exe`
- `external_source/hyp1.40/doc/` and `external_source/hyp1.40/source/`

Core logic: `src/hypo/`.

---

## External dependencies (what is / is not bundled)

Bundled in this repo snapshot:

- HypoInverse under `external_source/hyp1.40/`

Not bundled here (but referenced by scripts / container setup):

- LOKI (python package / binaries)
- NonLinLoc executables (Vel2Grid / Grid2Time)
- Local datasets and YAML presets under `/workspace/data/`
- WIN32 helper binaries mounted into the container (see `.devcontainer/compose-dev.yaml`)

Note: `Dockerfile` contains COPY steps for `external_source/NonLinLoc` and `external_source/loki`.
Those directories are not present in this zip snapshot; building the image will fail unless you supply them.

---

## Conventions for code changes (important for agents)

- Keep `proc/` scripts as thin wrappers; implement logic in `src/`.
- Avoid argparse-based CLIs; most scripts use “edit here” constants or external YAML presets.
- Fail fast: validate inputs early and raise explicit exceptions; do not introduce silent fallbacks.
- Avoid broad `try/except`; handle only expected errors and let everything else crash with a useful traceback.
- Formatting/linting uses `ruff` with **tab indentation** (`ruff.toml` at repo root).
- Do not introduce `_setup_syspath()` or any `sys.path` hacks.

---

## Minimal sanity checks after edits

- Lint/format:
  - `ruff check src proc`
  - `ruff format src proc`

- Smoke test (pick one):
  - Mobara LOKI runner: `proc/loki_hypo/mobara/run_loki_waveform_stacking_pipelines.py`
  - Forge DAS STALTA pass1: `proc/loki_hypo/forge/run_loki_waveform_stacking_pipelines_stalta_das_pass1.py`
  - GaMMA association: `proc/run_continuous/association/forge/run_gamma_forge.py`

When changing IO/metadata logic, verify:

- trace/channel counts match the station metadata used downstream
- timestamps are timezone-consistent (JST vs UTC) where relevant
- output artifacts are created where the pipelines expect them (e.g., LOKI `.phs`, CSV summaries)
