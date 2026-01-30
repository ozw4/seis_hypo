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
   - **Synthetic evaluation (HypoInverse)**: `proc/hypocenter_determination/synth_hypoinverse_eval/`

3. **Core libraries to skim**
   - LOKI pipelines: `src/pipelines/loki_waveform_stacking_pipelines.py`, `src/pipelines/loki_stalta_pipelines.py`
   - Travel-time pipelines: `src/pipelines/traveltime_pipelines.py`
   - HypoInverse + CRE utilities: `src/hypo/` (notably `hypoinverse_cmd.py`, `sta.py`, `cre.py`, `synth_eval/`)
   - Catalog helpers (event/station tables): `src/catalog/`

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

- `src/catalog/`
  - Station/event table builders and lightweight catalog IO helpers.

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
  - NonLinLoc control file and 1D velocity helpers (executables are not bundled in this snapshot).

- `src/loki_tools/`
  - LOKI header building, `.phs` parsing, comparisons, plotting utilities.

- `src/hypo/`
  - HypoInverse ARC/PRT handling, station/phase metadata, and join/QC helpers.
  - CRE utilities live in `src/hypo/cre.py`, station `.sta` helpers in `src/hypo/sta.py`.
  - Synthetic evaluation orchestration lives under `src/hypo/synth_eval/`.

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

Event cutting for LOKI commonly produces per-event folders like:

- `waveform.npy` — shape `(C, T)` float array (channels × samples)
- `meta.json` — event window metadata (UTC timestamps, shapes, optional well geometry)
- `stations.csv` — per-event station/channel table (optional; enabled in the cutter)

The cutter also often writes a root-level `manifest.csv` summarizing all event windows.
`src/io_util/stream.py::build_stream_from_forge_event_npy()` converts `waveform.npy` into an ObsPy `Stream`.

### 3) Pick CSV (continuous pipelines)

Continuous DAS pick runners write a CSV in the canonical schema:

- `segment_id` — segment identifier (int)
- `block_start` — window/block start index (int)
- `channel_id` — channel index (int; absolute or relative depending on config)
- `phase_type` — `'P'` or `'S'`
- `phase_time_ms` — UTC epoch milliseconds (int)
- `phase_time_iso` — ISO8601 string (UTC)
- `phase_prob` — pick probability/score (float)

Writer implementation: `src/pipelines/das_pick_csv_accumulator.py`.
GaMMA association runner (`proc/run_continuous/association/forge/run_gamma_forge.py`) normalizes/joins
this pick CSV with station metadata + velocity JSON.

### 4) Synthetic evaluation artifacts (HypoInverse)

Synthetic eval runs (under `proc/hypocenter_determination/synth_hypoinverse_eval/`) typically create:

- `station_synth.csv` — synth station table with at least:
  - `station_code`
  - `receiver_index` (0-based; used by QC to reconstruct station ordering)
  - `Latitude_deg`, `Longitude_deg`, `Elevation_m`
- HypoInverse `.sta` file generated from station metadata
- CRE model files (`P.cre`, `S.cre`) and a small text meta describing reference/shift, when `model_type: CRE`
- HypoInverse outputs such as `.arc` / `.prt` per run

See also `docs/cre_*_test_spec.md` and `docs/synth_eval_*_test_spec.md` for expected behavior/guarantees.

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

4. Travel-time + QC + stacking entry points:
   - Travel-time table generation: `proc/loki_hypo/forge/run_traveltime_pipeline.py`
   - Travel-time QC: `proc/loki_hypo/forge/run_traveltime_qc.py`
   - STALTA pass1 waveform stacking: `proc/loki_hypo/forge/run_loki_waveform_stacking_pipelines_stalta_das_pass1.py`

### B) Mobara (Hi-net style) → LOKI waveform stacking

- Prepare event inputs: `proc/loki_hypo/mobara/run_prepare_event.py`
- Travel-time pipeline: `proc/loki_hypo/mobara/run_traveltime_pipelines.py`
- Travel-time QC: `proc/loki_hypo/mobara/run_traveltime_qc.py`
- Waveform stacking: `proc/loki_hypo/mobara/run_loki_waveform_stacking_pipelines.py`
- EQT variant: `proc/loki_hypo/mobara/run_loki_waveform_stacking_pipelines_eqt.py`
- Optuna search: `proc/loki_hypo/mobara/run_loki_optuna_search.py`

### C) Continuous picking (Forge DAS) → pick CSV

- EqT continuous runner: `proc/run_continuous/pick/forge/run_eqt_continuous.py`
- Debug plotting helpers:
  - `proc/run_continuous/pick/forge/debug_plot_eqt_on_tdms.py`
  - `proc/run_continuous/pick/forge/debug_plot_phasenet_on_tdms.py`

### D) Association (GaMMA)

- `proc/run_continuous/association/forge/run_gamma_forge.py`

### E) Hypocenter determination (HypoInverse)

- `proc/hypocenter_determination/jma_mobara_hypoinverse/pipeline.py`
- `proc/hypocenter_determination/jma_mobara_hypoinverse/pipeline_with_das.py`
- `proc/hypocenter_determination/jma_mobara_hypoinverse/pipeline_with_jma_fpevent.py`

HypoInverse bundle:

- `external_source/hyp1.40/hypoinverse.exe`
- `external_source/hyp1.40/doc/` and `external_source/hyp1.40/source/`

Core logic: `src/hypo/`.

### F) Synthetic evaluation (HypoInverse + optional CRE)

- Entry script: `proc/hypocenter_determination/synth_hypoinverse_eval/run_synth_eval.py`
  - This script is intentionally non-CLI: edit `CONFIG_PATH` to point at a YAML under
    `proc/hypocenter_determination/synth_hypoinverse_eval/configs/`.

- Typical flow (high-level):
  1. Validate config + station/model settings
  2. Write `station_synth.csv` and HypoInverse `.sta`
  3. If `model_type: CRE`, write `.cre` and patch HypoInverse cmd for CRE usage
  4. Run HypoInverse and collect metrics + QC plots

Specs/guarantees: `docs/synth_eval_pipeline_orchestration_test_spec.md`, `docs/cre_test_spec.md`.

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

- Unit tests (fast):
  - `pytest -q`

- Smoke test (pick one):
  - Mobara LOKI runner: `proc/loki_hypo/mobara/run_loki_waveform_stacking_pipelines.py`
  - Forge DAS STALTA pass1: `proc/loki_hypo/forge/run_loki_waveform_stacking_pipelines_stalta_das_pass1.py`
  - GaMMA association: `proc/run_continuous/association/forge/run_gamma_forge.py`
  - Synth eval (edit `CONFIG_PATH` first): `proc/hypocenter_determination/synth_hypoinverse_eval/run_synth_eval.py`

When changing IO/metadata logic, verify:

- trace/channel counts match the station metadata used downstream
- timestamps are timezone-consistent (JST vs UTC) where relevant
- output artifacts are created where the pipelines expect them (e.g., LOKI `.phs`, pick CSV, `station_synth.csv`)
