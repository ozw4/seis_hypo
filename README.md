# Seismic Hypocenter / LOKI Waveform Stacking Pipelines

This repository contains end-to-end pipelines for:

- Preparing Hi-net WIN32 event folders (`event.json` + `.cnt/.ch`)
- Building travel-time tables (NonLinLoc `Vel2Grid`/`Grid2Time`) and a LOKI `header.hdr`
- Running LOKI waveform stacking/locator
- QC plots and LOKI↔JMA comparison outputs

All runnable entry points are under `proc/` and are driven by YAML presets under `data/config/`.

## Key Concepts

### Event directory format

The LOKI waveform pipeline expects event directories like:

`<base_input_dir>/<event_id>/`

containing at least:

- `event.json` (includes `origin_time_jst` preferred, or `origin_time`)
- one or more WIN32 files referenced from `event.json` (`win32.cnt_files` / `win32.ch_file`)

### Travel-time database

LOKI runs require travel-time tables and a header:

- `header.hdr` and `*.time.*.buf/*.hdr` are expected under a single “db” directory
- the travel-time pipeline writes these under: `<output_dir>/db/`

## Quickstart

### 1) Prepare event folders (WIN32 download)

Edit `data/config/prepare_events.yaml` to point to your catalog and station selection, then run:

`python proc/loki_hypo/run_prepare_event.py`

This writes event directories under `PrepareEventsConfig.base_input_dir`.

### 2) Build travel-time tables + LOKI header

Edit `data/config/traveltime_config.yaml` and run:

`python proc/loki_hypo/run_traveltime_pipelines.py`

Outputs are written under `TravelTimeBaseConfig.output_dir` (including `db/header.hdr`).

### 3) Run LOKI waveform stacking + plots + QC

Edit `data/config/loki_waveform_pipeline.yaml` and `data/config/loki_inputs.yaml`, then run:

`python proc/loki_hypo/run_loki_waveform_stacking_pipelines.py`

This performs:

- LOKI waveform stacking/locator
- Per-event waveform gather plots with LOKI picks (when `*_trial0.phs` exists)
- LOKI vs JMA QC compare (CSV + plots under `error_stats/`)

## LOKI Waveform Pipeline Configuration

The pipeline runner loads `LokiWaveformStackingPipelineConfig` from `data/config/loki_waveform_pipeline.yaml`.

Important fields:

- `base_input_dir`: directory containing `<event_id>/event.json`
- `base_traveltime_dir`: directory containing a `db/` folder with travel-time tables
- `loki_db_path`: path to travel-time `db/` directory (often `{base_traveltime_dir}/db`)
- `loki_hdr_filename`: header filename inside `loki_db_path` (typically `header.hdr`)
- `loki_output_path`: output directory for LOKI results and QC artifacts
- `loki_data_path`: temporary LOKI “data tree” (must be safe to delete)
- `event_glob`, `max_events`: event selection by directory name/glob

### Event filtering (YAML-driven)

You can filter which event directories are processed without code changes:

- `origin_time_start`: ISO string, inclusive
- `origin_time_end`: ISO string, inclusive
- `mag_min`, `mag_max`: JMA magnitude bounds (float)
- `drop_if_mag_missing`: if a magnitude filter is set and mag is missing, drop the event when `true`

Magnitude is read from `event.json` `extra` keys in priority order: `mag1`, `magnitude`, `mag`.

## Outputs

Under `LokiWaveformStackingPipelineConfig.loki_output_path`:

- One subdirectory per processed event (LOKI output)
- `compare_jma_vs_loki.csv` (LOKI↔JMA merged comparison)
- `loki_vs_jma.png` (map + sections plot)
- `error_stats/` (histograms, boxplots, outlier CSVs, magnitude-binned summary)

## Troubleshooting

- If an “excluded” event is still processed, check whether `loki_data_path` contains stale event leaf directories from a previous run; LOKI enumerates events by walking `loki_data_path`.
- If you see timezone-related errors, ensure `event.json` contains `origin_time_jst` with a timezone offset, or confirm the code treats timezone-naive `origin_time_jst` values as JST.

## Development

- Lint: `ruff check src proc`
- Format: `ruff format src proc`
