#!/usr/bin/env bash
set -euo pipefail

# =========================================
# settings: ここだけ自分の環境に合わせて編集
# =========================================
REPO_ROOT="/workspace"
EVENT_ROOT="/workspace/data/jma/waveform/event/202201"

MEAS_CSV="/workspace/data/jma/arrivetime/2022/measurement.csv"
EPI_CSV="/workspace/data/jma/arrivetime/2022/epicenter.csv"
PRES_CSV="/workspace/data/jma/station/monthly_presence_update.csv"
MAPPING_REPORT_CSV="/workspace/proc/prepare_data/jma/stationcode_match/v1/match_out_final/mapping_report.csv"
NEAR0_SUGGEST_CSV="/workspace/proc/prepare_data/jma/stationcode_match/v1/match_out_final/near0_suggestions.csv"
HINET_CHANNEL_TABLE="/workspace/data/jma/station/hinet_channelstbl_20260413.ch"

RUN_TAG="v1"
THREADS=8
TARGET_FS_HZ=100
SCAN_RATE_BLOCKS=1000

# =========================================
# move to repo
# =========================================
cd "$REPO_ROOT"
export PYTHONPATH="$PWD/src"

# =========================================
# check inputs
# =========================================
[[ -d "$EVENT_ROOT" ]] || { echo "EVENT_ROOT not found: $EVENT_ROOT" >&2; exit 1; }
[[ -f "$MEAS_CSV" ]] || { echo "MEAS_CSV not found: $MEAS_CSV" >&2; exit 1; }
[[ -f "$EPI_CSV" ]] || { echo "EPI_CSV not found: $EPI_CSV" >&2; exit 1; }
[[ -f "$PRES_CSV" ]] || { echo "PRES_CSV not found: $PRES_CSV" >&2; exit 1; }
[[ -f "$MAPPING_REPORT_CSV" ]] || { echo "MAPPING_REPORT_CSV not found: $MAPPING_REPORT_CSV" >&2; exit 1; }
[[ -f "$NEAR0_SUGGEST_CSV" ]] || { echo "NEAR0_SUGGEST_CSV not found: $NEAR0_SUGGEST_CSV" >&2; exit 1; }
[[ -f "$HINET_CHANNEL_TABLE" ]] || { echo "HINET_CHANNEL_TABLE not found: $HINET_CHANNEL_TABLE" >&2; exit 1; }

mapfile -d '' EVENT_DIRS < <(
  find "$EVENT_ROOT" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z
)

((${#EVENT_DIRS[@]} > 0)) || { echo "no event directories found under: $EVENT_ROOT" >&2; exit 1; }

echo "found ${#EVENT_DIRS[@]} event dirs"

# =========================================
# 01: active channel
# =========================================
echo "[01] get_active_ch"
python proc/jma_model_dataset/01_get_active_ch.py \
  --target-fs-hz "$TARGET_FS_HZ" \
  --scan-rate-blocks "$SCAN_RATE_BLOCKS" \
  --skip-if-exists \
  "${EVENT_DIRS[@]}"

# =========================================
# 03: missing continuous target list
# =========================================
echo "[03] make_missing_continuous"
python proc/jma_model_dataset/03_make_missing_continuous.py \
  --meas-csv "$MEAS_CSV" \
  --epi-csv "$EPI_CSV" \
  --pres-csv "$PRES_CSV" \
  --mapping-report-csv "$MAPPING_REPORT_CSV" \
  --near0-suggest-csv "$NEAR0_SUGGEST_CSV" \
  --skip-if-exists \
  "${EVENT_DIRS[@]}"

# =========================================
# 04: download missing continuous
# missing file がある event だけ実行
# =========================================
STEP4_DIRS=()
for event_dir in "${EVENT_DIRS[@]}"; do
  if compgen -G "$event_dir/flows/jma_model_dataset/missing/*_missing_continuous.txt" > /dev/null; then
    STEP4_DIRS+=("$event_dir")
  fi
done

if ((${#STEP4_DIRS[@]} > 0)); then
  echo "[04] get_missing_continuous_waveform (${#STEP4_DIRS[@]} events)"
  python proc/jma_model_dataset/04_run_get_missing_continuous_waveform.py \
    --run-tag "$RUN_TAG" \
    --threads "$THREADS" \
    --cleanup \
    --skip-if-exists \
    --skip-if-done \
    "${STEP4_DIRS[@]}"
else
  echo "[04] skipped: no *_missing_continuous.txt found"
fi

# =========================================
# 05: fill to 48 stations
# =========================================
echo "[05] fill_to_48_stations"
python proc/jma_model_dataset/05_run_fill_to_48_stations.py \
  --pres-csv "$PRES_CSV" \
  --hinet-channel-table "$HINET_CHANNEL_TABLE" \
  --run-tag "$RUN_TAG" \
  --threads "$THREADS" \
  --cleanup \
  --skip-if-exists \
  --skip-if-done \
  "${EVENT_DIRS[@]}"

# =========================================
# 06: export 100 Hz
# =========================================
echo "[06] export_100hz"
python proc/jma_model_dataset/06_export_100hz.py \
  --epi-csv "$EPI_CSV" \
  --target-fs-hz "$TARGET_FS_HZ" \
  --skip-if-exists \
  "${EVENT_DIRS[@]}"

echo "done"
