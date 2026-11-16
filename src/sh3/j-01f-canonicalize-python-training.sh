#!/usr/bin/env bash
set -Eeuo pipefail
export TZ=America/Chicago
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT="j-01f"
VERSION="v1"
MODE="${1:---inspect}"
case "$MODE" in --inspect|--build) ;; *) echo "USAGE: bash $0 [--inspect|--build]" >&2; exit 2;; esac
CLEAN="$ROOT/data/training/clean/python_mixed/data.jsonl"
TRAIN="$ROOT/data/training/prepared/balanced_multilingual/train.jsonl"
VALIDATION="$ROOT/data/training/prepared/balanced_multilingual/validation.jsonl"
OUTPUT="$ROOT/data/training/prepared/balanced_multilingual_canonical_v1"
SCRIPT="$ROOT/src/py3/j-01f-canonicalize-python-training.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_DIR="$ROOT/logs/j-01f"
LOG_PREFIX="j-01f-v1"
LOG_TIMESTAMP_FORMAT="%Y-%m%d-%H%M%S"
LOG_CAPTURE_STDOUT=true
LOG_CAPTURE_STDERR=true
LOG_TEE=true
mkdir -p "$LOG_DIR"
STAMP="$(date +"$LOG_TIMESTAMP_FORMAT")-$$-${RANDOM}"
LOG="$LOG_DIR/$LOG_PREFIX-$STAMP.log"
if [[ -e "$LOG" ]]; then echo "Log collision: $LOG" >&2; exit 2; fi
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
now(){ date +'%Y-%m-%dT%H:%M:%S%:z'; }
START_SECS="$(date +%s)"
START_CHICAGO="$(now)"
STATUS="FAIL"
finish(){ local rc=$?; local end elapsed; end="$(now)"; elapsed=$(( $(date +%s) - START_SECS )); if [[ $rc -eq 0 ]];then STATUS=PASS; fi; printf '[%s] %s FINAL experiment=%s version=%s mode=%s elapsed_sec=%s exit=%s start=%s end=%s log=%s output=%s\n' "$end" "$STATUS" "$EXPERIMENT" "$VERSION" "$MODE" "$elapsed" "$rc" "$START_CHICAGO" "$end" "$LOG" "$OUTPUT"; }
trap finish EXIT
printf '[%s] START experiment=%s version=%s mode=%s log=%s output=%s\n' "$(now)" "$EXPERIMENT" "$VERSION" "$MODE" "$LOG" "$OUTPUT"
STEP_START="$(date +%s)"; echo "[$(now)] START step=validate_files"
for file in "$CLEAN" "$TRAIN" "$VALIDATION" "$SCRIPT";do if [[ ! -f "$file" ]];then echo "Missing $file" >&2; exit 1;fi;done
echo "[$(now)] PASS step=validate_files elapsed_sec=$(( $(date +%s) - STEP_START ))"
STEP_START="$(date +%s)";echo "[$(now)] START step=$MODE"
"$PYTHON_BIN" "$SCRIPT" --mode "${MODE#--}" --clean "$CLEAN" --train "$TRAIN" --validation "$VALIDATION" --output "$OUTPUT"
echo "[$(now)] PASS step=$MODE elapsed_sec=$(( $(date +%s) - STEP_START ))"
