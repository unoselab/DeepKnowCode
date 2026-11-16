#!/usr/bin/env bash
set -Eeuo pipefail
export TZ='America/Chicago'
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXPERIMENT='j-01e'
SCRIPT_VERSION='v1'
LOG_DIR="$ROOT/logs/$EXPERIMENT"
RESULT_DIR="$ROOT/results/$EXPERIMENT"
LOG_PREFIX="$EXPERIMENT-$SCRIPT_VERSION"
LOG_TIMESTAMP_FORMAT='+%Y-%m%d-%H%M%S'
LOG_TEE='true'
LOG_STDERR='true'
V1_DIR="$ROOT/data/evaluation/ground_truth/python/v1"
V2_DIR="$ROOT/data/evaluation/ground_truth/python/v2"
TRAIN_FILE="$ROOT/data/training/prepared/balanced_multilingual/train.jsonl"
OUTPUT_POLICY='report-only'
DATASET_VERSIONS_JSON='["v1","v2"]'
REPORT_FILENAME='j-01e-training-overlap-report.json'
usage() { echo "Usage: bash src/sh3/j-01e-audit-python-train-overlap.sh --audit" >&2; exit 2; }
[[ "$#" -eq 1 ]] || usage
MODE="$1"
case "$MODE" in --audit) ;; *) usage ;; esac
mkdir -p "$LOG_DIR" "$RESULT_DIR"
START_EPOCH="$(date +%s)"
START_CHICAGO="$(date -Iseconds)"
STAMP="$(date "$LOG_TIMESTAMP_FORMAT")"
RUN_DIR="$(mktemp -d "$RESULT_DIR/${STAMP}-XXXXXX")"
LOG_FILE="$LOG_DIR/$LOG_PREFIX-$STAMP-${BASHPID}-${RUN_DIR##*-}.log"
( set -o noclobber; : > "$LOG_FILE" )
exec > >(tee -a "$LOG_FILE") 2>&1
step_name='initialization';step_start="$START_EPOCH";finished=0
log() { printf '[%s] %s\n' "$(date -Iseconds)" "$*"; }
start_step() { step_name="$1";step_start="$(date +%s)";log "START step=$step_name"; }
pass_step() { log "PASS step=$step_name elapsed_sec=$(($(date +%s)-step_start))"; }
finish() {
 local ec=$?
 if [[ $finished -eq 1 ]]; then return; fi
 finished=1
 local status='PASS'
 if [[ "$ec" -ne 0 ]]; then status='FAIL';log "FAIL step=$step_name elapsed_sec=$(($(date +%s)-step_start))"; fi
 log "$status FINAL experiment=$EXPERIMENT version=$SCRIPT_VERSION mode=$MODE elapsed_sec=$(($(date +%s)-START_EPOCH)) exit=$ec"
 printf "LOG='%s' RUN='%s' START_CHICAGO='%s' END_CHICAGO='%s'\n" "$LOG_FILE" "$RUN_DIR" "$START_CHICAGO" "$(date -Iseconds)"
}
trap finish EXIT
log "START experiment=$EXPERIMENT version=$SCRIPT_VERSION mode=$MODE log=$LOG_FILE run=$RUN_DIR"
start_step validate_files
[[ -f "$ROOT/src/py3/j-01e-audit-python-train-overlap.py" ]] || { log 'FAIL missing Python script';exit 1; }
[[ -f "$TRAIN_FILE" ]] || { log "FAIL missing train=$TRAIN_FILE";exit 1; }
pass_step
start_step "$MODE"
"$PYTHON_BIN" "$ROOT/src/py3/j-01e-audit-python-train-overlap.py" \
 --v1-dir "$V1_DIR" --v2-dir "$V2_DIR" --train-file "$TRAIN_FILE" --run-dir "$RUN_DIR" \
 --report-name "$REPORT_FILENAME" --experiment "$EXPERIMENT" \
 --version "$SCRIPT_VERSION" --output-policy "$OUTPUT_POLICY" \
 --expected-versions-json "$DATASET_VERSIONS_JSON"
pass_step
