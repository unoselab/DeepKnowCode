#!/usr/bin/env bash
set -Eeuo pipefail
export TZ=America/Chicago
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT='j-01g'
VERSION='v6'
MODE="${1:-}"
if [[ $# -ne 1 || "$MODE" != '--audit' ]]; then
  echo "Usage: bash src/sh3/j-01g-audit-python-train-overlap-leakage.sh --audit" >&2
  exit 2
fi
PYTHON_BIN='python3'
SCRIPT="$ROOT/src/py3/j-01g-audit-python-train-overlap-leakage.py"
ORIGINAL_DIR="$ROOT/data/training/prepared/balanced_multilingual"
ORIGINAL_TRAIN="$ORIGINAL_DIR/train.jsonl"
ORIGINAL_VALIDATION="$ORIGINAL_DIR/validation.jsonl"
CANONICAL_DIR="$ROOT/data/training/prepared/balanced_multilingual_canonical_v2"
TRAIN="$CANONICAL_DIR/train.jsonl"
VALIDATION="$CANONICAL_DIR/validation.jsonl"
V1_DIR="$ROOT/data/evaluation/ground_truth/python/v1"
V2_DIR="$ROOT/data/evaluation/ground_truth/python/v2"
CSV_LEFT_FIELD='left.function_id'
CSV_RIGHT_FIELD='right.function_id'
CSV_LABEL_FIELD='label'
EXPECTED_TRAIN_ROWS=19464
EXPECTED_TRAIN_PYTHON_ROWS=3732
EXPECTED_VALIDATION_ROWS=512
EXPECTED_VALIDATION_PYTHON_ROWS=40
EXPECTED_V1_ROWS=7942
EXPECTED_V1_POSITIVE=3971
EXPECTED_V1_NEGATIVE=3971
EXPECTED_V2_ROWS=7320
EXPECTED_V2_POSITIVE=3660
EXPECTED_V2_NEGATIVE=3660
EXPECTED_CSV_FILES=24
MAX_EXAMPLES=10
LOG_DIR="$ROOT/logs/j-01g"
RESULT_DIR="$ROOT/results/j-01g"
LOG_PREFIX='j-01g-v6'
LOG_TIMESTAMP_FORMAT='+%Y-%m%d-%H%M%S'
LOG_TEE=true
LOG_STDERR=true
mkdir -p -- "$LOG_DIR" "$RESULT_DIR"
START_EPOCH="$(date +%s)"
START_CHICAGO="$(date -Iseconds)"
STAMP="$(date "$LOG_TIMESTAMP_FORMAT")"
RUN_DIR="$(mktemp -d "$RESULT_DIR/${STAMP}-XXXXXX")"
LOG_FILE="$LOG_DIR/${LOG_PREFIX}-${STAMP}-${BASHPID}-${RUN_DIR##*-}.log"
( set -o noclobber; : > "$LOG_FILE" )
exec > >(tee -a "$LOG_FILE") 2>&1
step_name='init'
step_start="$START_EPOCH"
start_step() { step_name="$1"; step_start="$(date +%s)"; printf '[%s] START step=%s\n' "$(date -Iseconds)" "$step_name"; }
pass_step() { printf '[%s] PASS step=%s elapsed_sec=%s\n' "$(date -Iseconds)" "$step_name" "$(( $(date +%s) - step_start ))"; }
finish() {
  local rc=$?
  trap - EXIT ERR
  if (( rc == 0 )); then
    printf '[%s] PASS FINAL experiment=%s version=%s mode=%s elapsed_sec=%s exit=%s\n' "$(date -Iseconds)" "$EXPERIMENT" "$VERSION" "$MODE" "$(( $(date +%s) - START_EPOCH ))" "$rc"
  else
    printf '[%s] FAIL step=%s elapsed_sec=%s\n' "$(date -Iseconds)" "$step_name" "$(( $(date +%s) - step_start ))"
    printf '[%s] FAIL FINAL experiment=%s version=%s mode=%s elapsed_sec=%s exit=%s\n' "$(date -Iseconds)" "$EXPERIMENT" "$VERSION" "$MODE" "$(( $(date +%s) - START_EPOCH ))" "$rc"
  fi
  printf "LOG='%s' RUN='%s' REPORT='%s'\n" "$LOG_FILE" "$RUN_DIR" "$REPORT_FILE"
  printf "START_CHICAGO='%s' END_CHICAGO='%s'\n" "$START_CHICAGO" "$(date -Iseconds)"
  exit "$rc"
}
REPORT_FILE="$RUN_DIR/j-01g-training-evaluation-leakage-report.json"
trap finish EXIT
printf '[%s] START experiment=%s version=%s mode=%s log=%s run=%s\n' "$(date -Iseconds)" "$EXPERIMENT" "$VERSION" "$MODE" "$LOG_FILE" "$RUN_DIR"
start_step validate_files
for file in "$SCRIPT" "$TRAIN" "$VALIDATION" "$ORIGINAL_TRAIN" "$ORIGINAL_VALIDATION"; do
  [[ -f "$file" ]] || { echo "MISSING_FILE=$file" >&2; exit 2; }
done
for dir in "$V1_DIR" "$V2_DIR"; do
  [[ -d "$dir" ]] || { echo "MISSING_DIRECTORY=$dir" >&2; exit 2; }
done
pass_step
start_step audit
"$PYTHON_BIN" "$SCRIPT" \
  --version "$VERSION" --train "$TRAIN" --validation "$VALIDATION" \
  --original-train "$ORIGINAL_TRAIN" --original-validation "$ORIGINAL_VALIDATION" \
  --v1-dir "$V1_DIR" --v2-dir "$V2_DIR" --output "$REPORT_FILE" \
  --csv-left-field "$CSV_LEFT_FIELD" --csv-right-field "$CSV_RIGHT_FIELD" --csv-label-field "$CSV_LABEL_FIELD" \
  --expected-train-rows "$EXPECTED_TRAIN_ROWS" --expected-train-python-rows "$EXPECTED_TRAIN_PYTHON_ROWS" \
  --expected-validation-rows "$EXPECTED_VALIDATION_ROWS" --expected-validation-python-rows "$EXPECTED_VALIDATION_PYTHON_ROWS" \
  --expected-v1-rows "$EXPECTED_V1_ROWS" --expected-v1-positive "$EXPECTED_V1_POSITIVE" --expected-v1-negative "$EXPECTED_V1_NEGATIVE" \
  --expected-v2-rows "$EXPECTED_V2_ROWS" --expected-v2-positive "$EXPECTED_V2_POSITIVE" --expected-v2-negative "$EXPECTED_V2_NEGATIVE" \
  --expected-csv-files "$EXPECTED_CSV_FILES" --max-examples "$MAX_EXAMPLES"
pass_step
