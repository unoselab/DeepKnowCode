#!/usr/bin/env bash
# J-02a v2: standalone Java format-only canonicalization; no leakage audit.
set -Eeuo pipefail
export TZ=America/Chicago
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT='j-02a'
VERSION='v2'
MODE="${1:---inspect}"
case "$MODE" in --inspect|--build|--verify) ;; *) printf 'USAGE: bash %s [--inspect|--build|--verify]\n' "$0" >&2; exit 2;; esac

# All experiment configuration lives in this Shell file.
# Source files are expected to expose {"idx": string, "func": source-code string}.
# If the server holds them in data/training/raw/, edit BOTH values here; no fallback.
JAVA_SOURCE="$ROOT/data/training/raw/java_source/data.jsonl"
JAVA_MIXED="$ROOT/data/training/raw/java_mixed/data.jsonl"
INPUT_TRAIN="$ROOT/data/training/prepared/balanced_multilingual_canonical_v2/train.jsonl"
INPUT_VALIDATION="$ROOT/data/training/prepared/balanced_multilingual_canonical_v2/validation.jsonl"
OUTPUT="$ROOT/data/training/prepared/balanced_multilingual_canonical_v3"
SCRIPT="$ROOT/src/py3/j-02a-canonicalize-java-training.py"
PYTHON_BIN='python3'

EXPECTED_TRAIN_TOTAL=19464
EXPECTED_TRAIN_JAVA_SOURCE=12000
EXPECTED_TRAIN_JAVA_MIXED=3732
EXPECTED_TRAIN_PYTHON_MIXED=3732
EXPECTED_VALIDATION_TOTAL=512
EXPECTED_VALIDATION_JAVA_SOURCE=412
EXPECTED_VALIDATION_JAVA_MIXED=60
EXPECTED_VALIDATION_PYTHON_MIXED=40

LOG_DIR="$ROOT/logs/j-02a"
RESULTS_DIR="$ROOT/results/j-02a"
LOG_PREFIX='j-02a-v2'
LOG_TIMESTAMP_FORMAT='%Y-%m%d-%H%M%S'
LOG_CAPTURE_STDOUT=true
LOG_CAPTURE_STDERR=true
LOG_TEE=true
mkdir -p "$LOG_DIR" "$RESULTS_DIR"
STAMP="$(date +"$LOG_TIMESTAMP_FORMAT")-$$"
LOG="$(mktemp "$LOG_DIR/${LOG_PREFIX}-${STAMP}-XXXXXX.log")"
RUN="$(mktemp -d "$RESULTS_DIR/${STAMP}-XXXXXX")"
REPORT="$RUN/j-02a-java-format-report.json"

# Capture stdout + stderr; set -o pipefail preserves errors. The logfile is exclusive.
exec > >(tee -a "$LOG") 2>&1
now() { date +'%Y-%m-%dT%H:%M:%S%:z'; }
START_SEC="$(date +%s)"
START_CHICAGO="$(now)"
CURRENT_STEP=''
STEP_START=0
finish() {
  local rc=$? end duration state
  end="$(now)"
  duration=$(( $(date +%s) - START_SEC ))
  state='FAIL'; if [[ $rc -eq 0 ]]; then state='PASS'; fi
  printf '[%s] %s FINAL experiment=%s version=%s mode=%s elapsed_sec=%s exit=%s start=%s end=%s log=%s run=%s report=%s output=%s\n' \
    "$end" "$state" "$EXPERIMENT" "$VERSION" "$MODE" "$duration" "$rc" "$START_CHICAGO" "$end" "$LOG" "$RUN" "$REPORT" "$OUTPUT"
}
failed() {
  local rc=$?
  if [[ -n "$CURRENT_STEP" ]]; then
    printf '[%s] FAIL step=%s elapsed_sec=%s exit=%s\n' "$(now)" "$CURRENT_STEP" "$(( $(date +%s) - STEP_START ))" "$rc"
  fi
  exit "$rc"
}
trap finish EXIT
trap failed ERR
step_start() { CURRENT_STEP="$1"; STEP_START="$(date +%s)"; printf '[%s] START step=%s\n' "$(now)" "$CURRENT_STEP"; }
step_pass() { printf '[%s] PASS step=%s elapsed_sec=%s\n' "$(now)" "$CURRENT_STEP" "$(( $(date +%s) - STEP_START ))"; CURRENT_STEP=''; }
printf '[%s] START experiment=%s version=%s mode=%s log=%s run=%s output=%s\n' "$(now)" "$EXPERIMENT" "$VERSION" "$MODE" "$LOG" "$RUN" "$OUTPUT"

step_start 'validate_files'
for file in "$JAVA_SOURCE" "$JAVA_MIXED" "$INPUT_TRAIN" "$INPUT_VALIDATION" "$SCRIPT"; do
  if [[ ! -f "$file" ]]; then printf '[%s] ERROR missing file=%s\n' "$(now)" "$file" >&2; false; fi
done
step_pass

step_start "${MODE#--}_canonical_format"
"$PYTHON_BIN" "$SCRIPT" \
  --mode "${MODE#--}" \
  --java-source "$JAVA_SOURCE" --java-mixed "$JAVA_MIXED" \
  --train "$INPUT_TRAIN" --validation "$INPUT_VALIDATION" \
  --output "$OUTPUT" --report "$REPORT" \
  --expected-train-total "$EXPECTED_TRAIN_TOTAL" \
  --expected-train-java-source "$EXPECTED_TRAIN_JAVA_SOURCE" \
  --expected-train-java-mixed "$EXPECTED_TRAIN_JAVA_MIXED" \
  --expected-train-python-mixed "$EXPECTED_TRAIN_PYTHON_MIXED" \
  --expected-validation-total "$EXPECTED_VALIDATION_TOTAL" \
  --expected-validation-java-source "$EXPECTED_VALIDATION_JAVA_SOURCE" \
  --expected-validation-java-mixed "$EXPECTED_VALIDATION_JAVA_MIXED" \
  --expected-validation-python-mixed "$EXPECTED_VALIDATION_PYTHON_MIXED"
step_pass
