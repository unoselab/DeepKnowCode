#!/usr/bin/env bash
# J-02a v3: public 5-field train; deterministic ja_ IDs; existing canonical untouched.
set -Eeuo pipefail
export TZ=America/Chicago
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT='j-02a'
VERSION='v3'
MODE="${1:---inspect}"
case "$MODE" in --inspect|--apply|--verify) ;; *) echo "USAGE: bash $0 [--inspect|--apply|--verify]" >&2; exit 2 ;; esac

# All configurable paths, budgets and logging options are defined in Shell.
CANONICAL_TRAIN="$ROOT/data/training/prepared/balanced_multilingual_canonical_v3/train.jsonl"
CANONICAL_VALIDATION="$ROOT/data/training/prepared/balanced_multilingual_canonical_v3/validation.jsonl"
TARGET_TRAIN="$ROOT/data/training/prepared/balanced_multilingual/train.jsonl"
MAPPING_FILE="$ROOT/data/training/prepared/balanced_multilingual_id_mapping_v1/java_ids.jsonl"
BACKUP_DIR="$ROOT/data/training/prepared/backups"
SCRIPT="$ROOT/src/py3/j-02a-simplify-training-pairs.py"
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
LOG_PREFIX='j-02a-v3'
LOG_TIMESTAMP_FORMAT='%Y-%m%d-%H%M%S'
LOG_TEE=true
LOG_CAPTURE_STDOUT=true
LOG_CAPTURE_STDERR=true
mkdir -p "$LOG_DIR" "$RESULTS_DIR"
STAMP="$(date +"$LOG_TIMESTAMP_FORMAT")-$$"
LOG="$(mktemp "$LOG_DIR/${LOG_PREFIX}-${STAMP}-XXXXXX.log")"
RUN="$(mktemp -d "$RESULTS_DIR/${STAMP}-XXXXXX")"
REPORT="$RUN/j-02a-public-train-report.json"
exec > >(tee -a "$LOG") 2>&1
now() { date +'%Y-%m-%dT%H:%M:%S%:z'; }
START_SEC="$(date +%s)"
START_CHICAGO="$(now)"
CURRENT_STEP=''
STEP_START=0
finish() {
  local rc=$? status='FAIL'
  if [[ $rc -eq 0 ]]; then status='PASS'; fi
  echo "[$(now)] $status FINAL experiment=$EXPERIMENT version=$VERSION mode=$MODE exit=$rc elapsed_sec=$(( $(date +%s)-START_SEC )) start=$START_CHICAGO end=$(now) log=$LOG run=$RUN report=$REPORT mapping=$MAPPING_FILE"
}
failed() {
  local rc=$?
  if [[ -n "$CURRENT_STEP" ]]; then echo "[$(now)] FAIL step=$CURRENT_STEP elapsed_sec=$(( $(date +%s)-STEP_START )) exit=$rc"; fi
  exit "$rc"
}
trap finish EXIT
trap failed ERR
step_start() { CURRENT_STEP="$1"; STEP_START="$(date +%s)"; echo "[$(now)] START step=$CURRENT_STEP"; }
step_pass() { echo "[$(now)] PASS step=$CURRENT_STEP elapsed_sec=$(( $(date +%s)-STEP_START ))"; CURRENT_STEP=''; }
echo "[$(now)] START experiment=$EXPERIMENT version=$VERSION mode=$MODE log=$LOG run=$RUN target=$TARGET_TRAIN"
step_start validate_files
for f in "$CANONICAL_TRAIN" "$CANONICAL_VALIDATION" "$TARGET_TRAIN" "$SCRIPT"; do
  [[ -f "$f" ]] || { echo "ERROR Missing file: $f" >&2; false; }
done
step_pass
step_start "${MODE#--}_train_format"
"$PYTHON_BIN" "$SCRIPT" \
  --mode "${MODE#--}" --canonical-train "$CANONICAL_TRAIN" \
  --canonical-validation "$CANONICAL_VALIDATION" --target-train "$TARGET_TRAIN" \
  --mapping-file "$MAPPING_FILE" --backup-dir "$BACKUP_DIR" --report "$REPORT" \
  --expected-train-total "$EXPECTED_TRAIN_TOTAL" \
  --expected-train-java-source "$EXPECTED_TRAIN_JAVA_SOURCE" \
  --expected-train-java-mixed "$EXPECTED_TRAIN_JAVA_MIXED" \
  --expected-train-python-mixed "$EXPECTED_TRAIN_PYTHON_MIXED" \
  --expected-validation-total "$EXPECTED_VALIDATION_TOTAL" \
  --expected-validation-java-source "$EXPECTED_VALIDATION_JAVA_SOURCE" \
  --expected-validation-java-mixed "$EXPECTED_VALIDATION_JAVA_MIXED" \
  --expected-validation-python-mixed "$EXPECTED_VALIDATION_PYTHON_MIXED"
step_pass
