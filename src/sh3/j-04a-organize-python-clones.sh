#!/usr/bin/env bash
# DeepKnowCode J-04a v1 - Python C-01 batch copy-only organization.
set -Eeuo pipefail

# All runtime parameters are defined here; the Python implementation has no
# experiment-specific defaults. Server script filenames intentionally have no version suffix.
EXPERIMENT='j-04a'
SCRIPT_VERSION='v1'
MODE=''  # Required CLI choice: --inspect | --dry-run | --copy | --verify
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PYTHON_BIN='python3'
PYTHON_SCRIPT="$ROOT/src/py3/j-04a-organize-python-clones.py"
SOURCE_ROOT="$ROOT/results/c-01-batch"
PYTHON_ROOT="$ROOT/data/clone-detection/python"
EXPECTED_SUBJECTS='24'
ORGANIZER_NAME='organizer-20261103-013600-284803.json'
BATCH_NAME='20261102-150146-232146'
LOG_DIR="$ROOT/logs/j-04a"
LOG_PREFIX='j-04a-v1-python-clone-organization'
LOG_TIMESTAMP_FORMAT='+%Y%m%d-%H%M%S-%N'
LOG_TZ='America/Chicago'
LOG_CAPTURE_STDOUT='true'
LOG_CAPTURE_STDERR='true'
LOG_TEE_TERMINAL='true'

usage() {
  printf 'Usage: bash src/sh3/j-04a-organize-python-clones.sh {--inspect|--dry-run|--copy|--verify}\n' >&2
}
if [[ $# -ne 1 ]]; then usage; exit 2; fi
case "$1" in
  --inspect) MODE='inspect';;
  --dry-run) MODE='dry-run';;
  --copy) MODE='copy';;
  --verify) MODE='verify';;
  *) usage; exit 2;;
esac

export TZ="$LOG_TZ"
mkdir -p -- "$LOG_DIR"
START_EPOCH="$(date +%s)"
START_CHICAGO="$(date '+%Y-%m-%dT%H:%M:%S%z')"
STAMP="$(date "$LOG_TIMESTAMP_FORMAT")"
LOG_FILE="$(mktemp "$LOG_DIR/${LOG_PREFIX}-${STAMP}-pid$$-${MODE}-XXXXXXXX.log")"
readonly LOG_FILE
exec > >(tee -a -- "$LOG_FILE") 2>&1
now() { date '+%Y-%m-%dT%H:%M:%S%z'; }
FINAL_STATUS='FAIL'
STEP='initialization'
STEP_EPOCH="$START_EPOCH"
step_start() {
  STEP="$1"; STEP_EPOCH="$(date +%s)"
  printf '[%s] START step=%s\n' "$(now)" "$STEP"
}
step_pass() {
  printf '[%s] PASS step=%s elapsed_sec=%d\n' "$(now)" "$STEP" "$(( $(date +%s) - STEP_EPOCH ))"
}
finalize() {
  local rc=$?
  trap - EXIT
  if [[ $rc -eq 0 ]]; then FINAL_STATUS='PASS'; else
    printf '[%s] FAIL step=%s exit=%s elapsed_sec=%d\n' "$(now)" "$STEP" "$rc" "$(( $(date +%s) - STEP_EPOCH ))"
  fi
  printf '[%s] FINAL experiment=%s script_version=%s mode=%s status=%s exit=%s start=%s end=%s elapsed_sec=%d log=%s source=%s destination=%s\n' \
    "$(now)" "$EXPERIMENT" "$SCRIPT_VERSION" "$MODE" "$FINAL_STATUS" "$rc" "$START_CHICAGO" "$(now)" \
    "$(( $(date +%s) - START_EPOCH ))" "$LOG_FILE" "$SOURCE_ROOT" "$PYTHON_ROOT"
  exit "$rc"
}
trap finalize EXIT

printf '[%s] RUN experiment=%s version=%s mode=%s timezone=%s root=%s log=%s\n' \
  "$(now)" "$EXPERIMENT" "$SCRIPT_VERSION" "$MODE" "$LOG_TZ" "$ROOT" "$LOG_FILE"
step_start 'preflight'
[[ -f "$PYTHON_SCRIPT" && ! -L "$PYTHON_SCRIPT" ]] || { echo "Missing/unsafe Python script: $PYTHON_SCRIPT"; exit 2; }
[[ -d "$SOURCE_ROOT" && ! -L "$SOURCE_ROOT" ]] || { echo "Missing/unsafe source: $SOURCE_ROOT"; exit 2; }
[[ -d "$PYTHON_ROOT" && ! -L "$PYTHON_ROOT" ]] || { echo "Missing/unsafe target root: $PYTHON_ROOT"; exit 2; }
command -v "$PYTHON_BIN" >/dev/null || { echo "Python not available: $PYTHON_BIN"; exit 2; }
step_pass

step_start "$MODE"
"$PYTHON_BIN" -u "$PYTHON_SCRIPT" \
  --source-root "$SOURCE_ROOT" \
  --python-root "$PYTHON_ROOT" \
  --expected-subjects "$EXPECTED_SUBJECTS" \
  --organizer-name "$ORGANIZER_NAME" \
  --batch-name "$BATCH_NAME" \
  --mode "$MODE"
step_pass
