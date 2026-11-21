#!/usr/bin/env bash
# J-04d v4: J04c verified AFTER -> positive_pairs.csv + fragments.jsonl.
# Steps 1-3 ONLY. J-04e handles types/selection; J-04f handles copy-result.
# Standard execution sequence (always document these four commands):
# 1. Input validation
# bash src/sh3/j-04d-positive-clones.sh --inspect
# 2. Smoke test: pallets__werkzeug
# bash src/sh3/j-04d-positive-clones.sh --smoke
# 3. Full run: all 24 subjects
# bash src/sh3/j-04d-positive-clones.sh --all
# 4. Full result verification
# bash src/sh3/j-04d-positive-clones.sh --verify
# Run in order. Do not publish results here; J-04f owns --copy-result.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
EXPERIMENT="j-04d"
VERSION="v4"
INPUT_EXPERIMENT="j-04c"
INPUT_VERSION="v2"
SOURCE_DATA_VERSION="v8"
INPUT_ROOT="$ROOT/data/clone-detection/python/2after-preproc"
RESULT_ROOT="$ROOT/results/j-04d"
PYTHON_BIN="$HOME/miniconda3/envs/ast-py314/bin/python"
EXPECTED_PYTHON_VERSION="3.14"
PYTHON_SCRIPT="$ROOT/src/py3/j-04d-positive-clones.py"
EXPECTED_SUBJECTS=24
EXPECTED_FILES=22746
EXPECTED_PAIRS=11141
POSITIVE_LABEL=1
LANGUAGE="python"
CODE_ENCODING="utf-8"
CODE_ERRORS="surrogateescape"
COPY_MANIFEST_NAME="j04c-copy-manifest.json"
BATCH_SUMMARY_NAME="batch-summary.json"
SOURCE_RUN_MANIFEST_NAME="j04c-run.json"
SOURCE_PAIR_CSV_NAME="j04c_function_clone_pairs.csv"
SOURCE_HYBRID_SUMMARY_NAME="hybrid/summary.json"
INPUT_INVENTORY_NAME="input_inventory.csv"
STAGE_DIR_NAME="stage"
ALLOWED_DETECTOR_SOURCES="nicad_primary,jscpd_ast_fallback"
SMOKE_SUBJECT="pallets__werkzeug"
PAIR_PROGRESS_EVERY=1000
HASH_PROGRESS_EVERY=5000
SOURCE_CACHE_FILES=128
LOG_DIR="$ROOT/logs/j-04d"
LOG_PREFIX="j-04d-v4-positive-clones"
LOG_TIMESTAMP_FORMAT="%Y%m%d-%H%M%S-%N"
LOG_TIMEZONE="America/Chicago"
LOG_TEE_APPEND="false"
LOG_CAPTURE_STDOUT="true"
LOG_CAPTURE_STDERR="true"
LOG_STEP_LOGGING="true"
LOG_PRESERVE_EXIT_CODE="true"
export TZ="$LOG_TIMEZONE"
export LC_ALL=C

usage() {
  echo "Usage: bash src/sh3/j-04d-positive-clones.sh --inspect|--smoke|--all|--verify|--subject SUBJECT" >&2
  exit 2
}
(( $# >= 1 )) || usage
MODE="$1"; shift
SUBJECT="ALL"
case "$MODE" in
  --inspect|--all|--verify) (( $# == 0 )) || usage ;;
  --smoke) (( $# == 0 )) || usage; SUBJECT="$SMOKE_SUBJECT" ;;
  --subject)
    (( $# == 1 )) || usage
    SUBJECT="$1"
    [[ "$SUBJECT" =~ ^[a-z0-9][a-z0-9_.-]*__[a-z0-9][a-z0-9_.-]*$ ]] || usage ;;
  *) usage ;;
esac
MODE_NAME="${MODE#--}"
[[ "$MODE_NAME" != subject ]] || MODE_NAME="subject"
[[ -f "$PYTHON_SCRIPT" ]] || { echo "FAIL missing Python script: $PYTHON_SCRIPT" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "FAIL Python runtime not executable: $PYTHON_BIN" >&2; exit 2; }
command -v tee >/dev/null 2>&1 || { echo "FAIL tee missing" >&2; exit 2; }
mkdir -p "$LOG_DIR" "$RESULT_ROOT"
RUN_ID="$(date "+$LOG_TIMESTAMP_FORMAT")-pid$$-${MODE_NAME}-$(od -An -tx1 -N5 /dev/urandom | tr -d ' \n')"
RUN_DIR="$RESULT_ROOT/$RUN_ID"
LOG_FILE="$LOG_DIR/$LOG_PREFIX-$RUN_ID.log"
( set -C; : > "$LOG_FILE" ) || { echo "FAIL log path collision: $LOG_FILE" >&2; exit 2; }
exec > >(tee "$LOG_FILE") 2>&1
now() { date '+%Y-%m-%dT%H:%M:%S%:z'; }
START_TIME="$(now)"
START_EPOCH="$(date +%s)"
CURRENT_STEP="init"
FINAL_STATUS="PASS"
finish() {
  local code="$?"
  trap - EXIT
  if (( code != 0 )); then
    FINAL_STATUS="FAIL"
    if [[ -n "${STEP_EPOCH:-}" ]]; then
      echo "[$(now)] FAIL step=$CURRENT_STEP elapsed_sec=$(($(date +%s)-STEP_EPOCH)) exit=$code"
    fi
  fi
  echo "[$(now)] FINAL experiment=$EXPERIMENT version=$VERSION mode=$MODE_NAME subject=$SUBJECT status=$FINAL_STATUS exit=$code start=$START_TIME end=$(now) elapsed_sec=$(($(date +%s)-START_EPOCH)) log=$LOG_FILE result=$RUN_DIR input=$INPUT_ROOT"
  exit "$code"
}
trap finish EXIT
trap 'exit 130' INT TERM
step_start() { CURRENT_STEP="$1"; STEP_EPOCH="$(date +%s)"; echo "[$(now)] START step=$CURRENT_STEP"; }
step_pass() { echo "[$(now)] PASS step=$CURRENT_STEP elapsed_sec=$(($(date +%s)-STEP_EPOCH))"; }
step_start preflight
[[ -d "$INPUT_ROOT" ]] || { echo "FAIL input missing: $INPUT_ROOT" >&2; exit 2; }
"$PYTHON_BIN" -c 'import sys; assert f"{sys.version_info.major}.{sys.version_info.minor}" == sys.argv[1],sys.version' "$EXPECTED_PYTHON_VERSION" \
  || { echo "FAIL Python 3.14 runtime required" >&2; exit 2; }
step_pass

echo "[$(now)] RUN experiment=$EXPERIMENT version=$VERSION mode=$MODE_NAME run_id=$RUN_ID timezone=$LOG_TIMEZONE log=$LOG_FILE result=$RUN_DIR"
echo "[$(now)] CONFIG input=$INPUT_ROOT subjects=$EXPECTED_SUBJECTS files=$EXPECTED_FILES hybrid_pairs=$EXPECTED_PAIRS python=$PYTHON_BIN positive_label=$POSITIVE_LABEL code_encoding=$CODE_ENCODING code_errors=$CODE_ERRORS detector_sources=$ALLOWED_DETECTOR_SOURCES"
step_start "$MODE_NAME"
"$PYTHON_BIN" -u "$PYTHON_SCRIPT" \
  --experiment "$EXPERIMENT" \
  --version "$VERSION" \
  --input-experiment "$INPUT_EXPERIMENT" \
  --input-version "$INPUT_VERSION" \
  --source-data-version "$SOURCE_DATA_VERSION" \
  --expected-python-version "$EXPECTED_PYTHON_VERSION" \
  --run-id "$RUN_ID" \
  --mode "$MODE_NAME" \
  --subject "$SUBJECT" \
  --timezone "$LOG_TIMEZONE" \
  --language "$LANGUAGE" \
  --code-encoding "$CODE_ENCODING" \
  --code-errors "$CODE_ERRORS" \
  --copy-manifest-name "$COPY_MANIFEST_NAME" \
  --batch-summary-name "$BATCH_SUMMARY_NAME" \
  --source-run-manifest-name "$SOURCE_RUN_MANIFEST_NAME" \
  --source-pair-csv-name "$SOURCE_PAIR_CSV_NAME" \
  --source-hybrid-summary-name "$SOURCE_HYBRID_SUMMARY_NAME" \
  --input-inventory-name "$INPUT_INVENTORY_NAME" \
  --stage-dir-name "$STAGE_DIR_NAME" \
  --allowed-detector-sources "$ALLOWED_DETECTOR_SOURCES" \
  --input-root "$INPUT_ROOT" \
  --result-root "$RESULT_ROOT" \
  --run-dir "$RUN_DIR" \
  --expected-subjects "$EXPECTED_SUBJECTS" \
  --expected-files "$EXPECTED_FILES" \
  --expected-pairs "$EXPECTED_PAIRS" \
  --positive-label "$POSITIVE_LABEL" \
  --pair-progress-every "$PAIR_PROGRESS_EVERY" \
  --hash-progress-every "$HASH_PROGRESS_EVERY" \
  --source-cache-files "$SOURCE_CACHE_FILES"
step_pass
