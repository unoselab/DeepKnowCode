#!/usr/bin/env bash
# DeepKnowCode J-04e v1 — Python AST Type Census + Fixed Positive Dataset.
# Standalone Shell: all experiment parameters configured here; required Python CLI.
# Execution guide (run each step from the DeepKnowCode repository root):
#
# 1. Validate exact J-04d v4 ALL output and final verification receipt.
# bash src/sh3/j-04e-positive-clonetypes.sh --inspect
#
# 2. Smoke Test: pallets__werkzeug; Type Census, dedup, public/details verify.
# bash src/sh3/j-04e-positive-clonetypes.sh --smoke
#
# 3. Process all 24 subjects (11,141 J-04d candidate pairs).
# bash src/sh3/j-04e-positive-clonetypes.sh --all
#
# 4. Independently reclassify, re-deduplicate, and verify outputs.
# bash src/sh3/j-04e-positive-clonetypes.sh --verify
#
# Optional single-subject run:
# bash src/sh3/j-04e-positive-clonetypes.sh --subject pallets__werkzeug
#
# --copy-result is deliberately NOT supported; J-04f publishes to 3pos-clones.
# Do not modify J-04d outputs, J-04c data, or existing versioned artifacts.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
EXPERIMENT="j-04e"
VERSION="v1"
SOURCE_EXPERIMENT="j-04d"
SOURCE_VERSION="v4"
SOURCE_INPUT_VERSION="v2"
SOURCE_ALL_RUN_ID="20261120-233947-781428288-pid2627860-all-a4b45503f7"
SOURCE_VERIFY_RUN_ID="20261121-000212-461266246-pid2628670-verify-679064454e"
SOURCE_ALL_ROOT="$ROOT/results/j-04d/$SOURCE_ALL_RUN_ID"
SOURCE_VERIFY_ROOT="$ROOT/results/j-04d/$SOURCE_VERIFY_RUN_ID"
SOURCE_BATCH_NAME="batch-summary.json"
SOURCE_VERIFICATION_NAME="verification-all.json"
EXPECTED_SUBJECTS=24
EXPECTED_INPUT_FILES=22746
EXPECTED_PAIRS=11141
EXPECTED_FRAGMENTS=11959
SMOKE_SUBJECT="pallets__werkzeug"
RESULT_ROOT="$ROOT/results/j-04e"
PYTHON_BIN="$HOME/miniconda3/envs/ast-py314/bin/python"
PYTHON_SCRIPT="$ROOT/src/py3/j-04e-positive-clonetypes.py"
EXPECTED_PYTHON_VERSION="3.14"
LANGUAGE="python"
CODE_ENCODING="utf-8"
CODE_ERRORS="surrogateescape"
POSITIVE_LABEL=1
ALLOWED_DETECTORS="nicad_primary,jscpd_ast_fallback"
AST_PRIMARY_THRESHOLD="0.82"
AST_SENSITIVITY_THRESHOLDS="0.70,0.75,0.80,0.82,0.85,0.90"
AST_GRAM_SIZE=3
AST_MAX_CELLS=2000000
SELECTION_POLICY="all_available_types"
DEDUP_POLICY="unordered_source_sha256_pair"
SAMPLE_SEED=42
PAIR_PROGRESS_EVERY=1000
FRAGMENT_PROGRESS_EVERY=2000
EXAMPLES_PER_CATEGORY=3
GPU_CONFIGURATION="none"
LOG_DIR="$ROOT/logs/j-04e"
LOG_PREFIX="j-04e-v1-positive-clonetypes"
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
  echo "Usage: bash src/sh3/j-04e-positive-clonetypes.sh --inspect|--smoke|--all|--verify|--subject SUBJECT" >&2
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
  echo "[$(now)] FINAL experiment=$EXPERIMENT version=$VERSION mode=$MODE_NAME subject=$SUBJECT status=$FINAL_STATUS exit=$code start=$START_TIME end=$(now) elapsed_sec=$(($(date +%s)-START_EPOCH)) log=$LOG_FILE result=$RUN_DIR source_all=$SOURCE_ALL_ROOT source_verify=$SOURCE_VERIFY_ROOT"
  exit "$code"
}
trap finish EXIT
trap 'exit 130' INT TERM
step_start() { CURRENT_STEP="$1"; STEP_EPOCH="$(date +%s)"; echo "[$(now)] START step=$CURRENT_STEP"; }
step_pass() { echo "[$(now)] PASS step=$CURRENT_STEP elapsed_sec=$(($(date +%s)-STEP_EPOCH))"; }
step_start preflight
[[ -d "$SOURCE_ALL_ROOT" ]] || { echo "FAIL J04D ALL input missing: $SOURCE_ALL_ROOT" >&2; exit 2; }
[[ -d "$SOURCE_VERIFY_ROOT" ]] || { echo "FAIL J04D verify input missing: $SOURCE_VERIFY_ROOT" >&2; exit 2; }
[[ -f "$SOURCE_ALL_ROOT/$SOURCE_BATCH_NAME" ]] || { echo "FAIL J04D batch summary missing: $SOURCE_ALL_ROOT/$SOURCE_BATCH_NAME" >&2; exit 2; }
[[ -f "$SOURCE_VERIFY_ROOT/$SOURCE_VERIFICATION_NAME" ]] || { echo "FAIL J04D verification receipt missing: $SOURCE_VERIFY_ROOT/$SOURCE_VERIFICATION_NAME" >&2; exit 2; }
"$PYTHON_BIN" -c 'import sys; assert f"{sys.version_info.major}.{sys.version_info.minor}" == sys.argv[1],sys.version' "$EXPECTED_PYTHON_VERSION" \
  || { echo "FAIL required Python version: $EXPECTED_PYTHON_VERSION" >&2; exit 2; }
step_pass

echo "[$(now)] RUN experiment=$EXPERIMENT version=$VERSION mode=$MODE_NAME run_id=$RUN_ID timezone=$LOG_TIMEZONE log=$LOG_FILE result=$RUN_DIR"
echo "[$(now)] CONFIG source_experiment=$SOURCE_EXPERIMENT source_version=$SOURCE_VERSION source_input_version=$SOURCE_INPUT_VERSION source_all=$SOURCE_ALL_ROOT source_verify=$SOURCE_VERIFY_ROOT subjects=$EXPECTED_SUBJECTS input_files=$EXPECTED_INPUT_FILES candidate_pairs=$EXPECTED_PAIRS fragments=$EXPECTED_FRAGMENTS python=$PYTHON_BIN threshold=$AST_PRIMARY_THRESHOLD gram=$AST_GRAM_SIZE max_cells=$AST_MAX_CELLS dedup=$DEDUP_POLICY selection=$SELECTION_POLICY seed=$SAMPLE_SEED gpu=$GPU_CONFIGURATION"
step_start "$MODE_NAME"
"$PYTHON_BIN" -u "$PYTHON_SCRIPT" \
  --experiment "$EXPERIMENT" \
  --version "$VERSION" \
  --source-experiment "$SOURCE_EXPERIMENT" \
  --source-version "$SOURCE_VERSION" \
  --source-input-version "$SOURCE_INPUT_VERSION" \
  --source-all-run-id "$SOURCE_ALL_RUN_ID" \
  --source-verify-run-id "$SOURCE_VERIFY_RUN_ID" \
  --source-all-root "$SOURCE_ALL_ROOT" \
  --source-verify-root "$SOURCE_VERIFY_ROOT" \
  --source-batch-name "$SOURCE_BATCH_NAME" \
  --source-verification-name "$SOURCE_VERIFICATION_NAME" \
  --result-root "$RESULT_ROOT" \
  --run-dir "$RUN_DIR" \
  --run-id "$RUN_ID" \
  --mode "$MODE_NAME" \
  --subject "$SUBJECT" \
  --smoke-subject "$SMOKE_SUBJECT" \
  --timezone "$LOG_TIMEZONE" \
  --language "$LANGUAGE" \
  --code-encoding "$CODE_ENCODING" \
  --code-errors "$CODE_ERRORS" \
  --expected-python-version "$EXPECTED_PYTHON_VERSION" \
  --expected-subjects "$EXPECTED_SUBJECTS" \
  --expected-input-files "$EXPECTED_INPUT_FILES" \
  --expected-pairs "$EXPECTED_PAIRS" \
  --expected-fragments "$EXPECTED_FRAGMENTS" \
  --positive-label "$POSITIVE_LABEL" \
  --allowed-detectors "$ALLOWED_DETECTORS" \
  --primary-threshold "$AST_PRIMARY_THRESHOLD" \
  --sensitivity-thresholds "$AST_SENSITIVITY_THRESHOLDS" \
  --ast-gram-size "$AST_GRAM_SIZE" \
  --ast-max-cells "$AST_MAX_CELLS" \
  --selection-policy "$SELECTION_POLICY" \
  --dedup-policy "$DEDUP_POLICY" \
  --sample-seed "$SAMPLE_SEED" \
  --pair-progress-every "$PAIR_PROGRESS_EVERY" \
  --fragment-progress-every "$FRAGMENT_PROGRESS_EVERY" \
  --examples-per-category "$EXAMPLES_PER_CATEGORY"
step_pass
