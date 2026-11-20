#!/usr/bin/env bash
# J-04c v2: standalone C-01-method NiCad + per-file audit + jscpd + hybrid clone detection.
# Input files are the immutable, verified J-04b v8 clone-input files (never re-masked).
# Unlike the C-01 wrapper, this script does NOT call another Shell wrapper.
set -Eeuo pipefail

# Shell is the sole experiment-configuration authority. Python has no experiment defaults.
EXPERIMENT='j-04c'
SCRIPT_VERSION='v2'
SOURCE_DATA_VERSION='v8'
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PYTHON_BIN="$HOME/miniconda3/envs/ast-py314/bin/python"
PYTHON_EXPECTED_VERSION='3.14.10'
PYTHON_SCRIPT="$ROOT/src/py3/j-04c-clone-detection.py"
DATASET_ROOT="$ROOT/data/repositories/python-repo-clone-input-v8-filtered"
INPUT_ROOT="$DATASET_ROOT/repositories"
INPUT_POLICY='native_ast_only_exclude_all_fallback'
EXPECTED_SUBJECTS=24
EXPECTED_FILES=22746
EXPECTED_UPSTREAM_INCLUDED=22774
EXPECTED_UNPARSED=28
PUBLISH_ROOT="$ROOT/data/clone-detection/python/2after-preproc"
SMOKE_SUBJECT='pallets__werkzeug'
NICAD_HOME="$ROOT/tools/nicad-7.0.1-linux-x86_64"
NICAD_BIN="$NICAD_HOME/bin/nicad"
TXL_BIN="$ROOT/tools/opentxl-11.3.7-linux-x64/bin/txl"
NICAD_GRANULARITY='functions'
NICAD_LANGUAGE='py'
NICAD_CONFIG='default'
NICAD_XML_PATTERN='*functions-blind-clones-0.30.xml'
JSCPD_LAUNCHER='npx'
JSCPD_PACKAGE='jscpd@5.4.2'
JSCPD_FORMAT='python'
JSCPD_SIMILARITY='0.70'
JSCPD_MIN_LINES=10
JSCPD_MIN_TOKENS=50
JSCPD_REPORTER='json'
# C-01 hybrid merge source pinned to the verified implementation blob from GitHub.
MERGE_SCRIPT="$ROOT/src/py/c-01-hybrid-merge.py"
MERGE_GIT_BLOB_SHA='26ca4d1e37388b8e9cc1c40597b8693ae0fed528'
NI_MAPPING_THRESHOLD='0.65'
STRICT_FUNCTION_FRACTION='0.90'
BOUNDARY_TOLERANCE_LINES=2
# 'all' is the original C-01 behavior; changing to 'on_bulk_failures' requires a new version.
PROBE_MODE='all'
PROBE_TIMEOUT_SEC=100
PROBE_PROGRESS_EVERY=25
COPY_PROGRESS_EVERY=5000
NICAD_TIMEOUT_SEC=36000
JSCPD_TIMEOUT_SEC=36000
MERGE_TIMEOUT_SEC=36000
VERIFY_INPUT_HASHES='true'
LOG_TZ='America/Chicago'
LOG_DIR="$ROOT/logs/j-04c"
LOG_PREFIX='j-04c-v2-clone-detection'
LOG_TIMESTAMP_FORMAT='+%Y%m%d-%H%M%S-%N'
LOG_CAPTURE_STDOUT='true'
LOG_CAPTURE_STDERR='true'
LOG_TEE_TERMINAL='true'
LOG_STEP_TIMING='true'
RESULT_ROOT="$ROOT/results/j-04c"
# Same lock as C-01: NiCad/OpenTXL shared installation is not concurrency-safe.
NICAD_SHARED_LOCK="$ROOT/logs/c-01/nicad.lock"

usage() {
  echo 'Usage: bash src/sh3/j-04c-clone-detection.sh {--inspect|--dry-run|--smoke|--subject owner__repo|--all|--verify [--subject owner__repo]|--copy-result}' >&2
  exit 2
}
(( $# >= 1 )) || usage
MODE=''; SUBJECT='ALL'
case "$1" in
  --inspect|--dry-run|--all|--copy-result)
    (( $# == 1 )) || usage
    MODE="${1#--}" ;;
  --smoke)
    (( $# == 1 )) || usage
    MODE='smoke'; SUBJECT="$SMOKE_SUBJECT" ;;
  --subject)
    (( $# == 2 )) || usage
    MODE='subject'; SUBJECT="$2" ;;
  --verify)
    MODE='verify'
    if (( $# == 3 )) && [[ "$2" == '--subject' ]]; then SUBJECT="$3"
    else (( $# == 1 )) || usage; fi ;;
  *) usage ;;
esac
if [[ "$SUBJECT" != 'ALL' ]]; then
  [[ "$SUBJECT" =~ ^[a-z0-9][a-z0-9_.-]*__[a-z0-9][a-z0-9_.-]*$ ]] || usage
fi

export TZ="$LOG_TZ"
export LC_ALL=C
mkdir -p -- "$LOG_DIR" "$RESULT_ROOT" "$(dirname "$NICAD_SHARED_LOCK")"
START_EPOCH="$(date +%s)"
START_CHICAGO="$(date '+%Y-%m-%dT%H:%M:%S%z')"
STAMP="$(date "$LOG_TIMESTAMP_FORMAT")"
RUN_ID="${STAMP}-pid$$"
LOG_FILE="$(mktemp "$LOG_DIR/${LOG_PREFIX}-${STAMP}-pid$$-${MODE}-XXXXXXXX.log")"
REPORT_DIR="$(mktemp -d "$RESULT_ROOT/${STAMP}-pid$$-${MODE}-XXXXXXXX")"
OUTPUT_ROOT="$REPORT_DIR/subjects"
readonly LOG_FILE REPORT_DIR RUN_ID OUTPUT_ROOT
exec > >(tee -a -- "$LOG_FILE") 2>&1
now() { date '+%Y-%m-%dT%H:%M:%S%z'; }
STEP='initialization'; STEP_EPOCH="$START_EPOCH"
step_start() { STEP="$1"; STEP_EPOCH="$(date +%s)"; printf '[%s] START step=%s\n' "$(now)" "$STEP"; }
step_pass() { printf '[%s] PASS step=%s elapsed_sec=%d\n' "$(now)" "$STEP" "$(( $(date +%s)-STEP_EPOCH ))"; }
finalize() {
  local rc=$?; trap - EXIT
  local status='FAIL'
  if (( rc == 0 )); then status='PASS'; else
    printf '[%s] FAIL step=%s exit=%d elapsed_sec=%d\n' "$(now)" "$STEP" "$rc" "$(( $(date +%s)-STEP_EPOCH ))"
  fi
  printf '[%s] FINAL experiment=%s version=%s source_data=%s mode=%s subject=%s status=%s exit=%d start=%s end=%s elapsed_sec=%d log=%s report=%s input=%s output=%s\n' \
    "$(now)" "$EXPERIMENT" "$SCRIPT_VERSION" "$SOURCE_DATA_VERSION" "$MODE" "$SUBJECT" "$status" "$rc" \
    "$START_CHICAGO" "$(now)" "$(( $(date +%s)-START_EPOCH ))" "$LOG_FILE" "$REPORT_DIR" "$INPUT_ROOT" "$PUBLISH_ROOT"
  exit "$rc"
}
trap finalize EXIT
printf '[%s] RUN experiment=%s version=%s mode=%s run_id=%s log=%s report=%s timezone=%s\n' \
  "$(now)" "$EXPERIMENT" "$SCRIPT_VERSION" "$MODE" "$RUN_ID" "$LOG_FILE" "$REPORT_DIR" "$LOG_TZ"
printf '[%s] CONFIG input=%s result_root=%s publish_root=%s subjects=%s files=%s ni=%s-%s-%s jscpd=%s similarity=%s min_lines=%s min_tokens=%s probe_mode=%s python=%s\n' \
  "$(now)" "$INPUT_ROOT" "$RESULT_ROOT" "$PUBLISH_ROOT" "$EXPECTED_SUBJECTS" "$EXPECTED_FILES" \
  "$NICAD_GRANULARITY" "$NICAD_LANGUAGE" "$NICAD_CONFIG" "$JSCPD_PACKAGE" \
  "$JSCPD_SIMILARITY" "$JSCPD_MIN_LINES" "$JSCPD_MIN_TOKENS" "$PROBE_MODE" "$PYTHON_BIN"

BASE_ARGS=(--experiment "$EXPERIMENT" --version "$SCRIPT_VERSION" \
  --source-data-version "$SOURCE_DATA_VERSION" --input-policy "$INPUT_POLICY" --log-timezone "$LOG_TZ" \
  --expected-python-version "$PYTHON_EXPECTED_VERSION" --root "$ROOT" \
  --dataset-root "$DATASET_ROOT" --input-root "$INPUT_ROOT" --output-root "$OUTPUT_ROOT" \
  --result-root "$RESULT_ROOT" --publish-root "$PUBLISH_ROOT" --batch-mode "$MODE" \
  --expected-subjects "$EXPECTED_SUBJECTS" --expected-files "$EXPECTED_FILES" \
  --expected-upstream-included "$EXPECTED_UPSTREAM_INCLUDED" --expected-unparsed "$EXPECTED_UNPARSED" \
  --nicad-home "$NICAD_HOME" --nicad-bin "$NICAD_BIN" --txl-bin "$TXL_BIN" \
  --nicad-granularity "$NICAD_GRANULARITY" --nicad-language "$NICAD_LANGUAGE" \
  --nicad-config "$NICAD_CONFIG" --nicad-xml-pattern "$NICAD_XML_PATTERN" \
  --jscpd-launcher "$JSCPD_LAUNCHER" --jscpd-package "$JSCPD_PACKAGE" \
  --jscpd-format "$JSCPD_FORMAT" --jscpd-similarity "$JSCPD_SIMILARITY" \
  --jscpd-min-lines "$JSCPD_MIN_LINES" --jscpd-min-tokens "$JSCPD_MIN_TOKENS" \
  --jscpd-reporter "$JSCPD_REPORTER" --merge-script "$MERGE_SCRIPT" \
  --merge-git-blob-sha "$MERGE_GIT_BLOB_SHA" \
  --ni-mapping-threshold "$NI_MAPPING_THRESHOLD" \
  --strict-function-fraction "$STRICT_FUNCTION_FRACTION" \
  --boundary-tolerance-lines "$BOUNDARY_TOLERANCE_LINES" \
  --probe-mode "$PROBE_MODE" --probe-timeout-sec "$PROBE_TIMEOUT_SEC" \
  --probe-progress-every "$PROBE_PROGRESS_EVERY" --copy-progress-every "$COPY_PROGRESS_EVERY" \
  --nicad-timeout-sec "$NICAD_TIMEOUT_SEC" --jscpd-timeout-sec "$JSCPD_TIMEOUT_SEC" \
  --merge-timeout-sec "$MERGE_TIMEOUT_SEC" --report-dir "$REPORT_DIR" --run-id "$RUN_ID")
call_py() {
  local run_mode="$1" chosen_subject="$2" hash_check="$3"
  "$PYTHON_BIN" -u "$PYTHON_SCRIPT" "${BASE_ARGS[@]}" --mode "$run_mode" \
    --subject "$chosen_subject" --hash-check "$hash_check"
}

step_start 'preflight'
[[ -x "$PYTHON_BIN" && -f "$PYTHON_SCRIPT" && ! -L "$PYTHON_SCRIPT" ]] || {
  echo 'Missing J-04c Python runtime or implementation' >&2; exit 2;
}
[[ -d "$DATASET_ROOT" && -d "$INPUT_ROOT" && ! -L "$DATASET_ROOT" && ! -L "$INPUT_ROOT" ]] || {
  echo 'Missing/unsafe J-04b v8 verified dataset' >&2; exit 2;
}
[[ ! -L "$PUBLISH_ROOT" && ! -L "$RESULT_ROOT" ]] || { echo 'Output root symlink rejected' >&2; exit 2; }
if [[ "$MODE" == 'subject' || "$MODE" == 'smoke' || "$MODE" == 'all' || "$MODE" == 'copy-result' ]]; then
  # This is THE C-01 lock, not a parallel private J-04c lock.
  exec 9>"$NICAD_SHARED_LOCK"
  flock -n 9 || { echo 'Another C-01/J-04c NiCad run is active' >&2; exit 6; }
fi
step_pass

if [[ "$MODE" == 'inspect' || "$MODE" == 'dry-run' ]]; then
  step_start "$MODE"
  call_py "$MODE" 'ALL' "$VERIFY_INPUT_HASHES"
  step_pass
elif [[ "$MODE" == 'verify' ]]; then
  step_start 'verify'
  call_py verify "$SUBJECT" "$VERIFY_INPUT_HASHES"
  step_pass
elif [[ "$MODE" == 'copy-result' ]]; then
  step_start 'verify_and_copy_complete_all'
  call_py copy-result ALL "$VERIFY_INPUT_HASHES"
  step_pass
else
  step_start 'dataset_integrity'
  call_py inspect ALL "$VERIFY_INPUT_HASHES"
  call_py dry-run ALL false
  step_pass
  if [[ "$MODE" == 'subject' || "$MODE" == 'smoke' ]]; then
    SELECTED_SUBJECTS=("$SUBJECT")
  else
    SUBJECT_TEXT="$(call_py list-subjects ALL false)"
    mapfile -t SELECTED_SUBJECTS <<< "$SUBJECT_TEXT"
    (( ${#SELECTED_SUBJECTS[@]} == EXPECTED_SUBJECTS )) || {
      echo "Expected $EXPECTED_SUBJECTS subjects, got ${#SELECTED_SUBJECTS[@]}" >&2; exit 2;
    }
  fi
  for item in "${SELECTED_SUBJECTS[@]}"; do
    step_start "detect_${item}"
    call_py run-subject "$item" false
    step_pass
    step_start "verify_${item}"
    call_py verify "$item" false
    step_pass
  done
  step_start 'batch_summary'
  call_py summary "$SUBJECT" false
  step_pass
fi
