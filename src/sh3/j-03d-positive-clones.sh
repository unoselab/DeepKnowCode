#!/usr/bin/env bash
# =============================================================================
# DeepKnowCode J-03d v2 - Positive Clone Dataset Export
# =============================================================================
# Run from the DeepKnowCode project root in this order:
#
# 1. Preflight verification (47 J-03c AFTER subjects)
# bash src/sh3/j-03d-positive-clones.sh --inspect
#
# 2. Smoke test: apache__ant
# bash src/sh3/j-03d-positive-clones.sh --smoke
#
# 3. Full 47-repository export
# bash src/sh3/j-03d-positive-clones.sh --all
#
# 4. Final verification
# bash src/sh3/j-03d-positive-clones.sh --verify
#
# Optional single subject:
# bash src/sh3/j-03d-positive-clones.sh --subject apache__camel
#
# Authoritative clone pair INPUT: data/clone-detection/2after-preproc/
# Java code SOURCE: data/repositories/java-repo-preproc/
# Positive pair OUTPUT: data/clone-detection/3pos-clones/
# Never read 1before-preproc or overwrite existing verified output.
# Shell owns all parameters and America/Chicago step-level logging.
# =============================================================================
set -Eeuo pipefail

# --------------------------- Experiment configuration -----------------------
EXPERIMENT="j-03d"
VERSION="v2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
EXPECTED_INPUT_RELATIVE="data/clone-detection/2after-preproc"
EXPECTED_OUTPUT_RELATIVE="data/clone-detection/3pos-clones"
INPUT_ROOT="$ROOT/$EXPECTED_INPUT_RELATIVE"
SOURCE_ROOT="$ROOT/data/repositories/java-repo-preproc"
OUTPUT_ROOT="$ROOT/$EXPECTED_OUTPUT_RELATIVE"
RESULT_ROOT="$ROOT/results/j-03d"
PYTHON_BIN="python3"
PYTHON_SCRIPT="$ROOT/src/py3/j-03d-positive-clones.py"
EXPECTED_SUBJECTS=47
EXPECTED_TOTAL_PAIRS=3790008
SMOKE_SUBJECT="apache__ant"
EXPECTED_INPUT_EXPERIMENT="j-03c"
EXPECTED_INPUT_VERSION="v1"
EXPECTED_INPUT_PHASE="after"
EXPECTED_INPUT_STATUS="ARCHIVED_VERIFIED"
EXPECTED_PREPROC_EXPERIMENT="j-03b"
EXPECTED_PREPROC_VERSION="v2"
EXPECTED_PREPROC_STATUS="PREPARED"
EXPECTED_NICAD_CONFIG="default"
EXPECTED_NICAD_THRESHOLD="0.30"
EXPECTED_DETECTOR="NiCad7/functions/java"
EXPECTED_LANGUAGE="java"
POSITIVE_LABEL=1
CODE_ENCODING="utf-8"
CODE_ERRORS="surrogateescape"
PROGRESS_EVERY_FRAGMENTS=10000
PROGRESS_EVERY_PAIRS=50000

# ----------------------------- Logging configuration ------------------------
LOG_DIR="$ROOT/logs/j-03d"
LOG_PREFIX="j-03d-v2-positive-clones"
LOG_TIMESTAMP_FORMAT="%Y%m%d-%H%M%S-%6N"
LOG_TIME_ZONE="America/Chicago"
LOG_TEE_APPEND="true"
LOG_CAPTURE_STDOUT="true"
LOG_CAPTURE_STDERR="true"
LOG_STEP_TIMING="true"
LOG_SHOW_TERMINAL="true"
export TZ="$LOG_TIME_ZONE"
export LC_ALL=C
now() { date '+%Y-%m-%dT%H:%M:%S%:z'; }
usage() {
  echo "Usage: bash src/sh3/j-03d-positive-clones.sh --inspect|--smoke|--subject SUBJECT|--all|--verify" >&2
  exit 2
}
(( $# >= 1 )) || usage
MODE="$1"; shift
SUBJECT=""
case "$MODE" in
  --inspect|--smoke|--all|--verify) (( $# == 0 )) || usage ;;
  --subject)
    (( $# == 1 )) || usage
    SUBJECT="$1"
    [[ "$SUBJECT" =~ ^[a-z0-9][a-z0-9_.-]*__[a-z0-9][a-z0-9_.-]*$ ]] || usage ;;
  *) usage ;;
esac
[[ -f "$PYTHON_SCRIPT" ]] || { echo "Missing implementation: $PYTHON_SCRIPT" >&2; exit 2; }
command -v flock >/dev/null 2>&1 || { echo "flock command required" >&2; exit 2; }
mkdir -p "$LOG_DIR" "$RESULT_ROOT" "$OUTPUT_ROOT"
TAG="$(date "+$LOG_TIMESTAMP_FORMAT")-$$-$(od -An -tx1 -N6 /dev/urandom | tr -d ' \n')"
RUN_DIR="$RESULT_ROOT/$(date '+%Y%m%d-%H%M%S')-${MODE#--}-$TAG"
mkdir "$RUN_DIR"
LOG_FILE="$LOG_DIR/$LOG_PREFIX-$TAG.log"
( set -C; : > "$LOG_FILE" ) || { echo "Log collision: $LOG_FILE" >&2; exit 2; }
exec > >(tee -a "$LOG_FILE") 2>&1

START_EPOCH="$(date +%s)"
START_TIME="$(now)"
CURRENT_STEP="init"
CURRENT_SUBJECT="global"
FINAL_STATUS="PASS"
finish() {
  local ec="$?"
  trap - EXIT
  if (( ec != 0 )); then FINAL_STATUS="FAIL"; fi
  echo "[$(now)] $FINAL_STATUS FINAL experiment=$EXPERIMENT version=$VERSION mode=$MODE subject=$CURRENT_SUBJECT exit=$ec start=$START_TIME end=$(now) elapsed_sec=$(($(date +%s)-START_EPOCH)) run=$RUN_DIR log=$LOG_FILE input=$INPUT_ROOT output=$OUTPUT_ROOT"
  exit "$ec"
}
trap finish EXIT
trap 'exit 130' INT TERM

exec 9>"$ROOT/data/clone-detection/.j03d-positive-clones.lock"
flock -n 9 || { echo "[$(now)] FAIL another J-03d writer is active" >&2; exit 2; }
COMMON=(
  --experiment "$EXPERIMENT" --version "$VERSION"
  --project-root "$ROOT" --input-root "$INPUT_ROOT"
  --expected-input-relative "$EXPECTED_INPUT_RELATIVE"
  --source-root "$SOURCE_ROOT" --output-root "$OUTPUT_ROOT"
  --expected-output-relative "$EXPECTED_OUTPUT_RELATIVE" --run-dir "$RUN_DIR"
  --expected-subjects "$EXPECTED_SUBJECTS" --expected-total-pairs "$EXPECTED_TOTAL_PAIRS"
  --expected-input-experiment "$EXPECTED_INPUT_EXPERIMENT"
  --expected-input-version "$EXPECTED_INPUT_VERSION"
  --expected-input-phase "$EXPECTED_INPUT_PHASE"
  --expected-input-status "$EXPECTED_INPUT_STATUS"
  --expected-preproc-experiment "$EXPECTED_PREPROC_EXPERIMENT"
  --expected-preproc-version "$EXPECTED_PREPROC_VERSION"
  --expected-preproc-status "$EXPECTED_PREPROC_STATUS"
  --expected-nicad-config "$EXPECTED_NICAD_CONFIG"
  --expected-nicad-threshold "$EXPECTED_NICAD_THRESHOLD"
  --expected-detector "$EXPECTED_DETECTOR" --expected-language "$EXPECTED_LANGUAGE"
  --positive-label "$POSITIVE_LABEL"
  --code-encoding "$CODE_ENCODING" --code-errors "$CODE_ERRORS"
  --time-zone "$LOG_TIME_ZONE"
  --progress-every-fragments "$PROGRESS_EVERY_FRAGMENTS"
  --progress-every-pairs "$PROGRESS_EVERY_PAIRS"
)
runpy() {
  local action="$1" subject="$2"
  "$PYTHON_BIN" "$PYTHON_SCRIPT" "${COMMON[@]}" --mode "$action" --subject "$subject"
}
step() {
  local name="$1"; shift
  local start rc=0
  CURRENT_STEP="$name"
  start="$(date +%s)"
  echo "[$(now)] START step=$name subject=$CURRENT_SUBJECT"
  "$@" || rc=$?
  if (( rc == 0 )); then
    echo "[$(now)] PASS step=$name subject=$CURRENT_SUBJECT elapsed_sec=$(($(date +%s)-start))"
  else
    echo "[$(now)] FAIL step=$name subject=$CURRENT_SUBJECT exit=$rc elapsed_sec=$(($(date +%s)-start))"
  fi
  return "$rc"
}
subjects() {
  local path
  SUBJECTS=()
  for path in "$INPUT_ROOT"/*__*; do
    [[ -d "$path" && ! -L "$path" ]] || continue
    SUBJECTS+=("${path##*/}")
  done
  (( ${#SUBJECTS[@]} == EXPECTED_SUBJECTS )) || {
    echo "AFTER_SUBJECT_COUNT_MISMATCH: ${#SUBJECTS[@]} != $EXPECTED_SUBJECTS" >&2
    return 2
  }
  mapfile -t SUBJECTS < <(printf '%s\n' "${SUBJECTS[@]}" | LC_ALL=C sort)
}

echo "[$(now)] START experiment=$EXPERIMENT version=$VERSION mode=$MODE run=$RUN_DIR log=$LOG_FILE"
echo "[$(now)] CONFIG input=$INPUT_ROOT input_phase=$EXPECTED_INPUT_PHASE source=$SOURCE_ROOT output=$OUTPUT_ROOT expected_subjects=$EXPECTED_SUBJECTS expected_pairs=$EXPECTED_TOTAL_PAIRS detector=$EXPECTED_DETECTOR config=$EXPECTED_NICAD_CONFIG threshold=$EXPECTED_NICAD_THRESHOLD preproc_version=$EXPECTED_PREPROC_VERSION label=$POSITIVE_LABEL encoding=$CODE_ENCODING errors=$CODE_ERRORS time_zone=$LOG_TIME_ZONE progress_fragments=$PROGRESS_EVERY_FRAGMENTS progress_pairs=$PROGRESS_EVERY_PAIRS"
case "$MODE" in
  --inspect)
    CURRENT_SUBJECT="global"
    step preflight runpy inspect "$SMOKE_SUBJECT" ;;
  --smoke)
    CURRENT_SUBJECT="$SMOKE_SUBJECT"
    step smoke_export runpy export "$SMOKE_SUBJECT"
    step smoke_verify runpy verify "$SMOKE_SUBJECT" ;;
  --subject)
    CURRENT_SUBJECT="$SUBJECT"
    step subject_export runpy export "$SUBJECT"
    step subject_verify runpy verify "$SUBJECT" ;;
  --all)
    CURRENT_SUBJECT="global"
    step preflight runpy inspect "$SMOKE_SUBJECT"
    subjects
    idx=0
    for SUBJECT in "${SUBJECTS[@]}"; do
      idx=$((idx+1)); CURRENT_SUBJECT="$SUBJECT"
      echo "[$(now)] SUBJECT_START index=$idx/$EXPECTED_SUBJECTS subject=$SUBJECT"
      step subject_export runpy export "$SUBJECT"
      echo "[$(now)] SUBJECT_DONE index=$idx/$EXPECTED_SUBJECTS subject=$SUBJECT"
    done
    CURRENT_SUBJECT="global"
    step aggregate runpy summarize "$SMOKE_SUBJECT" ;;
  --verify)
    subjects
    idx=0
    for SUBJECT in "${SUBJECTS[@]}"; do
      idx=$((idx+1)); CURRENT_SUBJECT="$SUBJECT"
      echo "[$(now)] VERIFY_START index=$idx/$EXPECTED_SUBJECTS subject=$SUBJECT"
      step subject_verify runpy verify "$SUBJECT"
    done
    CURRENT_SUBJECT="global"
    step aggregate runpy summarize "$SMOKE_SUBJECT" ;;
esac
