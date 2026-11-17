#!/usr/bin/env bash
# Standalone J-03c v1. All experiment settings and logging controlled by this Shell.
set -Eeuo pipefail
EXPERIMENT="j-03c"
VERSION="v1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
REPO_ROOT="$ROOT/data/repositories/java-repos"
PREPROC_ROOT="$ROOT/data/repositories/java-repo-preproc"
J03A_RUN="$ROOT/results/j-03a/20261116-165011-200729-all-9Gte3a"
J03B_SMOKE_RUN="$ROOT/results/j-03b/20261117-114822-047349-778239-56ec5466cbc5-smoke"
BEFORE_ROOT="$ROOT/data/clone-detection/1before-preproc"
AFTER_ROOT="$ROOT/data/clone-detection/2after-preproc"
RESULT_ROOT="$ROOT/results/j-03c"
LOG_DIR="$ROOT/logs/j-03c"
LOG_PREFIX="j-03c-v1-nicad-archive"
LOG_TIMESTAMP_FORMAT="%Y%m%d-%H%M%S-%6N"
LOG_TEE_APPEND="true"
LOG_CAPTURE_STDOUT="true"
LOG_CAPTURE_STDERR="true"
LOG_STEP_TIMING="true"
PYTHON_BIN="python3"
PYTHON_SCRIPT="$ROOT/src/py3/j-03c-nicad-archive.py"
NICAD_HOME="$ROOT/tools/nicad-7.0.1-linux-x86_64"
OPENTXL_HOME="$ROOT/tools/opentxl-11.3.7-linux-x64"
NICAD_GRANULARITY="functions"
NICAD_LANGUAGE="java"
NICAD_CONFIG="default"
NICAD_THRESHOLD="0.30"
PREPROC_VERSION="v2"
BASELINE_VERSION="v3"
UNICODE_POLICY="preserve_original_audited"
EXPECTED_SUBJECTS=47
SMOKE_SUBJECT="apache__ant"
REQUIRE_CLEAN="true"
STRICT_PARSE_FAILURES="false"
COPY_NICAD_CLASSES_XML="true"
PROGRESS_EVERY_PAIRS=20000
REUSE_VERIFIED_J03B_SMOKE="true"
export TZ="America/Chicago"
export LC_ALL=C
export PATH="$OPENTXL_HOME/bin:$NICAD_HOME/bin:$PATH"
now() { date '+%Y-%m-%dT%H:%M:%S%:z'; }
usage() {
  echo "Usage: bash src/sh3/j-03c-nicad-archive.sh --inspect|--smoke|--before-subject SUBJECT|--after-subject SUBJECT|--before-all|--after-all|--all" >&2
  exit 2
}
(( $# >= 1 )) || usage
MODE="$1"; shift
SUBJECT=""
case "$MODE" in
  --inspect|--smoke|--before-all|--after-all|--all) (( $# == 0 )) || usage ;;
  --before-subject|--after-subject)
    (( $# == 1 )) || usage
    SUBJECT="$1"
    [[ "$SUBJECT" =~ ^[a-z0-9][a-z0-9_.-]*__[a-z0-9][a-z0-9_.-]*$ ]] || usage ;;
  *) usage ;;
esac
mkdir -p "$LOG_DIR" "$RESULT_ROOT"
TAG="$(date "+$LOG_TIMESTAMP_FORMAT")-$$-$(head -c 6 /dev/urandom | od -An -tx1 | tr -d ' \n')"
RUN="$RESULT_ROOT/$TAG-${MODE#--}"
mkdir "$RUN"
LOG_FILE="$LOG_DIR/$LOG_PREFIX-$TAG.log"
[[ ! -e "$LOG_FILE" ]] || { echo "Refusing overwrite: $LOG_FILE" >&2; exit 2; }
exec > >(tee -a "$LOG_FILE") 2>&1
START_EPOCH="$(date +%s)"
CURRENT_STEP="init"
CURRENT_SUBJECT="global"
ACTIVE_CHILD=""
FINAL_STATUS="PASS"
finish() {
  local ec="$?"
  trap - EXIT
  if (( ec != 0 )); then FINAL_STATUS="FAIL"; fi
  echo "[$(now)] ${FINAL_STATUS} FINAL experiment=$EXPERIMENT version=$VERSION mode=$MODE subject=$CURRENT_SUBJECT exit=$ec start_epoch=$START_EPOCH end_epoch=$(date +%s) elapsed_sec=$(($(date +%s)-START_EPOCH)) before=$BEFORE_ROOT after=$AFTER_ROOT result_run=$RUN log=$LOG_FILE"
  exit "$ec"
}
terminate() {
  trap - TERM INT
  if [[ -n "$ACTIVE_CHILD" ]]; then
    kill -TERM -- "-$ACTIVE_CHILD" 2>/dev/null || kill -TERM "$ACTIVE_CHILD" 2>/dev/null || true
  fi
  exit 130
}
trap finish EXIT
trap terminate TERM INT
COMMON=(--experiment "$EXPERIMENT" --version "$VERSION" --subject "$SMOKE_SUBJECT" \
  --repo-root "$REPO_ROOT" --preproc-root "$PREPROC_ROOT" --j03a-run "$J03A_RUN" \
  --before-root "$BEFORE_ROOT" --after-root "$AFTER_ROOT" --report-dir "$RUN" \
  --expected-subjects "$EXPECTED_SUBJECTS" --config "$NICAD_CONFIG" \
  --threshold "$NICAD_THRESHOLD" --granularity "$NICAD_GRANULARITY" \
  --language "$NICAD_LANGUAGE" --preproc-version "$PREPROC_VERSION" \
  --baseline-version "$BASELINE_VERSION" \
  --unicode-policy "$UNICODE_POLICY" --require-clean "$REQUIRE_CLEAN" \
  --strict-parse "$STRICT_PARSE_FAILURES" --copy-class-xml "$COPY_NICAD_CLASSES_XML" \
  --progress-every "$PROGRESS_EVERY_PAIRS")
runpy() {
  local kind="$1" subject="$2"; shift 2
  "$PYTHON_BIN" "$PYTHON_SCRIPT" "${COMMON[@]}" --mode "$kind" --subject "$subject" "$@"
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
subject_names() {
  local candidate
  SUBJECTS=()
  for candidate in "$REPO_ROOT"/*__*; do
    [[ -d "$candidate" && -d "$candidate/.git" ]] || continue
    SUBJECTS+=("${candidate##*/}")
  done
  if (( ${#SUBJECTS[@]} != EXPECTED_SUBJECTS )); then
    echo "Subject count mismatch: ${#SUBJECTS[@]} != $EXPECTED_SUBJECTS" >&2
    return 2
  fi
  mapfile -t SUBJECTS < <(printf '%s\n' "${SUBJECTS[@]}" | LC_ALL=C sort)
}
# Prevent concurrently running archive writers; use flock only when available.
command -v flock >/dev/null 2>&1 || { echo 'flock required' >&2; exit 2; }
mkdir -p "$ROOT/data/clone-detection"
exec 9>"$ROOT/data/clone-detection/.j03c-archive.lock"
flock -n 9 || { echo 'Another J-03c archive run is active' >&2; exit 2; }
run_nicad() {
  local sub="$1" work xml rc=0 start
  local -a matches
  CURRENT_SUBJECT="$sub"
  # Prior archive is authoritative if all stored hashes pass. Never recompute or replace it.
  if [[ -d "$AFTER_ROOT/$sub" ]]; then
    step verify_after runpy verify-after "$sub" || return $?
    step compare_existing runpy after "$sub"
    return $?
  fi
  if [[ ! -d "$PREPROC_ROOT/$sub" ]]; then
    echo "Preprocessed source missing: $PREPROC_ROOT/$sub" >&2
    return 2
  fi
  # apache__ant from the successful J-03b v2 smoke is verified against its
  # SHA256 and J-03a baseline by Python before it can be published.
  local legacy=""
  if [[ "$REUSE_VERIFIED_J03B_SMOKE" == "true" && "$sub" == "$SMOKE_SUBJECT" && -f "$J03B_SMOKE_RUN/$sub-comparison.json" ]]; then
    mapfile -t matches < <(find "$J03B_SMOKE_RUN/nicad-$sub/nicadclones" -type f -name "*-clones-$NICAD_THRESHOLD.xml" ! -name '*-classes-*' | LC_ALL=C sort)
    (( ${#matches[@]} == 1 )) || { echo "Unexpected J-03b smoke XML count: ${#matches[@]}" >&2; return 2; }
    xml="${matches[0]}"
    legacy="$J03B_SMOKE_RUN/$sub-comparison.json"
    echo "[$(now)] REUSE_VERIFIED_J03B_XML subject=$sub xml=$xml comparison=$legacy"
  else
    work="$RUN/nicad-$sub"
    mkdir "$work"
    start="$(date +%s)"
    CURRENT_STEP="nicad_after_preproc"
    echo "[$(now)] START step=$CURRENT_STEP subject=$sub source=$PREPROC_ROOT/$sub work=$work"
    setsid bash -c 'set -Eeuo pipefail; cd "$1"; bash "$2/scripts/nicadpair.sh" "$2" "$3" "$4" "$5" "$6"' \
      bash "$work" "$NICAD_HOME/lib/nicad" "$NICAD_GRANULARITY" "$NICAD_LANGUAGE" "$PREPROC_ROOT/$sub" "$NICAD_CONFIG" &
    ACTIVE_CHILD=$!
    wait "$ACTIVE_CHILD" || rc=$?
    ACTIVE_CHILD=""
    if (( rc != 0 )); then
      echo "[$(now)] FAIL step=$CURRENT_STEP subject=$sub exit=$rc elapsed_sec=$(($(date +%s)-start))"
      return "$rc"
    fi
    echo "[$(now)] PASS step=$CURRENT_STEP subject=$sub elapsed_sec=$(($(date +%s)-start))"
    mapfile -t matches < <(find "$work/nicadclones" -type f -name "*-clones-$NICAD_THRESHOLD.xml" ! -name '*-classes-*' | LC_ALL=C sort)
    (( ${#matches[@]} == 1 )) || { echo "NiCad expected one pair XML; found ${#matches[@]}" >&2; return 2; }
    xml="${matches[0]}"
  fi
  if [[ -n "$legacy" ]]; then
    step publish_after runpy after "$sub" --nicad-xml "$xml" --legacy-comparison-file "$legacy"
  else
    step publish_after runpy after "$sub" --nicad-xml "$xml"
  fi
}
run_before_all() {
  local sub idx=0 failures=0
  subject_names || return $?
  for sub in "${SUBJECTS[@]}"; do
    idx=$((idx+1)); CURRENT_SUBJECT="$sub"
    echo "[$(now)] BEFORE_START subject=$sub index=$idx/${#SUBJECTS[@]}"
    if step publish_before runpy before "$sub"; then
      echo "[$(now)] BEFORE_PASS subject=$sub index=$idx/${#SUBJECTS[@]}"
    else
      failures=$((failures+1))
      echo "[$(now)] BEFORE_FAIL subject=$sub index=$idx/${#SUBJECTS[@]}"
    fi
  done
  echo "[$(now)] BEFORE_TOTAL expected=${#SUBJECTS[@]} failures=$failures"
  (( failures == 0 ))
}
run_after_all() {
  local sub idx=0 failures=0 differences=0 rc=0
  subject_names || return $?
  CURRENT_SUBJECT="global"
  step preflight_preprocessed runpy preflight-after "$SMOKE_SUBJECT" || return $?
  for sub in "${SUBJECTS[@]}"; do
    idx=$((idx+1)); CURRENT_SUBJECT="$sub"
    echo "[$(now)] AFTER_START subject=$sub index=$idx/${#SUBJECTS[@]}"
    rc=0
    run_nicad "$sub" || rc=$?
    if (( rc == 0 )); then
      echo "[$(now)] AFTER_PASS subject=$sub index=$idx/${#SUBJECTS[@]}"
    elif (( rc == 3 )); then
      differences=$((differences+1))
      echo "[$(now)] AFTER_DIFFERENT subject=$sub index=$idx/${#SUBJECTS[@]}"
    else
      failures=$((failures+1))
      echo "[$(now)] AFTER_FAIL subject=$sub index=$idx/${#SUBJECTS[@]} exit=$rc"
    fi
  done
  echo "[$(now)] AFTER_TOTAL expected=${#SUBJECTS[@]} failures=$failures differences=$differences"
  CURRENT_SUBJECT="global"
  step aggregate runpy summarize "$SMOKE_SUBJECT" || return $?
  (( failures == 0 )) || return 2
  (( differences == 0 )) || return 3
}
echo "[$(now)] START experiment=$EXPERIMENT version=$VERSION mode=$MODE run=$RUN log=$LOG_FILE"
echo "[$(now)] CONFIG before=$BEFORE_ROOT after=$AFTER_ROOT raw=$REPO_ROOT preproc=$PREPROC_ROOT baseline=$J03A_RUN baseline_version=$BASELINE_VERSION preproc_version=$PREPROC_VERSION expected_subjects=$EXPECTED_SUBJECTS config=$NICAD_CONFIG threshold=$NICAD_THRESHOLD granularity=$NICAD_GRANULARITY language=$NICAD_LANGUAGE reuse_smoke=$REUSE_VERIFIED_J03B_SMOKE smoke_run=$J03B_SMOKE_RUN strict_parse=$STRICT_PARSE_FAILURES"
case "$MODE" in
  --inspect) CURRENT_SUBJECT="global"; step inventory runpy inspect "$SMOKE_SUBJECT" ;;
  --smoke)
    CURRENT_SUBJECT="$SMOKE_SUBJECT"
    step publish_before runpy before "$SMOKE_SUBJECT"
    run_nicad "$SMOKE_SUBJECT" ;;
  --before-subject)
    CURRENT_SUBJECT="$SUBJECT"
    step publish_before runpy before "$SUBJECT" ;;
  --after-subject)
    CURRENT_SUBJECT="$SUBJECT"
    step publish_before runpy before "$SUBJECT"
    run_nicad "$SUBJECT" ;;
  --before-all) run_before_all; CURRENT_SUBJECT="global"; step aggregate runpy summarize "$SMOKE_SUBJECT" ;;
  --after-all) run_after_all ;;
  --all)
    run_before_all
    run_after_all ;;
esac
