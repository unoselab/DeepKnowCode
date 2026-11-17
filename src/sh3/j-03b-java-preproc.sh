#!/usr/bin/env bash
# J-03b v5 standalone shell: source preprocessing plus real NiCad clone baseline comparison.
set -Eeuo pipefail
EXPERIMENT="j-03b"
VERSION="v5"
PREPROC_DATA_VERSION="v2"
HTML_TOP_N=10
HTML_SUBDIRECTORY="inspection"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
REPO_ROOT="$ROOT/data/repositories/java-repos"
PREPROC_ROOT="$ROOT/data/repositories/java-repo-preproc"
BASELINE_J03A_RUN="$ROOT/results/j-03a/20261116-165011-200729-all-9Gte3a"
NICAD_HOME="$ROOT/tools/nicad-7.0.1-linux-x86_64"
OPENTXL_HOME="$ROOT/tools/opentxl-11.3.7-linux-x64"
PYTHON_BIN="python3"
PYTHON_SCRIPT="$ROOT/src/py3/j-03b-java-preproc.py"
EXPECTED_SUBJECTS=47
SMOKE_SUBJECT="apache__ant"
REQUIRE_CLEAN="true"
EMBEDDED_TEST_POLICY="retain_audit_only"
PROGRESS_EVERY=5000
JAVA_UNICODE_ESCAPE_POLICY="preserve_original_audited"
NICAD_GRANULARITY="functions"
NICAD_LANGUAGE="java"
NICAD_CONFIG="default"
NICAD_THRESHOLD="0.30"
SKIP_DIRS=(test tests testing testdata test_data testcase testcases testfixture testfixtures test-fixture test-fixtures testsrc testsources test-sources fixtures fixture integration-test integration-tests integration_testing integrationtests)
SKIP_GLOBS=('Test*.java' '*Test.java' '*Tests.java' '*TestCase.java' '*TestSuite.java' '*IT.java' 'IT*.java')
LOG_DIR="$ROOT/logs/j-03b"
RESULT_DIR="$ROOT/results/j-03b"
LOG_PREFIX="j-03b-v5-java-preproc"
LOG_TIMESTAMP_FORMAT="%Y%m%d-%H%M%S-%6N"
LOG_TEE_APPEND="true"
LOG_CAPTURE_STDOUT="true"
LOG_CAPTURE_STDERR="true"
LOG_STEP_TIMING="true"
export TZ="America/Chicago"
export PATH="$OPENTXL_HOME/bin:$NICAD_HOME/bin:$PATH"
export LC_ALL=C
usage() {
  echo "Usage: bash src/sh3/j-03b-java-preproc.sh --inspect|--smoke|--subject NAME|--all|--compare-subject NAME|--html-all|--html-subject NAME" >&2
  exit 2
}
(( $# >= 1 )) || usage
MODE="$1"; shift
SUBJECT=""
case "$MODE" in
 --inspect|--smoke|--all|--html-all) (( $# == 0 )) || usage ;;
 --subject|--compare-subject|--html-subject) (( $# == 1 )) || usage; SUBJECT="$1"; [[ "$SUBJECT" =~ ^[a-z0-9][a-z0-9_.-]*__[a-z0-9][a-z0-9_.-]*$ ]] || usage ;;
 *) usage ;;
esac
mkdir -p "$LOG_DIR" "$RESULT_DIR"
TAG="$(date "+$LOG_TIMESTAMP_FORMAT")-$$-$(head -c 6 /dev/urandom | od -An -tx1 | tr -d ' \n')"
RUN="$RESULT_DIR/$TAG-${MODE#--}"
mkdir "$RUN"
LOG_FILE="$LOG_DIR/$LOG_PREFIX-$TAG.log"
[[ ! -e "$LOG_FILE" ]] || { echo "Refusing overwrite: $LOG_FILE" >&2; exit 2; }
exec > >(tee -a "$LOG_FILE") 2>&1
START_EPOCH="$(date +%s)"
CURRENT_STEP="init"; CURRENT_SUBJECT="global"; ACTIVE_CHILD=""
now() { date "+%Y-%m-%dT%H:%M:%S%:z"; }
finish() {
  local ec="$?"; trap - EXIT
  if (( ec != 0 )); then echo "[$(now)] FAIL FINAL experiment=$EXPERIMENT version=$VERSION mode=$MODE subject=$CURRENT_SUBJECT step=$CURRENT_STEP exit=$ec log=$LOG_FILE run=$RUN elapsed_sec=$(($(date +%s)-START_EPOCH))";
  else echo "[$(now)] PASS FINAL experiment=$EXPERIMENT version=$VERSION mode=$MODE subject=$CURRENT_SUBJECT exit=0 log=$LOG_FILE run=$RUN elapsed_sec=$(($(date +%s)-START_EPOCH))"; fi
  exit "$ec"
}
terminate() {
  trap - TERM INT
  if [[ -n "$ACTIVE_CHILD" ]]; then kill -TERM -- "-$ACTIVE_CHILD" 2>/dev/null || kill -TERM "$ACTIVE_CHILD" 2>/dev/null || true; fi
  exit 130
}
trap finish EXIT
trap terminate TERM INT
ARGS=(--experiment "$EXPERIMENT" --version "$PREPROC_DATA_VERSION" --script-version "$VERSION" --preproc-data-version "$PREPROC_DATA_VERSION" --html-top-n "$HTML_TOP_N" --html-subdirectory "$HTML_SUBDIRECTORY" --mode inspect --repo-root "$REPO_ROOT" --preproc-root "$PREPROC_ROOT" --report-dir "$RUN" --expected-subjects "$EXPECTED_SUBJECTS" --subject "$SMOKE_SUBJECT" --require-clean "$REQUIRE_CLEAN" --embedded-test-policy "$EMBEDDED_TEST_POLICY" --progress-every "$PROGRESS_EVERY" --java-unicode-escape-policy "$JAVA_UNICODE_ESCAPE_POLICY")
for val in "${SKIP_DIRS[@]}"; do ARGS+=(--skip-dir "$val"); done
for val in "${SKIP_GLOBS[@]}"; do ARGS+=(--skip-glob "$val"); done
run_python() {
  local kind="$1" sub="$2"; shift 2
  "$PYTHON_BIN" "$PYTHON_SCRIPT" "${ARGS[@]}" --mode "$kind" --subject "$sub" "$@"
}
step() {
  local title="$1"; shift
  CURRENT_STEP="$title"
  local start="$(date +%s)" rc=0
  echo "[$(now)] START step=$CURRENT_STEP subject=$CURRENT_SUBJECT"
  "$@" || rc=$?
  if (( rc == 0 )); then echo "[$(now)] PASS step=$CURRENT_STEP subject=$CURRENT_SUBJECT elapsed_sec=$(($(date +%s)-start))";
  else echo "[$(now)] FAIL step=$CURRENT_STEP subject=$CURRENT_SUBJECT exit=$rc elapsed_sec=$(($(date +%s)-start))"; fi
  return "$rc"
}
resolve_baseline() {
  local sub="$1" d
  [[ -d "$BASELINE_J03A_RUN/subjects/$sub/attempts" ]] || { echo "Baseline subject attempts not found: $sub" >&2; return 2; }
  mapfile -t attempts < <(find "$BASELINE_J03A_RUN/subjects/$sub/attempts" -mindepth 1 -maxdepth 1 -type d | LC_ALL=C sort -r)
  for d in "${attempts[@]}"; do
    if [[ -s "$d/clone_pairs.csv" && -s "$d/run_summary.json" ]]; then printf '%s\n' "$d/clone_pairs.csv"; return 0; fi
  done
  echo "Baseline pair CSV missing: $sub" >&2; return 2
}
compare_subject() {
  local sub="$1" baseline work xml rc=0
  CURRENT_SUBJECT="$sub"
  [[ -d "$PREPROC_ROOT/$sub" ]] || { echo "Preprocessed subject missing: $sub" >&2; return 2; }
  baseline="$(resolve_baseline "$sub")" || return $?
  work="$RUN/nicad-$sub"; mkdir "$work"
  echo "[$(now)] START step=nicad_preprocessed subject=$sub source=$PREPROC_ROOT/$sub baseline=$baseline"
  local start="$(date +%s)"
  setsid bash -c 'set -Eeuo pipefail; cd "$1"; bash "$2/scripts/nicadpair.sh" "$2" "$3" "$4" "$5" "$6"' \
    bash "$work" "$NICAD_HOME/lib/nicad" "$NICAD_GRANULARITY" "$NICAD_LANGUAGE" "$PREPROC_ROOT/$sub" "$NICAD_CONFIG" &
  ACTIVE_CHILD=$!
  wait "$ACTIVE_CHILD" || rc=$?
  ACTIVE_CHILD=""
  if (( rc )); then echo "[$(now)] FAIL step=nicad_preprocessed subject=$sub exit=$rc"; return "$rc"; fi
  echo "[$(now)] PASS step=nicad_preprocessed subject=$sub elapsed_sec=$(($(date +%s)-start))"
  mapfile -t xmls < <(find "$work/nicadclones" -type f -name "*-clones-$NICAD_THRESHOLD.xml" | grep -v -- '-classes-' || true)
  (( ${#xmls[@]} == 1 )) || { echo "Expected one NiCad XML; found ${#xmls[@]}" >&2; return 2; }
  step compare_location_pairs run_python compare "$sub" --baseline-csv "$baseline" --nicad-xml "${xmls[0]}"
}
echo "[$(now)] START experiment=$EXPERIMENT version=$VERSION mode=$MODE run=$RUN log=$LOG_FILE"
echo "[$(now)] CONFIG repository_root=$REPO_ROOT preproc_root=$PREPROC_ROOT baseline=$BASELINE_J03A_RUN expected_subjects=$EXPECTED_SUBJECTS filter=J03a_v3 comment_mask=byte_line_preserving unicode_escape_policy=$JAVA_UNICODE_ESCAPE_POLICY nicad_threshold=$NICAD_THRESHOLD"
html_for_subject() {
  local sub="$1"; CURRENT_SUBJECT="$sub"
  step verify_existing run_python verify-existing "$sub"
  step html_inspection run_python html "$sub"
}
html_all() {
  step inventory run_python inspect "$SMOKE_SUBJECT"
  mapfile -t subjects < <(find "$REPO_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | LC_ALL=C sort)
  local failed=0 ix=0
  for sub in "${subjects[@]}"; do
    ix=$((ix+1))
    echo "[$(now)] SUBJECT_HTML_START subject=$sub index=$ix/${#subjects[@]}"
    if html_for_subject "$sub"; then echo "[$(now)] SUBJECT_HTML_PASS subject=$sub index=$ix/${#subjects[@]}";
    else failed=$((failed+1)); echo "[$(now)] SUBJECT_HTML_FAIL subject=$sub index=$ix/${#subjects[@]}"; fi
  done
  echo "[$(now)] SUBJECT_HTML_TOTAL total=${#subjects[@]} failed=$failed"
  (( failed == 0 ))
}
case "$MODE" in
 --html-all) html_all ;;
 --html-subject) html_for_subject "$SUBJECT" ;;
 --inspect) step inventory run_python inspect "$SMOKE_SUBJECT" ;;
 --smoke)
   step inventory run_python inspect "$SMOKE_SUBJECT"
   CURRENT_SUBJECT="$SMOKE_SUBJECT"
   step prepare_subject run_python prepare "$SMOKE_SUBJECT"
   compare_subject "$SMOKE_SUBJECT" ;;
 --subject)
   CURRENT_SUBJECT="$SUBJECT"; step prepare_subject run_python prepare "$SUBJECT" ;;
 --compare-subject) compare_subject "$SUBJECT" ;;
 --all)
   step inventory run_python inspect "$SMOKE_SUBJECT"
   mapfile -t subjects < <(find "$REPO_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | LC_ALL=C sort)
   failed=0; ix=0
   for sub in "${subjects[@]}"; do
     ix=$((ix+1)); CURRENT_SUBJECT="$sub"
     echo "[$(now)] SUBJECT_START subject=$sub index=$ix/${#subjects[@]}"
     if [[ -d "$PREPROC_ROOT/$sub" ]]; then
       if step verify_existing run_python verify-existing "$sub"; then
         echo "[$(now)] SUBJECT_VERIFIED subject=$sub index=$ix/${#subjects[@]} immutable=true"
       else
         failed=$((failed+1)); echo "[$(now)] SUBJECT_FAIL subject=$sub index=$ix/${#subjects[@]} reason=existing_policy_or_source_mismatch"
       fi
       continue
     fi
     if step prepare_subject run_python prepare "$sub"; then echo "[$(now)] SUBJECT_PASS subject=$sub index=$ix/${#subjects[@]}";
     else failed=$((failed+1)); echo "[$(now)] SUBJECT_FAIL subject=$sub index=$ix/${#subjects[@]}"; fi
   done
   echo "[$(now)] SUBJECT_TOTAL total=${#subjects[@]} failed=$failed"
   if (( failed == 0 )); then
     html_all
   else
     return_code=1; echo "PREPROC_INCOMPLETE_SKIP_HTML failed=$failed"; exit "$return_code"
   fi ;;
esac
