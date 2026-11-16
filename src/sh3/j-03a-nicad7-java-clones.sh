#!/usr/bin/env bash
# J-03a v3: standalone subject-level NiCad 7 Java clone detection.
set -Eeuo pipefail

# All experiment parameters are owned by this Shell Script.
EXPERIMENT="j-03a"
VERSION="v3"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
REPOSITORY_ROOT="$ROOT/data/repositories/java-repos"
NICAD_HOME="$ROOT/tools/nicad-7.0.1-linux-x86_64"
OPENTXL_HOME="$ROOT/tools/opentxl-11.3.7-linux-x64"
PYTHON_BIN="python3"
PYTHON_SCRIPT="$ROOT/src/py3/j-03a-nicad7-java-clones.py"
GRANULARITY="functions"
LANGUAGE="java"
NICAD_CONFIG="default"
NICAD_THRESHOLD="0.30"
EXPECTED_REPOSITORIES=47
SMOKE_REPOSITORY="apache__ant"
REQUIRE_CLEAN="true"
STRICT_PARSE_FAILURES="false"
SKIP_DIRECTORIES=(test tests testing testdata test_data testcase testcases testfixture testfixtures test-fixture test-fixtures testsrc testsources test-sources fixtures fixture integration-test integration-tests integration_testing integrationtests)
SKIP_FILE_GLOBS=('Test*.java' '*Test.java' '*Tests.java' '*TestCase.java' '*TestSuite.java' '*IT.java' 'IT*.java')
EMBEDDED_TEST_POLICY="retain_audit_only"
STAGE_DISK_MULTIPLIER="1.10"
STAGE_RESERVE_BYTES=536870912
STAGE_PROGRESS_EVERY=5000
LOG_DIR="$ROOT/logs/$EXPERIMENT"
LOG_PREFIX="$EXPERIMENT-$VERSION-nicad7-java"
RESULT_DIR="$ROOT/results/$EXPERIMENT"
LOG_TIMESTAMP_FORMAT="%Y%m%d-%H%M%S-%6N"
LOG_TEE_APPEND="true"
LOG_CAPTURE_STDOUT="true"
LOG_CAPTURE_STDERR="true"
LOG_STEP_TIMING="true"
CONTINUE_AFTER_SUBJECT_FAILURE="true"
SUBJECT_ORDER="lexicographic"
RESUME_REQUIRE_SHA256="true"

export TZ="America/Chicago"
export PATH="$OPENTXL_HOME/bin:$NICAD_HOME/bin:$PATH"
usage() {
  echo "USAGE: bash src/sh3/j-03a-nicad7-java-clones.sh [--inspect|--smoke|--all|--subject owner__repo|--resume /absolute/or/relative/prior-v3-all-run]" >&2
}
(( $# >= 1 )) || { usage; exit 2; }
MODE="$1"
SUBJECT=""
RESUME_ROOT=""
case "$MODE" in
  --inspect|--smoke|--all) (( $# == 1 )) || { usage; exit 2; } ;;
  --subject) (( $# == 2 )) || { usage; exit 2; }; SUBJECT="$2" ;;
  --resume) (( $# == 2 )) || { usage; exit 2; }; RESUME_ROOT="$2" ;;
  *) usage; exit 2 ;;
esac
[[ "$SUBJECT" == "" || "$SUBJECT" =~ ^[a-z0-9][a-z0-9_.-]*__[a-z0-9][a-z0-9_.-]*$ ]] || { echo "Invalid subject: $SUBJECT" >&2; exit 2; }
mkdir -p -- "$LOG_DIR" "$RESULT_DIR"
if [[ "$MODE" == "--resume" ]]; then
  [[ -d "$RESUME_ROOT" && ! -L "$RESUME_ROOT" ]] || { echo "Invalid resume run directory" >&2; exit 2; }
  RUN_DIR="$(cd "$RESUME_ROOT" && pwd -P)"
  [[ "$RUN_DIR" == "$RESULT_DIR"/* ]] || { echo "Resume root outside results/j-03a" >&2; exit 2; }
  [[ -f "$RUN_DIR/configuration.txt" ]] || { echo "Missing v3 run configuration" >&2; exit 2; }
else
  RUN_DIR="$(mktemp -d "$RESULT_DIR/$(date +"$LOG_TIMESTAMP_FORMAT")-${MODE#--}-XXXXXX")"
fi
LOG_FILE="$(mktemp "$LOG_DIR/$LOG_PREFIX-$(date +"$LOG_TIMESTAMP_FORMAT")-XXXXXX.log")"
START_SEC="$(date +%s)"
START_CHICAGO="$(date +'%Y-%m-%dT%H:%M:%S%:z')"
CURRENT_STEP="initialization"
CURRENT_SUBJECT="none"
ACTIVE_CHILD=""
STOP_REQUESTED=0
exec > >(tee -a "$LOG_FILE") 2>&1
now() { date +'%Y-%m-%dT%H:%M:%S%:z'; }
finish() {
  local rc="$1" final="FAIL"
  (( rc == 0 )) && final="PASS"
  echo "[$(now)] $final FINAL experiment=$EXPERIMENT version=$VERSION mode=$MODE status=$final exit=$rc elapsed_sec=$(( $(date +%s)-START_SEC )) start=$START_CHICAGO end=$(now) step=$CURRENT_STEP subject=$CURRENT_SUBJECT log=$LOG_FILE run=$RUN_DIR"
}
cleanup_signal() {
  local sig="$1"
  trap - TERM INT
  STOP_REQUESTED=1
  echo "[$(now)] INTERRUPTED signal=$sig subject=$CURRENT_SUBJECT pid_group=${ACTIVE_CHILD:-none}" >&2
  if [[ -n "$ACTIVE_CHILD" ]]; then
    # setsid creates a dedicated process group for NiCad and all of its child processes.
    kill -TERM -- "-$ACTIVE_CHILD" 2>/dev/null || true
    wait "$ACTIVE_CHILD" 2>/dev/null || true
    ACTIVE_CHILD=""
  fi
  if [[ "$sig" == "INT" ]]; then exit 130; else exit 143; fi
}
trap 'rc=$?; finish "$rc"' EXIT
trap 'cleanup_signal TERM' TERM
trap 'cleanup_signal INT' INT

# Avoid simultaneous runs: only one J-03a detector on a server.
if [[ "$MODE" != "--inspect" ]]; then
  exec 9>>"$LOG_DIR/$EXPERIMENT-nicad7.lock"
  flock -n 9 || { echo "J-03a NiCad lock busy; verify old v2 has exited" >&2; exit 6; }
fi

CONFIG_TEXT="$(printf '%s\n' \
 "experiment=$EXPERIMENT" "version=$VERSION" "repo_root=$REPOSITORY_ROOT" \
 "nicad=$NICAD_HOME" "opentxl=$OPENTXL_HOME" "config=$NICAD_CONFIG" \
 "granularity=$GRANULARITY" "language=$LANGUAGE" "threshold=$NICAD_THRESHOLD" \
 "expected_repos=$EXPECTED_REPOSITORIES" "clean=$REQUIRE_CLEAN" \
 "strict_parse=$STRICT_PARSE_FAILURES" "skip_dirs=${SKIP_DIRECTORIES[*]}" \
 "skip_file_globs=${SKIP_FILE_GLOBS[*]}" "embedded=$EMBEDDED_TEST_POLICY" \
 "stage_disk_multiplier=$STAGE_DISK_MULTIPLIER" "stage_reserve_bytes=$STAGE_RESERVE_BYTES" \
 "stage_progress_every=$STAGE_PROGRESS_EVERY" "continue_after_failure=$CONTINUE_AFTER_SUBJECT_FAILURE" \
 "subject_order=$SUBJECT_ORDER" "resume_sha=$RESUME_REQUIRE_SHA256")"
if [[ "$MODE" == "--resume" ]]; then
  [[ "$(cat "$RUN_DIR/configuration.txt")" == "$CONFIG_TEXT" ]] || { echo "Resume configuration mismatch; no processing" >&2; exit 7; }
else
  printf '%s\n' "$CONFIG_TEXT" > "$RUN_DIR/configuration.txt"
fi
NICAD_LIB="$NICAD_HOME/lib/nicad"
python_args() {
  local action="$1" output="$2" subject="$3" p_mode="$4"
  shift 4
  local a=("$action" --root "$ROOT" --repo-root "$REPOSITORY_ROOT" \
    --nicad-home "$NICAD_HOME" --txl-home "$OPENTXL_HOME" \
    --output "$output" --work-dir "$output" --stage-root "$output/staged_sources" \
    --run-root "$RUN_DIR" --subject "$subject" \
    --experiment "$EXPERIMENT" --version "$VERSION" "--mode=$p_mode" \
    --expected-repos "$EXPECTED_REPOSITORIES" --smoke-repository "$SMOKE_REPOSITORY" \
    --language "$LANGUAGE" --granularity "$GRANULARITY" --config "$NICAD_CONFIG" \
    --require-clean "$REQUIRE_CLEAN" --strict-parse-failures "$STRICT_PARSE_FAILURES" \
    --stage-disk-multiplier "$STAGE_DISK_MULTIPLIER" --stage-reserve-bytes "$STAGE_RESERVE_BYTES" \
    --stage-progress-every "$STAGE_PROGRESS_EVERY" --embedded-test-policy "$EMBEDDED_TEST_POLICY" \
    --nicad-threshold "$NICAD_THRESHOLD")
  local item
  for item in "${SKIP_DIRECTORIES[@]}"; do a+=(--skip-dir "$item"); done
  for item in "${SKIP_FILE_GLOBS[@]}"; do a+=(--skip-file-glob "$item"); done
  "$PYTHON_BIN" -u "$PYTHON_SCRIPT" "${a[@]}" "$@"
}
step() {
  local label="$1"; shift
  CURRENT_STEP="$label"
  local st="$(date +%s)" rc=0
  echo "[$(now)] START step=$label subject=$CURRENT_SUBJECT"
  "$@" || rc=$?
  if (( rc == 0 )); then
    echo "[$(now)] PASS step=$label subject=$CURRENT_SUBJECT elapsed_sec=$(( $(date +%s)-st ))"
  else
    echo "[$(now)] FAIL step=$label subject=$CURRENT_SUBJECT exit=$rc elapsed_sec=$(( $(date +%s)-st ))"
  fi
  return "$rc"
}
run_nicad() {
  local output="$1" subject="$2" st="$(date +%s)" rc=0
  CURRENT_STEP="nicad7_functions_java"
  echo "[$(now)] START step=$CURRENT_STEP subject=$subject source=$output/staged_sources/$subject"
  # NiCad writes only inside its per-subject per-attempt workspace.
  setsid bash -c '
    set -Eeuo pipefail
    cd -- "$1"
    bash "$2/scripts/nicadpair.sh" "$2" "$3" "$4" "$5" "$6"
  ' bash "$output" "$NICAD_LIB" "$GRANULARITY" "$LANGUAGE" "$output/staged_sources/$subject" "$NICAD_CONFIG" &
  ACTIVE_CHILD=$!
  wait "$ACTIVE_CHILD" || rc=$?
  ACTIVE_CHILD=""
  if (( rc == 0 )); then
    echo "[$(now)] PASS step=$CURRENT_STEP subject=$subject elapsed_sec=$(( $(date +%s)-st ))"
  else
    echo "[$(now)] FAIL step=$CURRENT_STEP subject=$subject exit=$rc elapsed_sec=$(( $(date +%s)-st ))"
  fi
  return "$rc"
}

[[ -d "$REPOSITORY_ROOT" ]] || { echo "Missing repository root" >&2; exit 2; }
echo "[$(now)] START experiment=$EXPERIMENT version=$VERSION mode=$MODE run=$RUN_DIR log=$LOG_FILE"
echo "[$(now)] CONFIG per_subject_nicad=true expected_subjects=$EXPECTED_REPOSITORIES smoke=$SMOKE_REPOSITORY filter=enabled granularity=$GRANULARITY language=$LANGUAGE config=$NICAD_CONFIG"

# One global immutable preflight inventory on new runs.
if [[ "$MODE" != "--resume" ]]; then
  CURRENT_SUBJECT="all_inventory"
  step global_inventory python_args inventory "$RUN_DIR" "$SMOKE_REPOSITORY" --inspect
fi
if [[ "$MODE" == "--inspect" ]]; then
  echo "[$(now)] PASS inspect_only no_clone_detection_performed=true"
  exit 0
fi

case "$MODE" in
  --smoke) SELECTED=("$SMOKE_REPOSITORY") ;;
  --subject) SELECTED=("$SUBJECT") ;;
  --all|--resume)
    mapfile -t SELECTED < <(find "$REPOSITORY_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | LC_ALL=C sort)
    [[ ${#SELECTED[@]} -eq "$EXPECTED_REPOSITORIES" ]] || { echo "Repository count mismatch" >&2; exit 2; }
    ;;
esac
TOTAL=${#SELECTED[@]}
COMPLETED=0
FAILED=0
SKIPPED=0
for s in "${SELECTED[@]}"; do
  CURRENT_SUBJECT="$s"
  [[ "$s" =~ ^[a-z0-9][a-z0-9_.-]*__[a-z0-9][a-z0-9_.-]*$ ]] || { echo "Invalid selected repository" >&2; exit 2; }
  [[ -d "$REPOSITORY_ROOT/$s/.git" ]] || { echo "Missing Git checkout: $s" >&2; exit 2; }
  echo "[$(now)] SUBJECT_START subject=$s index=$((COMPLETED+FAILED+SKIPPED+1))/$TOTAL"
  if [[ "$MODE" == "--resume" ]]; then
    # A prior completed attempt is reusable only after independently re-validating it.
    while IFS= read -r attempt; do
      if python_args verify "$attempt" "$s" --subject > /dev/null 2>&1; then
        echo "[$(now)] SUBJECT_SKIP_VERIFIED subject=$s previous_attempt=$attempt"
        SKIPPED=$((SKIPPED+1))
        continue 2
      fi
    done < <(find "$RUN_DIR/subjects/$s/attempts" -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null | LC_ALL=C sort -r)
  fi
  mkdir -p "$RUN_DIR/subjects/$s/attempts"
  attempt="$(mktemp -d "$RUN_DIR/subjects/$s/attempts/$(date +"$LOG_TIMESTAMP_FORMAT")-XXXXXX")"
  rc=0
  step "${s}_inventory" python_args inventory "$attempt" "$s" --subject || rc=$?
  if (( rc == 0 )); then step "${s}_stage" python_args stage "$attempt" "$s" --subject || rc=$?; fi
  if (( rc == 0 )); then run_nicad "$attempt" "$s" || rc=$?; fi
  if (( rc == 0 )); then step "${s}_parse" python_args parse "$attempt" "$s" --subject || rc=$?; fi
  if (( rc == 0 )); then step "${s}_verify" python_args verify "$attempt" "$s" --subject || rc=$?; fi
  if (( rc == 0 )); then
    COMPLETED=$((COMPLETED+1))
    echo "[$(now)] SUBJECT_PASS subject=$s index=$((COMPLETED+FAILED+SKIPPED))/$TOTAL output=$attempt/clone_pairs.csv"
  else
    FAILED=$((FAILED+1))
    echo "[$(now)] SUBJECT_FAIL subject=$s index=$((COMPLETED+FAILED+SKIPPED))/$TOTAL exit=$rc attempt=$attempt" >&2
    [[ "$CONTINUE_AFTER_SUBJECT_FAILURE" == "true" ]] || break
  fi
done
CURRENT_SUBJECT="summary"
step aggregate python_args aggregate "$RUN_DIR" "$SMOKE_REPOSITORY" --inspect || exit $?
echo "[$(now)] SUBJECT_TOTAL total=$TOTAL completed_this_run=$COMPLETED skipped_verified=$SKIPPED failed_this_run=$FAILED"
(( FAILED == 0 )) || exit 1
