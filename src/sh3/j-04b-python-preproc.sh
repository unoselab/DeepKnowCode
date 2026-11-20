#!/usr/bin/env bash
# DeepKnowCode J-04b v2 - static Python source preprocessing, preserving v1 artifacts.
# v1 output: python-repo-preproc; v2 output: python-repo-preproc-v2 (never overwrite v1).
#
# Standard execution sequence (run from the DeepKnowCode repository root):
#
# Step 1: Inspect all 24 Python repositories and confirm C-01 Git HEAD alignment
# bash src/sh3/j-04b-python-preproc.sh --inspect
#
# Step 2: Validate the output plan without writing preprocessed source files
# bash src/sh3/j-04b-python-preproc.sh --dry-run
#
# Step 3: Smoke-test one subject (pallets__werkzeug)
# bash src/sh3/j-04b-python-preproc.sh --smoke
#
# Step 4: Preprocess all 24 subjects, verifying existing immutable outputs
# bash src/sh3/j-04b-python-preproc.sh --all
#
# Step 5: Independently verify every source SHA256, output SHA256 and manifest
# bash src/sh3/j-04b-python-preproc.sh --verify
#
# Optional: preprocess one specified subject
# bash src/sh3/j-04b-python-preproc.sh --subject django__django
#
set -Eeuo pipefail

# SHELL-OWNED CONFIGURATION: Python must receive every effective setting explicitly.
EXPERIMENT='j-04b'
SCRIPT_VERSION='v2'
PREPROC_DATA_VERSION='v2'
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PYTHON_BIN='python3'
PYTHON_SCRIPT="$ROOT/src/py3/j-04b-python-preproc.py"
REPO_ROOT="$ROOT/data/repositories/python-repos"
PREPROC_ROOT="$ROOT/data/repositories/python-repo-preproc-v2"
BASELINE_ORGANIZER="$ROOT/results/c-01-batch/organizer-20261103-013600-284803.json"
EXPECTED_SUBJECTS=24
SMOKE_SUBJECT='pallets__werkzeug'
REQUIRE_CLEAN='true'
REQUIRE_BASELINE_COMMIT='true'
PARSE_FAILURE_POLICY='preserve_original_audited'
MASK_HEAD_DOCSTRINGS='true'
MASK_STANDALONE_TRIPLE_STRINGS='true'
PROGRESS_EVERY=5000
SKIP_DIRS=(test tests testing testdata test_data testcase testcases testfixture testfixtures test-fixture test-fixtures testsrc testsources test-sources fixtures fixture integration-test integration-tests integration_testing integrationtests)
SKIP_GLOBS=('test_*.py' '*_test.py' 'tests_*.py' 'conftest.py')
# Tool directives are kept intact. Ordinary comments are masked to byte-identical length.
PROTECTED_COMMENT_PREFIXES=('type:' 'noqa' 'fmt:' 'pragma:' 'pylint:' 'pyright:' 'mypy:' 'isort:' 'ruff:' 'nosec' 'cython:' 'pytest:' 'region' 'endregion')
LOG_TZ='America/Chicago'
LOG_DIR="$ROOT/logs/$EXPERIMENT"
LOG_PREFIX="$EXPERIMENT-$SCRIPT_VERSION-python-preproc"
LOG_TIMESTAMP_FORMAT='+%Y%m%d-%H%M%S-%N'
LOG_CAPTURE_STDOUT='true'
LOG_CAPTURE_STDERR='true'
LOG_TEE_TERMINAL='true'
LOG_STEP_TIMING='true'
RESULT_DIR="$ROOT/results/$EXPERIMENT"

usage() {
  echo 'Usage: bash src/sh3/j-04b-python-preproc.sh {--inspect|--dry-run|--smoke|--subject owner__repo|--all|--verify}' >&2
  exit 2
}
(( $# >= 1 )) || usage
MODE=''
SUBJECT="$SMOKE_SUBJECT"
case "$1" in
  --inspect|--dry-run|--smoke|--all|--verify)
    (( $# == 1 )) || usage
    MODE="${1#--}" ;;
  --subject)
    (( $# == 2 )) || usage
    MODE='subject'; SUBJECT="$2"
    [[ "$SUBJECT" =~ ^[a-z0-9][a-z0-9_.-]*__[a-z0-9][a-z0-9_.-]*$ ]] || usage ;;
  *) usage ;;
esac

export TZ="$LOG_TZ"
export LC_ALL=C
mkdir -p -- "$LOG_DIR" "$RESULT_DIR"
START_EPOCH="$(date +%s)"
START_CHICAGO="$(date '+%Y-%m-%dT%H:%M:%S%z')"
STAMP="$(date "$LOG_TIMESTAMP_FORMAT")"
LOG_FILE="$(mktemp "$LOG_DIR/${LOG_PREFIX}-${STAMP}-pid$$-${MODE}-XXXXXXXX.log")"
RUN_DIR="$(mktemp -d "$RESULT_DIR/${STAMP}-pid$$-${MODE}-XXXXXXXX")"
readonly LOG_FILE RUN_DIR
exec > >(tee -a -- "$LOG_FILE") 2>&1
now() { date '+%Y-%m-%dT%H:%M:%S%z'; }
STEP='initialization'
STEP_EPOCH="$START_EPOCH"
step_start() {
  STEP="$1"
  STEP_EPOCH="$(date +%s)"
  printf '[%s] START step=%s\n' "$(now)" "$STEP"
}
step_pass() {
  printf '[%s] PASS step=%s elapsed_sec=%d\n' "$(now)" "$STEP" "$(( $(date +%s)-STEP_EPOCH ))"
}
finalize() {
  local rc=$?
  trap - EXIT
  local status='FAIL'
  if (( rc == 0 )); then status='PASS'; else
    printf '[%s] FAIL step=%s exit=%d elapsed_sec=%d\n' "$(now)" "$STEP" "$rc" "$(( $(date +%s)-STEP_EPOCH ))"
  fi
  printf '[%s] FINAL experiment=%s version=%s data_version=%s mode=%s subject=%s status=%s exit=%d start=%s end=%s elapsed_sec=%d log=%s result=%s source=%s destination=%s\n' \
    "$(now)" "$EXPERIMENT" "$SCRIPT_VERSION" "$PREPROC_DATA_VERSION" "$MODE" "$SUBJECT" "$status" "$rc" \
    "$START_CHICAGO" "$(now)" "$(( $(date +%s)-START_EPOCH ))" "$LOG_FILE" "$RUN_DIR" "$REPO_ROOT" "$PREPROC_ROOT"
  exit "$rc"
}
trap finalize EXIT
printf '[%s] RUN experiment=%s version=%s mode=%s log=%s result=%s timezone=%s\n' \
  "$(now)" "$EXPERIMENT" "$SCRIPT_VERSION" "$MODE" "$LOG_FILE" "$RUN_DIR" "$LOG_TZ"

step_start 'preflight'
[[ -f "$PYTHON_SCRIPT" && ! -L "$PYTHON_SCRIPT" ]] || { echo "Missing/unsafe Python implementation: $PYTHON_SCRIPT" >&2; exit 2; }
[[ -d "$REPO_ROOT" && ! -L "$REPO_ROOT" ]] || { echo "Missing/unsafe Python source checkout root: $REPO_ROOT" >&2; exit 2; }
[[ -f "$BASELINE_ORGANIZER" && ! -L "$BASELINE_ORGANIZER" ]] || { echo "Missing/unsafe baseline organizer: $BASELINE_ORGANIZER" >&2; exit 2; }
[[ ! -L "$PREPROC_ROOT" ]] || { echo "Unsafe output root symlink: $PREPROC_ROOT" >&2; exit 2; }
command -v "$PYTHON_BIN" >/dev/null || { echo "Python missing: $PYTHON_BIN" >&2; exit 2; }
step_pass

step_start "$MODE"
ARGS=(--experiment "$EXPERIMENT" --version "$SCRIPT_VERSION" \
  --preproc-data-version "$PREPROC_DATA_VERSION" --mode "$MODE" --subject "$SUBJECT" \
  --repo-root "$REPO_ROOT" --preproc-root "$PREPROC_ROOT" \
  --baseline-organizer "$BASELINE_ORGANIZER" --report-dir "$RUN_DIR" \
  --require-clean "$REQUIRE_CLEAN" --require-baseline-commit "$REQUIRE_BASELINE_COMMIT" \
  --parse-failure-policy "$PARSE_FAILURE_POLICY" \
  --mask-head-docstrings "$MASK_HEAD_DOCSTRINGS" \
  --mask-standalone-triple-strings "$MASK_STANDALONE_TRIPLE_STRINGS" \
  --expected-subjects "$EXPECTED_SUBJECTS" --progress-every "$PROGRESS_EVERY")
for item in "${SKIP_DIRS[@]}"; do ARGS+=(--skip-dir "$item"); done
for item in "${SKIP_GLOBS[@]}"; do ARGS+=(--skip-glob "$item"); done
for item in "${PROTECTED_COMMENT_PREFIXES[@]}"; do ARGS+=(--protected-comment-prefix "$item"); done
printf '[%s] CONFIG expected=%d policy=%s mask_head_docstrings=%s mask_standalone_triple_strings=%s require_clean=%s require_baseline_commit=%s smoke=%s source=%s destination=%s\n' \
  "$(now)" "$EXPECTED_SUBJECTS" "$PARSE_FAILURE_POLICY" "$MASK_HEAD_DOCSTRINGS" "$MASK_STANDALONE_TRIPLE_STRINGS" "$REQUIRE_CLEAN" "$REQUIRE_BASELINE_COMMIT" \
  "$SMOKE_SUBJECT" "$REPO_ROOT" "$PREPROC_ROOT"
"$PYTHON_BIN" -u "$PYTHON_SCRIPT" "${ARGS[@]}"
step_pass
