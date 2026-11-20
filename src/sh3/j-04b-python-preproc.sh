#!/usr/bin/env bash
# DeepKnowCode J-04b v8: Python 3.12 main + isolated Python 3.14 native AST worker.
# v1-v7 outputs remain immutable; v8 output: python-repo-preproc-v8.
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
# Optional: build verified Clone Detection input from finished v8 preprocessing.
# bash src/sh3/j-04b-python-preproc.sh --exclude-unparsed-files
# bash src/sh3/j-04b-python-preproc.sh --verify --exclude-unparsed-files
# Or build it during the full run: --all --exclude-unparsed-files
#
# Optional: preprocess one specified subject
# bash src/sh3/j-04b-python-preproc.sh --subject django__django
#
set -Eeuo pipefail

# SHELL-OWNED CONFIGURATION: Python must receive every effective setting explicitly.
EXPERIMENT='j-04b'
SCRIPT_VERSION='v8'
PREPROC_DATA_VERSION='v8'
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PYTHON_BIN="$HOME/miniconda3/envs/deepknowcode/bin/python3"
PYTHON_MIN_VERSION='3.12'
MAIN_PYTHON_EXPECTED_VERSION='3.12.16'
AST_PYTHON_BIN="$HOME/miniconda3/envs/ast-py314/bin/python"
AST_PYTHON_MIN_VERSION='3.14'
AST_PYTHON_EXPECTED_VERSION='3.14.10'
AST_PARSER_MODE='external_py314_native'
AST_SUPPRESS_SYNTAX_WARNINGS='true'
PARSER_COMPAT_PEP758='false'
PYTHON_SCRIPT="$ROOT/src/py3/j-04b-python-preproc.py"
AST_WORKER_SCRIPT="$ROOT/src/py3/j-04b-ast-worker.py"
REPO_ROOT="$ROOT/data/repositories/python-repos"
PREPROC_ROOT="$ROOT/data/repositories/python-repo-preproc-v8"
# Clone-detection input policy, explicitly configured here (never in Python defaults).
CLONE_INPUT_ROOT="$ROOT/data/repositories/python-repo-clone-input-v8-filtered"
CLONE_INPUT_POLICY='native_ast_only_exclude_all_fallback'
CLONE_COPY_MODE='copy_verified'
CLONE_EXPECTED_TOTAL_INCLUDED=22774
CLONE_EXPECTED_UNPARSED_FILES=28
BASELINE_ORGANIZER="$ROOT/results/c-01-batch/organizer-20261103-013600-284803.json"
EXPECTED_SUBJECTS=24
SMOKE_SUBJECT='pallets__werkzeug'
REQUIRE_CLEAN='true'
REQUIRE_BASELINE_COMMIT='true'
PARSE_FAILURE_POLICY='preserve_original_audited'
MASK_HEAD_DOCSTRINGS='true'
MASK_STANDALONE_TRIPLE_STRINGS='true'
PROGRESS_EVERY=5000
BASELINE_SKIP_DIRS=(test tests testing testdata test_data testcase testcases testfixture testfixtures test-fixture test-fixtures testsrc testsources test-sources fixtures fixture integration-test integration-tests integration_testing integrationtests)
NONPRODUCTION_SKIP_DIRS=(doc docs documentation example examples demo demos sample samples benchmark benchmarks bench build dist generated)
SKIP_DIRS=("${BASELINE_SKIP_DIRS[@]}" "${NONPRODUCTION_SKIP_DIRS[@]}")
SKIP_GLOBS=('test*.py' '*test.py' 'conftest.py')
# Also tracked as approved candidates for a read-only v3-to-v5 impact audit.
INVENTORY_CANDIDATE_DIRS=(doc docs documentation example examples demo demos sample samples benchmark benchmarks bench build dist generated)
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

# Read-only scan: no .py processing and no writes to Git worktrees or output tree.
# bash src/sh3/j-04b-python-preproc.sh --inventory-exclusions
usage() {
  echo 'Usage: bash src/sh3/j-04b-python-preproc.sh {--inspect|--inventory-exclusions|--dry-run|--smoke|--subject owner__repo|--all [--exclude-unparsed-files]|--verify [--exclude-unparsed-files]|--exclude-unparsed-files}' >&2
  exit 2
}
(( $# >= 1 )) || usage
MODE=''
SUBJECT="$SMOKE_SUBJECT"
EXCLUDE_UNPARSED_FILES='false'
case "$1" in
  --inspect|--inventory-exclusions|--dry-run|--smoke)
    (( $# == 1 )) || usage
    MODE="${1#--}" ;;
  --all|--verify)
    MODE="${1#--}"
    if (( $# == 2 )); then
      [[ "$2" == '--exclude-unparsed-files' ]] || usage
      EXCLUDE_UNPARSED_FILES='true'
    else
      (( $# == 1 )) || usage
    fi ;;
  --exclude-unparsed-files)
    (( $# == 1 )) || usage
    MODE='exclude-unparsed-files'
    EXCLUDE_UNPARSED_FILES='true' ;;
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
[[ -f "$AST_WORKER_SCRIPT" && ! -L "$AST_WORKER_SCRIPT" ]] || { echo "Missing/unsafe AST worker: $AST_WORKER_SCRIPT" >&2; exit 2; }
[[ -x "$AST_PYTHON_BIN" && -f "$AST_PYTHON_BIN" ]] || { echo "Missing AST Python: $AST_PYTHON_BIN" >&2; exit 2; }
[[ -d "$REPO_ROOT" && ! -L "$REPO_ROOT" ]] || { echo "Missing/unsafe Python source checkout root: $REPO_ROOT" >&2; exit 2; }
[[ -f "$BASELINE_ORGANIZER" && ! -L "$BASELINE_ORGANIZER" ]] || { echo "Missing/unsafe baseline organizer: $BASELINE_ORGANIZER" >&2; exit 2; }
[[ ! -L "$CLONE_INPUT_ROOT" ]] || { echo "Unsafe clone input symlink: $CLONE_INPUT_ROOT" >&2; exit 2; }
[[ ! -L "$PREPROC_ROOT" ]] || { echo "Unsafe output root symlink: $PREPROC_ROOT" >&2; exit 2; }
command -v "$PYTHON_BIN" >/dev/null || { echo "Python missing: $PYTHON_BIN" >&2; exit 2; }
step_pass

step_start "$MODE"
ARGS=(--experiment "$EXPERIMENT" --version "$SCRIPT_VERSION" \
  --preproc-data-version "$PREPROC_DATA_VERSION" --mode "$MODE" --subject "$SUBJECT" \
  --repo-root "$REPO_ROOT" --preproc-root "$PREPROC_ROOT" \
  --clone-input-root "$CLONE_INPUT_ROOT" --clone-input-policy "$CLONE_INPUT_POLICY" \
  --clone-copy-mode "$CLONE_COPY_MODE" \
  --clone-expected-total-included "$CLONE_EXPECTED_TOTAL_INCLUDED" \
  --clone-expected-unparsed-files "$CLONE_EXPECTED_UNPARSED_FILES" \
  --exclude-unparsed-files "$EXCLUDE_UNPARSED_FILES" \
  --baseline-organizer "$BASELINE_ORGANIZER" --report-dir "$RUN_DIR" \
  --require-clean "$REQUIRE_CLEAN" --require-baseline-commit "$REQUIRE_BASELINE_COMMIT" \
  --parse-failure-policy "$PARSE_FAILURE_POLICY" \
  --parser-compat-pep758 "$PARSER_COMPAT_PEP758" --python-min-version "$PYTHON_MIN_VERSION" \
  --main-python-expected-version "$MAIN_PYTHON_EXPECTED_VERSION" \
  --ast-python-bin "$AST_PYTHON_BIN" --ast-worker-script "$AST_WORKER_SCRIPT" \
  --ast-python-expected-version "$AST_PYTHON_EXPECTED_VERSION" \
  --ast-python-min-version "$AST_PYTHON_MIN_VERSION" \
  --ast-parser-mode "$AST_PARSER_MODE" \
  --ast-suppress-syntax-warnings "$AST_SUPPRESS_SYNTAX_WARNINGS" \
  --mask-head-docstrings "$MASK_HEAD_DOCSTRINGS" \
  --mask-standalone-triple-strings "$MASK_STANDALONE_TRIPLE_STRINGS" \
  --expected-subjects "$EXPECTED_SUBJECTS" --progress-every "$PROGRESS_EVERY")
for item in "${SKIP_DIRS[@]}"; do ARGS+=(--skip-dir "$item"); done
for item in "${BASELINE_SKIP_DIRS[@]}"; do ARGS+=(--inventory-baseline-skip-dir "$item"); done
for item in "${SKIP_GLOBS[@]}"; do ARGS+=(--skip-glob "$item"); done
for item in "${INVENTORY_CANDIDATE_DIRS[@]}"; do ARGS+=(--candidate-dir "$item"); done
for item in "${PROTECTED_COMMENT_PREFIXES[@]}"; do ARGS+=(--protected-comment-prefix "$item"); done
printf '[%s] CONFIG expected=%d policy=%s mask_head_docstrings=%s mask_standalone_triple_strings=%s parser_compat_pep758=%s python_min_version=%s main_python=%s main_expected=%s ast_mode=%s ast_python=%s ast_expected=%s ast_min=%s ast_worker=%s require_clean=%s require_baseline_commit=%s smoke=%s source=%s destination=%s\n' \
  "$(now)" "$EXPECTED_SUBJECTS" "$PARSE_FAILURE_POLICY" "$MASK_HEAD_DOCSTRINGS" "$MASK_STANDALONE_TRIPLE_STRINGS" \
  "$PARSER_COMPAT_PEP758" "$PYTHON_MIN_VERSION" "$("$PYTHON_BIN" -VV 2>&1)" "$MAIN_PYTHON_EXPECTED_VERSION" \
  "$AST_PARSER_MODE" "$AST_PYTHON_BIN" "$AST_PYTHON_EXPECTED_VERSION" "$AST_PYTHON_MIN_VERSION" "$AST_WORKER_SCRIPT" \
  "$REQUIRE_CLEAN" "$REQUIRE_BASELINE_COMMIT" "$SMOKE_SUBJECT" "$REPO_ROOT" "$PREPROC_ROOT"
printf '[%s] CLONE_CONFIG enabled=%s policy=%s copy_mode=%s expected_included=%s expected_unparsed=%s destination=%s\n' \
  "$(now)" "$EXCLUDE_UNPARSED_FILES" "$CLONE_INPUT_POLICY" "$CLONE_COPY_MODE" \
  "$CLONE_EXPECTED_TOTAL_INCLUDED" "$CLONE_EXPECTED_UNPARSED_FILES" "$CLONE_INPUT_ROOT"
"$PYTHON_BIN" -u "$PYTHON_SCRIPT" "${ARGS[@]}"
step_pass
