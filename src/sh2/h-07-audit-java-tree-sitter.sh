#!/usr/bin/env bash
set -Eeuo pipefail

VERSION='v2'
EXPERIMENT='h-07'
MODE='audit'
TIMEZONE='America/Chicago'
LOG_TIMESTAMP_FORMAT='+%Y%m%d-%H%M%S-%N'
EXPECTED_CODE_ROWS=33658
EXPECTED_CODE_SHA256='923418867148fb1c1d7ec37d449adfad2d24bc98d1d08be2086c0d75d2c4c4be'
PARSER_PACKAGE_VERSION='0.25.2'
GRAMMAR_PACKAGE_VERSION='0.23.5'
MAX_EXAMPLES=30
MAX_PARSE_FAILURE_RATE='1.0' # Census only; failures explicitly reported, not discarded
LOG_PREFIX='h-07-java-tree-sitter'
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CODE_MAP="$ROOT/data/evaluation/code/java.jsonl"
PYTHON_SCRIPT="$ROOT/src/py2/h-07-audit-java-tree-sitter.py"
LOG_DIR="$ROOT/logs/h-07/java"
RESULTS_DIR="$ROOT/results/h-07/java"

[[ "${1:-}" == '--audit' && "$#" -eq 1 ]] || { echo 'Usage: bash src/sh2/h-07-audit-java-tree-sitter.sh --audit' >&2; exit 2; }
mkdir -p "$LOG_DIR" "$RESULTS_DIR"
STAMP="$(TZ="$TIMEZONE" date "$LOG_TIMESTAMP_FORMAT")-$$"
RUN="$RESULTS_DIR/$STAMP"
LOG="$LOG_DIR/${LOG_PREFIX}-${VERSION}-${STAMP}.log"
mkdir "$RUN"
( set -o noclobber; : > "$LOG" ) || { echo "Log exists: $LOG" >&2; exit 2; }
exec > >(tee -a "$LOG") 2>&1
START=$(date +%s)
STEP_START=0
STEP='INITIALIZE'
now() { TZ="$TIMEZONE" date '+%Y-%m-%dT%H:%M:%S%z'; }
finish() {
  code=$?
  trap - EXIT
  local end duration status
  end=$(date +%s); duration=$((end - START))
  status=PASS; (( code == 0 )) || status=FAIL
  if (( code != 0 )); then
    echo "[h-07 $VERSION] STEP=$STEP FAIL ELAPSED_SECONDS=$((end-STEP_START)) CHICAGO=$(now)"
  fi
  echo "[h-07 $VERSION] FINAL_STATUS=$status EXIT=$code DURATION_SECONDS=$duration CHICAGO=$(now) RUN=$RUN LOG=$LOG"
  exit "$code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
step() { STEP="$1"; STEP_START=$(date +%s); echo "[h-07 $VERSION] STEP=$STEP START CHICAGO=$(now)"; }
pass() { echo "[h-07 $VERSION] STEP=$STEP PASS ELAPSED_SECONDS=$(($(date +%s)-STEP_START)) CHICAGO=$(now)"; }
echo "[h-07 $VERSION] START MODE=$MODE CHICAGO=$(now) RUN=$RUN LOG=$LOG"
echo "[h-07 $VERSION] CODE_MAP=$CODE_MAP EXPECTED_ROWS=$EXPECTED_CODE_ROWS EXPECTED_SHA256=$EXPECTED_CODE_SHA256 TREE_SITTER=$PARSER_PACKAGE_VERSION TREE_SITTER_JAVA=$GRAMMAR_PACKAGE_VERSION MAX_FAILURE_RATE=$MAX_PARSE_FAILURE_RATE"
step PREFLIGHT
[[ -f "$PYTHON_SCRIPT" && -f "$CODE_MAP" ]] || { echo 'Missing script or Java code map' >&2; exit 2; }
python -c 'import tree_sitter,tree_sitter_java' || exit 2
pass
step JAVA_PARSE_AUDIT
python -u "$PYTHON_SCRIPT" \
  --code-map "$CODE_MAP" \
  --output "$RUN" \
  --expected-code-rows "$EXPECTED_CODE_ROWS" \
  --expected-code-sha256 "$EXPECTED_CODE_SHA256" \
  --parser-package-version "$PARSER_PACKAGE_VERSION" \
  --grammar-package-version "$GRAMMAR_PACKAGE_VERSION" \
  --timezone "$TIMEZONE" \
  --max-examples "$MAX_EXAMPLES" \
  --max-parse-failure-rate "$MAX_PARSE_FAILURE_RATE"
pass
