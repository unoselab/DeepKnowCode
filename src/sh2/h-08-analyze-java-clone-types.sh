#!/usr/bin/env bash
set -Eeuo pipefail
VERSION='v1'
EXPERIMENT='h-08'
TIMEZONE='America/Chicago'
LOG_TIMESTAMP_FORMAT='+%Y%m%d-%H%M%S-%N'
LOG_PREFIX='h-08-java-clone-types'
EXPECTED_CODE_SHA256='923418867148fb1c1d7ec37d449adfad2d24bc98d1d08be2086c0d75d2c4c4be'
EXPECTED_H07_RECORDS_SHA256='59cf82e0c82861d0d502115cff1d915880e19fa16178978b726a2ce2241293cf'
EXPECTED_CODE_ROWS=33658
EXPECTED_CSV_FILES=48
EXPECTED_PAIRS=35192
EXPECTED_POSITIVE=17596
EXPECTED_NEGATIVE=17596
TREE_SITTER_VERSION='0.25.2'
TREE_SITTER_JAVA_VERSION='0.23.5'
THRESHOLDS='0.70,0.75,0.80,0.82,0.85,0.90,0.95'
PRIMARY_THRESHOLD='0.82'
GRAM_SIZE=5
SAMPLE_PER_TYPE=12
PROGRESS_INTERVAL=5000
MAX_AST_COMPARISON_CELLS=4000000
H07_RUN_NAME='20261114-152406-857361557-36044'
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CODE_MAP="$ROOT/data/evaluation/code/java.jsonl"
GROUND_TRUTH_DIR="$ROOT/data/evaluation/ground_truth/java"
H07_RUN="$ROOT/results/h-07/java/$H07_RUN_NAME"
H07_RECORDS="$H07_RUN/function_parse_records.jsonl"
H07_AUDIT="$H07_RUN/parse_audit.json"
PYTHON_SCRIPT="$ROOT/src/py2/h-08-analyze-java-clone-types.py"
LOG_DIR="$ROOT/logs/h-08/java"
RESULTS_DIR="$ROOT/results/h-08/java"
[[ "${1:-}" == '--analyze' && "$#" -eq 1 ]] || { echo 'Usage: bash src/sh2/h-08-analyze-java-clone-types.sh --analyze' >&2; exit 2; }
mkdir -p "$LOG_DIR" "$RESULTS_DIR"
STAMP="$(TZ="$TIMEZONE" date "$LOG_TIMESTAMP_FORMAT")-$$"
RUN="$RESULTS_DIR/$STAMP"
LOG="$LOG_DIR/${LOG_PREFIX}-${VERSION}-${STAMP}.log"
mkdir "$RUN"
( set -o noclobber; : > "$LOG" ) || exit 2
exec > >(tee -a "$LOG") 2>&1
START=$(date +%s)
STEP_START=$START
STEP='INITIALIZE'
now() { TZ="$TIMEZONE" date '+%Y-%m-%dT%H:%M:%S%z'; }
finish() {
  code=$?
  trap - EXIT
  local end duration status
  end=$(date +%s)
  duration=$((end-START))
  status=PASS
  (( code == 0 )) || status=FAIL
  if (( code != 0 )); then echo "[h-08 $VERSION] STEP=$STEP FAIL ELAPSED_SECONDS=$((end-STEP_START)) CHICAGO=$(now)"; fi
  echo "[h-08 $VERSION] FINAL_STATUS=$status EXIT=$code DURATION_SECONDS=$duration CHICAGO=$(now) RUN=$RUN LOG=$LOG"
  exit "$code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
step() { STEP="$1"; STEP_START=$(date +%s); echo "[h-08 $VERSION] STEP=$STEP START CHICAGO=$(now)"; }
pass() { echo "[h-08 $VERSION] STEP=$STEP PASS ELAPSED_SECONDS=$(($(date +%s)-STEP_START)) CHICAGO=$(now)"; }
echo "[h-08 $VERSION] START MODE=analyze CHICAGO=$(now) RUN=$RUN LOG=$LOG"
echo "[h-08 $VERSION] H07_RUN=$H07_RUN THRESHOLDS=$THRESHOLDS PRIMARY=$PRIMARY_THRESHOLD GRAM_SIZE=$GRAM_SIZE"
step PREFLIGHT
for file in "$PYTHON_SCRIPT" "$CODE_MAP" "$H07_RECORDS" "$H07_AUDIT"; do [[ -f "$file" ]] || { echo "Missing: $file" >&2; exit 2; }; done
[[ -d "$GROUND_TRUTH_DIR" ]] || { echo "Missing Ground Truth: $GROUND_TRUTH_DIR" >&2; exit 2; }
python -c 'import tree_sitter,tree_sitter_java' || exit 2
pass
step PAIR_AST_CLASSIFICATION
python -u "$PYTHON_SCRIPT" \
  --code-map "$CODE_MAP" \
  --h07-records "$H07_RECORDS" \
  --h07-audit "$H07_AUDIT" \
  --ground-truth-dir "$GROUND_TRUTH_DIR" \
  --output "$RUN" \
  --expected-code-sha256 "$EXPECTED_CODE_SHA256" \
  --expected-h07-records-sha256 "$EXPECTED_H07_RECORDS_SHA256" \
  --timezone "$TIMEZONE" \
  --tree-sitter-version "$TREE_SITTER_VERSION" \
  --tree-sitter-java-version "$TREE_SITTER_JAVA_VERSION" \
  --expected-code-rows "$EXPECTED_CODE_ROWS" \
  --expected-csv-files "$EXPECTED_CSV_FILES" \
  --expected-pairs "$EXPECTED_PAIRS" \
  --expected-positive "$EXPECTED_POSITIVE" \
  --expected-negative "$EXPECTED_NEGATIVE" \
  --thresholds "$THRESHOLDS" \
  --primary-threshold "$PRIMARY_THRESHOLD" \
  --gram-size "$GRAM_SIZE" \
  --sample-per-type "$SAMPLE_PER_TYPE" \
  --progress-interval "$PROGRESS_INTERVAL" \
  --max-ast-comparison-cells "$MAX_AST_COMPARISON_CELLS"
pass
