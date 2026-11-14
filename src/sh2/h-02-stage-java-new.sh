#!/usr/bin/env bash
# H-02 v3 — standalone, non-destructive staging of 48 curated Java CSVs.
# Does not invoke any other H-Series shell wrapper.
set -Eeuo pipefail
export TZ='America/Chicago'
export PYTHONUNBUFFERED=1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PYTHON_BIN='python'
SCRIPT="$ROOT/src/py2/h-02-stage-java-new.py"
EXPERIMENT='h-02-stage-java-new'
VERSION='v3'
EXECUTION_MODE='stage-curated-legacy-java-csv'
LOG_DIR="$ROOT/logs/h-02/java-new"
RESULT_PARENT="$ROOT/results/h-02/java-new"
LOG_PREFIX='h-02-stage-java-new-v3'
LOG_TIMESTAMP_FORMAT='%Y%m%d-%H%M%S-%N'
LOG_EVENT_FORMAT='%Y-%m-%dT%H:%M:%S%z'
STAMP="$(TZ=America/Chicago date +"$LOG_TIMESTAMP_FORMAT")-$$"
RUN_DIR="$RESULT_PARENT/$STAMP"
LOG_FILE="$LOG_DIR/$LOG_PREFIX-$STAMP.log"

# Explicit, pinned source evidence from the user's successful 2026-11-14 run.
H02_RUN="$ROOT/results/h-02/java/20261114-015707-702319464-15844"
H03_PREPARE_RUN="$ROOT/results/h-03/java/20261114-015824-982476124-15969"
H04_RUN="$ROOT/results/h-04/java/20261114-015907-491511711-16044"
SOURCE_VERSION='v2'
ORIGINAL_CSV_DIR="$ROOT/../2026JournalData/RQ4/c-evaluate_clone/java/data/data-java/ground_truth_test_refactorability"
CODE_MAP="$ROOT/data/evaluation/code/java.jsonl"
DESTINATION="$ROOT/data/evaluation/ground_truth/java_new"
EXPECTED_CODE_MAP_SHA256='923418867148fb1c1d7ec37d449adfad2d24bc98d1d08be2086c0d75d2c4c4be'
EXPECTED_H02_MANIFEST_SHA256='d9022957e958bc5d6cde1bccf80c251a0fe34e803d30ffb57062290b1e5c3035'
EXPECTED_G02_MANIFEST_SHA256='61968384ce8855105a08658029c4e27825faf53d3fa93bb4211b3bca7438d292'
EXPECTED_H02_STATUS='H02_V2_CONFLICT_GROUP_EXCLUDED_LEGACY_LABELS_UNVERIFIED'
EXPECTED_H03_STATUS='H03_V2_REVIEW_PACKETS_READY_NOT_GOLD'
EXPECTED_H04_STATUS='H04_V2_EXACT_NORMALIZED_ISOLATION_PASS'
SEMANTIC_REVIEW_SCOPE='OUT_OF_SCOPE_BY_PROTOCOL'
EXPECTED_PROJECTS=48
EXPECTED_ORIGINAL_PAIRS=35194
EXPECTED_PAIRS=35192
EXPECTED_POSITIVE=17596
EXPECTED_NEGATIVE=17596
EXPECTED_UNIQUE_CODE_PAIRS=34138
EXPECTED_EXCLUDED_ROWS=2
EXPECTED_EXCLUDED_PAIR_KEY='3ffdaead7b2aa62790bc09bfedc2cca4f6b12c042c200fec7a92bb585dc41b40'
EXPECTED_EXCLUDED_SOURCE_ROWS='pred_activemq_with_refactorability.csv:483,pred_activemq_with_refactorability.csv:613'
EXPECTED_TRAIN_PAIRS=19464
EXPECTED_VAL_PAIRS=512

now() { TZ='America/Chicago' date +"$LOG_EVENT_FORMAT"; }
step() {
  local label="$1"; shift
  local start end rc
  start="$(date +%s)"
  printf '[%s %s] STEP=%s START CHICAGO=%s\n' "$EXPERIMENT" "$VERSION" "$label" "$(now)"
  set +e
  "$@"
  rc=$?
  set -e
  end="$(date +%s)"
  if (( rc == 0 )); then
    printf '[%s %s] STEP=%s PASS ELAPSED_SECONDS=%s CHICAGO=%s\n' "$EXPERIMENT" "$VERSION" "$label" "$((end-start))" "$(now)"
  else
    printf '[%s %s] STEP=%s FAIL EXIT=%s ELAPSED_SECONDS=%s CHICAGO=%s\n' "$EXPERIMENT" "$VERSION" "$label" "$rc" "$((end-start))" "$(now)"
  fi
  return "$rc"
}
call_python() {
  local mode="$1"
  local -a args=(
    --mode "$mode" --root "$ROOT" --destination "$DESTINATION" --output-dir "$RUN_DIR"
    --h02-run "$H02_RUN" --h03-prepare-run "$H03_PREPARE_RUN" --h04-run "$H04_RUN"
    --original-csv-dir "$ORIGINAL_CSV_DIR" --code-map "$CODE_MAP"
    --source-version "$SOURCE_VERSION" --expected-code-map-sha256 "$EXPECTED_CODE_MAP_SHA256"
    --expected-h02-manifest-sha256 "$EXPECTED_H02_MANIFEST_SHA256"
    --expected-g02-manifest-sha256 "$EXPECTED_G02_MANIFEST_SHA256"
    --expected-h02-status "$EXPECTED_H02_STATUS" --expected-h03-status "$EXPECTED_H03_STATUS"
    --expected-h04-status "$EXPECTED_H04_STATUS" --semantic-review-scope "$SEMANTIC_REVIEW_SCOPE"
    --expected-projects "$EXPECTED_PROJECTS" --expected-original-pairs "$EXPECTED_ORIGINAL_PAIRS"
    --expected-pairs "$EXPECTED_PAIRS" --expected-positive "$EXPECTED_POSITIVE"
    --expected-negative "$EXPECTED_NEGATIVE" --expected-unique-code-pairs "$EXPECTED_UNIQUE_CODE_PAIRS"
    --expected-excluded-rows "$EXPECTED_EXCLUDED_ROWS"
    --expected-excluded-pair-key "$EXPECTED_EXCLUDED_PAIR_KEY"
    --expected-excluded-source-rows "$EXPECTED_EXCLUDED_SOURCE_ROWS"
    --expected-train-pairs "$EXPECTED_TRAIN_PAIRS" --expected-val-pairs "$EXPECTED_VAL_PAIRS"
  )
  "$PYTHON_BIN" -u "$SCRIPT" "${args[@]}"
}
main() {
  [[ $# -eq 1 && "$1" == '--stage' ]] || {
    echo "Usage: bash src/sh2/h-02-stage-java-new.sh --stage" >&2
    return 2
  }
  [[ -f "$SCRIPT" ]] || { echo "Missing Python implementation $SCRIPT" >&2; return 2; }
  [[ ! -e "$DESTINATION" ]] || {
    echo "BLOCKED: existing destination will not be overwritten: $DESTINATION" >&2
    return 2
  }
  mkdir -p "$RESULT_PARENT"
  mkdir "$RUN_DIR" || { echo "Refusing existing output run $RUN_DIR" >&2; return 2; }
  printf '[%s %s] START MODE=%s CHICAGO=%s RUN=%s LOG=%s\n' "$EXPERIMENT" "$VERSION" "$EXECUTION_MODE" "$(now)" "$RUN_DIR" "$LOG_FILE"
  printf '[%s %s] SOURCE_H02=%s SOURCE_H03=%s SOURCE_H04=%s DESTINATION=%s\n' \
    "$EXPERIMENT" "$VERSION" "$H02_RUN" "$H03_PREPARE_RUN" "$H04_RUN" "$DESTINATION"
  step 'PREFLIGHT' call_python preflight
  step 'STAGE_CURATED_CSV' call_python stage
  printf '[%s %s] OUTPUT_MANIFEST=%s\n' "$EXPERIMENT" "$VERSION" "$RUN_DIR/java_new_manifest.json"
}
mkdir -p "$LOG_DIR"
( set -C; : > "$LOG_FILE" ) || { echo "Refusing to overwrite log $LOG_FILE" >&2; exit 2; }
start_epoch="$(date +%s)"
set +e
main "$@" 2>&1 | tee -a "$LOG_FILE"
rc="${PIPESTATUS[0]}"
set -e
end_epoch="$(date +%s)"
if (( rc == 0 )); then FINAL_STATUS='PASS'; else FINAL_STATUS='FAIL'; fi
printf '[%s %s] FINAL_STATUS=%s EXIT=%s DURATION_SECONDS=%s CHICAGO=%s DESTINATION=%s RUN=%s LOG=%s\n' \
  "$EXPERIMENT" "$VERSION" "$FINAL_STATUS" "$rc" "$((end_epoch-start_epoch))" "$(now)" \
  "$DESTINATION" "$RUN_DIR" "$LOG_FILE" | tee -a "$LOG_FILE"
exit "$rc"
