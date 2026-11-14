#!/usr/bin/env bash
# H-SERIES v2 / h-05 (standalone Shell; no nested shell wrapper).
# Shell owns ALL environment, input paths, hashes, quantities and output paths.
set -Eeuo pipefail
export TZ='America/Chicago'
export PYTHONUNBUFFERED=1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PYTHON_BIN='python'
SCRIPT="$ROOT/src/py2/h-java-ground-truth.py"
EXPERIMENT='h-05'
VERSION='v2'
LOG_PREFIX='h-05-v2'
LOG_TIMESTAMP_FORMAT='%Y%m%d-%H%M%S-%N'
LOG_EVENT_FORMAT='%Y-%m-%dT%H:%M:%S%z'
LOG_DIR="$ROOT/logs/$EXPERIMENT/java"
RESULT_PARENT="$ROOT/results/$EXPERIMENT/java"
STAMP="$(TZ=America/Chicago date +"$LOG_TIMESTAMP_FORMAT")-$$"
RUN_DIR="$RESULT_PARENT/$STAMP"
LOG_FILE="$LOG_DIR/$LOG_PREFIX-$STAMP.log"
SOURCE_DIR="$ROOT/../2026JournalData/RQ4/c-evaluate_clone/java/data/data-java/ground_truth_test_refactorability"
CODE_MAP="$ROOT/data/evaluation/code/java.jsonl"
EXPECTED_CODE_SHA256='923418867148fb1c1d7ec37d449adfad2d24bc98d1d08be2086c0d75d2c4c4be'
EXPECTED_PROJECTS=48
EXPECTED_PAIRS=35194
EXPECTED_POSITIVE=17597
EXPECTED_NEGATIVE=17597
EXPECTED_CODE_ROWS=33658
REFERENCE_PROJECTS=48
REFERENCE_POSITIVE=70262
REFERENCE_NEGATIVE=70262
G02_PREPARED="$ROOT/results/g-02/gpt-oss-120b/2026-1113-123203-1794594723-285189-27336-pipeline/00-prepare/prepared"
EXPECTED_G02_TRAIN_SHA256='546142986ff39e08ccf38db6b6c12284c200f5bf20775c1b0b97f1a34fb63eeb'
EXPECTED_G02_VALIDATION_SHA256='cf78092416d3ca8d039daf5c722490ec60d419232d17cc7a812f6a125d5b686c'
EXPECTED_G02_PREPARED_MANIFEST_SHA256='61968384ce8855105a08658029c4e27825faf53d3fa93bb4211b3bca7438d292'
EXPECTED_G02_TRAIN_ROWS=19464
EXPECTED_G02_VALIDATION_ROWS=512
MAX_OVERLAP_EXAMPLES=100
H02_RUN=''
H03_PREPARE_RUN=''
H03_RUN=''
H04_RUN=''
REVIEWER_A=''
REVIEWER_B=''
FROZEN_DIR=''
AUTHORIZE_DEDUP='false'
ACCEPT_EXACT_NORMALIZED_SCOPE='false'
# Immutable, operator-approved exclusion of BOTH conflicting source occurrences.
EXCLUDE_CONFLICT_PAIR_KEY='3ffdaead7b2aa62790bc09bfedc2cca4f6b12c042c200fec7a92bb585dc41b40'
EXCLUDE_CONFLICT_SOURCE_ROWS='pred_activemq_with_refactorability.csv:483,pred_activemq_with_refactorability.csv:613'
EXPECTED_CONFLICT_GROUPS_BEFORE_EXCLUSION=1
EXPECTED_EXCLUDED_ROWS=2
EXPECTED_CURATED_PAIRS=35192
EXPECTED_CURATED_POSITIVE=17596
EXPECTED_CURATED_NEGATIVE=17596
EXPECTED_CURATED_UNIQUE_CODE_PAIRS=34138
STAGE=''

now() { TZ=America/Chicago date +"$LOG_EVENT_FORMAT"; }

step() {
  local label="$1"; shift
  local begin end rc
  begin="$(date +%s)"
  printf '[%s %s] STEP=%s START CHICAGO=%s\n' "$EXPERIMENT" "$VERSION" "$label" "$(now)"
  set +e
  "$@"
  rc=$?
  set -e
  end="$(date +%s)"
  if (( rc == 0 )); then
    printf '[%s %s] STEP=%s PASS ELAPSED_SECONDS=%s CHICAGO=%s\n' "$EXPERIMENT" "$VERSION" "$label" "$((end-begin))" "$(now)"
  else
    printf '[%s %s] STEP=%s FAIL EXIT=%s ELAPSED_SECONDS=%s CHICAGO=%s\n' "$EXPERIMENT" "$VERSION" "$label" "$rc" "$((end-begin))" "$(now)"
  fi
  return "$rc"
}

run_stage() {
  local -a args=(
    "$STAGE" --root "$ROOT" --out-dir "$RUN_DIR"
    --source-dir "$SOURCE_DIR" --code-map "$CODE_MAP"
    --expected-code-sha256 "$EXPECTED_CODE_SHA256"
    --g02-prepared "$G02_PREPARED"
    --expected-g02-train-sha256 "$EXPECTED_G02_TRAIN_SHA256"
    --expected-g02-validation-sha256 "$EXPECTED_G02_VALIDATION_SHA256"
    --expected-g02-prepared-manifest-sha256 "$EXPECTED_G02_PREPARED_MANIFEST_SHA256"
    --expected-g02-train-rows "$EXPECTED_G02_TRAIN_ROWS"
    --expected-g02-validation-rows "$EXPECTED_G02_VALIDATION_ROWS"
    --h02-run "$H02_RUN" --h03-prepare-run "$H03_PREPARE_RUN"
    --h03-run "$H03_RUN" --h04-run "$H04_RUN"
    --reviewer-a "$REVIEWER_A" --reviewer-b "$REVIEWER_B"
    --frozen-dir "$FROZEN_DIR"
    --authorize-dedup "$AUTHORIZE_DEDUP"
    --accept-exact-normalized-scope "$ACCEPT_EXACT_NORMALIZED_SCOPE"
    --expected-projects "$EXPECTED_PROJECTS" --expected-pairs "$EXPECTED_PAIRS"
    --expected-positive "$EXPECTED_POSITIVE" --expected-negative "$EXPECTED_NEGATIVE"
    --expected-code-rows "$EXPECTED_CODE_ROWS" --reference-projects "$REFERENCE_PROJECTS"
    --reference-positive "$REFERENCE_POSITIVE" --reference-negative "$REFERENCE_NEGATIVE"
    --max-overlap-examples "$MAX_OVERLAP_EXAMPLES"
    --exclude-conflict-pair-key "$EXCLUDE_CONFLICT_PAIR_KEY"
    --exclude-conflict-source-rows "$EXCLUDE_CONFLICT_SOURCE_ROWS"
    --expected-conflict-groups-before-exclusion "$EXPECTED_CONFLICT_GROUPS_BEFORE_EXCLUSION"
    --expected-excluded-rows "$EXPECTED_EXCLUDED_ROWS"
    --expected-curated-pairs "$EXPECTED_CURATED_PAIRS"
    --expected-curated-positive "$EXPECTED_CURATED_POSITIVE"
    --expected-curated-negative "$EXPECTED_CURATED_NEGATIVE"
    --expected-curated-unique-code-pairs "$EXPECTED_CURATED_UNIQUE_CODE_PAIRS"
  )
  step "$STAGE" "$PYTHON_BIN" -u "$SCRIPT" "${args[@]}"
}

main() {
    [[ "${1:-}" == '--freeze' && $# -eq 6 ]] || { echo "Usage: bash $0 --freeze /abs/H02_RUN /abs/H03_VALIDATE_RUN /abs/H04_RUN --authorize-reviewed-dedup --accept-exact-normalized-scope" >&2; return 2; }
    [[ "$5" == '--authorize-reviewed-dedup' && "$6" == '--accept-exact-normalized-scope' ]] || { echo 'Explicit reviewed dedup and exact/normalized-only leakage scope acknowledgements are REQUIRED' >&2; return 2; }
    STAGE='h-05'; H02_RUN="$2"; H03_RUN="$3"; H04_RUN="$4"
    AUTHORIZE_DEDUP='true'; ACCEPT_EXACT_NORMALIZED_SCOPE='true'
    FROZEN_DIR="$ROOT/data/evaluation/ground_truth/java/frozen-h05-v2-$STAMP"
  [[ -f "$SCRIPT" ]] || { echo "Missing Python implementation: $SCRIPT" >&2; return 2; }
  if [[ "$STAGE" != 'h-06' ]]; then
    [[ -d "$SOURCE_DIR" && -f "$CODE_MAP" ]] || { echo "Missing Java source CSV directory or code map" >&2; return 2; }
  fi
  mkdir -p "$RESULT_PARENT"
  mkdir "$RUN_DIR" || { echo "Output run directory already exists: $RUN_DIR" >&2; return 2; }
  printf '[%s %s] START MODE=%s CHICAGO=%s RUN=%s LOG=%s\n' "$EXPERIMENT" "$VERSION" "$STAGE" "$(now)" "$RUN_DIR" "$LOG_FILE"
  printf '[%s %s] SOURCE=%s CODE_MAP_SHA256=%s H02=%s H03=%s H04=%s FROZEN=%s\n' "$EXPERIMENT" "$VERSION" "$SOURCE_DIR" "$EXPECTED_CODE_SHA256" "$H02_RUN" "$H03_RUN" "$H04_RUN" "$FROZEN_DIR"
  run_stage
}

mkdir -p "$LOG_DIR"
# Always create a UNIQUE log; do not overwrite any existing file.
( set -C; : > "$LOG_FILE" ) || { echo "Refusing to overwrite log: $LOG_FILE" >&2; exit 2; }
start_epoch="$(date +%s)"
set +e
main "$@" 2>&1 | tee -a "$LOG_FILE"
rc="${PIPESTATUS[0]}"
set -e
end_epoch="$(date +%s)"
if (( rc == 0 )); then final='PASS'; else final='FAIL'; fi
printf '[%s %s] FINAL_STATUS=%s EXIT=%s DURATION_SECONDS=%s CHICAGO=%s RUN=%s LOG=%s\n' "$EXPERIMENT" "$VERSION" "$final" "$rc" "$((end_epoch-start_epoch))" "$(now)" "$RUN_DIR" "$LOG_FILE" | tee -a "$LOG_FILE"
exit "$rc"
