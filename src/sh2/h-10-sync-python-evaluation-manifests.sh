#!/usr/bin/env bash
set -Eeuo pipefail

# H-10 v3: all run configuration is explicit in Shell; Python implements only logic.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERSION="v3"
GROUND_TRUTH_RELATIVE_PATH="data/evaluation/ground_truth/python/verified_pairs.jsonl"
GROUND_TRUTH="$ROOT/$GROUND_TRUTH_RELATIVE_PATH"
LEGACY_RELATIVE_PATH="data/evaluation/code/python.jsonl"
LEGACY_MAP="$ROOT/$LEGACY_RELATIVE_PATH"
MANIFEST="$ROOT/data/evaluation/manifest.json"
MANIFEST_DETAILS="$ROOT/data/evaluation/manifest_details.json"
ARCHIVE_DIR="$ROOT/data/evaluation/ground_truth_archives/h10-python-legacy-code-map"
BACKUP_DIR="$ROOT/data/evaluation/ground_truth_archives/h10-python-manifest-sync"
AUDIT_DIR="$ROOT/results/h-10/python/legacy-code-map-retirement"
PYTHON_SCRIPT="$ROOT/src/py2/h-10-sync-python-evaluation-manifests.py"

EXPECTED_PAIRS=7942
EXPECTED_POSITIVE=3971
EXPECTED_NEGATIVE=3971
EXPECTED_PROJECTS=24
EXPECTED_GT_STATUS="D05BJ_V2_BINARY_GROUND_TRUTH_CONSOLIDATED"
EXPECTED_GT_SHA256="2d171cdee52f6ab8229853b0cc5590b71e1a7dd6919a417dd88fb17d0ec0cb8c"
EXPECTED_LEGACY_CODE_IDS=3006
EXPECTED_LEGACY_SHA256="c048816b5755f7d8c60b92ba524c3eae74e35cc352dab2cb0e988b025e2bd944"
POSITIVE_FILENAME="positive_verified_pairs.jsonl"
NEGATIVE_FILENAME="negative_verified_pairs.jsonl"

LOG_DIR="$ROOT/logs/h-10/python/legacy-code-map-retirement"
LOG_PREFIX="h-10-python-legacy-map-retirement"
LOG_TIMESTAMP_FORMAT="%Y%m%d-%H%M%S-%N"
LOG_TZ="America/Chicago"

if [[ $# -ne 1 || ( "$1" != "--preflight" && "$1" != "--apply" ) ]]; then
    echo "Usage: bash src/sh2/h-10-sync-python-evaluation-manifests.sh --preflight|--apply" >&2
    exit 2
fi
MODE="${1#--}"
mkdir -p "$LOG_DIR"
STAMP="$(TZ="$LOG_TZ" date +"$LOG_TIMESTAMP_FORMAT")-$$"
LOG="$LOG_DIR/$LOG_PREFIX-$VERSION-$STAMP.log"
( set -o noclobber; : > "$LOG" )
exec > >(tee -a "$LOG") 2>&1
START=$SECONDS
now() { TZ="$LOG_TZ" date '+%Y-%m-%dT%H:%M:%S%z'; }
step() { printf '[h-10 %s] STEP=%s STATUS=%s CHICAGO=%s ELAPSED_SECONDS=%s\n' \
    "$VERSION" "$1" "$2" "$(now)" "$((SECONDS-START))"; }
finish() {
    local rc=$?
    trap - EXIT
    if (( rc == 0 )); then step FINAL PASS; else step FINAL FAIL; fi
    printf '[h-10 %s] EXIT=%s MODE=%s LOG=%s DURATION_SECONDS=%s\n' \
        "$VERSION" "$rc" "$MODE" "$LOG" "$((SECONDS-START))"
    exit "$rc"
}
trap finish EXIT

run_python() {
    local operation="$1"
    python "$PYTHON_SCRIPT" \
        --mode "$operation" --version "$VERSION" --root "$ROOT" \
        --ground-truth "$GROUND_TRUTH" --gt-relative-path "$GROUND_TRUTH_RELATIVE_PATH" \
        --legacy-map "$LEGACY_MAP" --legacy-relative-path "$LEGACY_RELATIVE_PATH" \
        --manifest "$MANIFEST" --manifest-details "$MANIFEST_DETAILS" \
        --archive-dir "$ARCHIVE_DIR" --backup-dir "$BACKUP_DIR" \
        --audit-dir "$AUDIT_DIR" --expected-pairs "$EXPECTED_PAIRS" \
        --expected-positive "$EXPECTED_POSITIVE" --expected-negative "$EXPECTED_NEGATIVE" \
        --expected-projects "$EXPECTED_PROJECTS" --expected-gt-status "$EXPECTED_GT_STATUS" \
        --expected-gt-sha256 "$EXPECTED_GT_SHA256" \
        --expected-legacy-code-ids "$EXPECTED_LEGACY_CODE_IDS" \
        --expected-legacy-sha256 "$EXPECTED_LEGACY_SHA256" \
        --positive-filename "$POSITIVE_FILENAME" --negative-filename "$NEGATIVE_FILENAME" \
        --timezone "$LOG_TZ"
}

step FILES_AND_DEPENDENCIES START
[[ -f "$PYTHON_SCRIPT" && -f "$GROUND_TRUTH" && -f "$LEGACY_MAP" && \
   -f "$MANIFEST" && -f "$MANIFEST_DETAILS" ]] || {
    echo "Required Python implementation or dataset/manifest is missing"; exit 1;
}
step FILES_AND_DEPENDENCIES PASS
step DATASET_CODE_AND_MANIFEST_PREFLIGHT START
run_python preflight
step DATASET_CODE_AND_MANIFEST_PREFLIGHT PASS
if [[ "$MODE" == "apply" ]]; then
    step ARCHIVE_VERIFY_MANIFEST_SYNC_AND_RETIRE START
    run_python apply
    step ARCHIVE_VERIFY_MANIFEST_SYNC_AND_RETIRE PASS
fi
