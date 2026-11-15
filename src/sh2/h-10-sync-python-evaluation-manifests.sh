#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERSION="v2"
DATASET="$ROOT/data/evaluation/ground_truth/python/verified_pairs.jsonl"
CODE_MAP_RELATIVE_PATH="data/evaluation/code/python.jsonl"
CODE_MAP="$ROOT/$CODE_MAP_RELATIVE_PATH"
MANIFEST="$ROOT/data/evaluation/manifest.json"
MANIFEST_DETAILS="$ROOT/data/evaluation/manifest_details.json"
BACKUP_DIR="$ROOT/data/evaluation/ground_truth_archives/h10-python-manifest-sync"
AUDIT_DIR="$ROOT/results/h-10/python/manifest-sync"
LOG_DIR="$ROOT/logs/h-10/python/manifest-sync"
LOG_PREFIX="h-10-python-manifest-sync"
LOG_TIMESTAMP_FORMAT="%Y%m%d-%H%M%S-%N"
LOG_TZ="America/Chicago"
EXPECTED_PAIRS=7942
EXPECTED_POSITIVE=3971
EXPECTED_NEGATIVE=3971
EXPECTED_PROJECTS=24
# Historical SHA256 quoted by D-05bj v2 discussion; fail closed on mismatch.
EXPECTED_SHA256="2d171cdee52f6ab8229853b0cc5590b71e1a7dd6919a417dd88fb17d0ec0cb8c"
DATASET_RELATIVE_PATH="data/evaluation/ground_truth/python/verified_pairs.jsonl"
LABEL_FIELD="gold_label"
PROJECT_FIELD="project"
PROVENANCE="D-05bj v2 consolidated binary evaluation ground truth"
REFERENCE_DOCUMENT="README-1111-DeepKnowCode-Dataset.md"
PYTHON_SCRIPT="$ROOT/src/py2/h-10-sync-python-evaluation-manifests.py"

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
step() { echo "[h-10 $VERSION] STEP=$1 STATUS=$2 CHICAGO=$(now) ELAPSED_SECONDS=$((SECONDS-START))"; }
finish() {
  rc=$?
  trap - EXIT
  if ((rc == 0)); then step FINAL PASS; else step FINAL FAIL; fi
  echo "[h-10 $VERSION] EXIT=$rc MODE=$MODE LOG=$LOG DURATION_SECONDS=$((SECONDS-START))"
  exit "$rc"
}
trap finish EXIT
step PREFLIGHT START
[[ -f "$PYTHON_SCRIPT" ]] || { echo "Missing Python implementation: $PYTHON_SCRIPT"; exit 1; }
[[ -f "$DATASET" && -f "$CODE_MAP" && -f "$MANIFEST" && -f "$MANIFEST_DETAILS" ]] || { echo "Missing data or manifest"; exit 1; }
step PREFLIGHT PASS
step PYTHON_MANIFEST_SYNC START
python "$PYTHON_SCRIPT" \
  --mode "$MODE" --version "$VERSION" --dataset "$DATASET" \
  --code-map "$CODE_MAP" --code-map-relative-path "$CODE_MAP_RELATIVE_PATH" \
  --manifest "$MANIFEST" --manifest-details "$MANIFEST_DETAILS" \
  --backup-dir "$BACKUP_DIR" --audit-dir "$AUDIT_DIR" \
  --expected-pairs "$EXPECTED_PAIRS" --expected-positive "$EXPECTED_POSITIVE" \
  --expected-negative "$EXPECTED_NEGATIVE" --expected-projects "$EXPECTED_PROJECTS" \
  --expected-sha256 "$EXPECTED_SHA256" \
  --dataset-relative-path "$DATASET_RELATIVE_PATH" \
  --label-field "$LABEL_FIELD" --project-field "$PROJECT_FIELD" \
  --provenance "$PROVENANCE" --reference-document "$REFERENCE_DOCUMENT" \
  --timezone "$LOG_TZ"
step PYTHON_MANIFEST_SYNC PASS
