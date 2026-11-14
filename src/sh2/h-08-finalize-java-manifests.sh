#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERSION="v1"
DATASET_DIR="data/evaluation/ground_truth/java"
COMPACT_MANIFEST="data/evaluation/manifest.json"
DETAILED_MANIFEST="data/evaluation/manifest_details.json"
CODE_MAP="data/evaluation/code/java.jsonl"
TARGET_CSV="pred_activemq_with_refactorability.csv"
EXPECTED_CODE_SHA256="923418867148fb1c1d7ec37d449adfad2d24bc98d1d08be2086c0d75d2c4c4be"
EXPECTED_TARGET_SHA256="5d91eb5f1cd0ec9edffab8c75e45282e64ae56e554170f7493618ce1b6ab14b6"
OLD_LEFT="activemq_119_159"; OLD_RIGHT="activemq_130_271"
NEW_LEFT="activemq_1032_727"; NEW_RIGHT="activemq_1202_1047"
EXPECTED_CSV_FILES=48; EXPECTED_PAIRS=35192; EXPECTED_POSITIVE=17596; EXPECTED_NEGATIVE=17596
EXPECTED_TARGET_ROWS=1444; EXPECTED_CODE_ROWS=33658; EXPECTED_REFERENCED_CODE_IDS=33658
BACKUP_DIR="data/evaluation/ground_truth_archives/h08-manifest-finalization"
RESULT_DIR="results/h-08/java/manifest-finalization"
LOG_DIR="$ROOT/logs/h-08/java/manifest-finalization"
LOG_PREFIX="h-08-java-manifest-finalization-v1"
LOG_TIMESTAMP_FORMAT="%Y%m%d-%H%M%S-%N"
MODE="${1:---preflight}"
case "$MODE" in --preflight) MODE=preflight;; --apply) MODE=apply;; *) echo 'Usage: bash src/sh2/h-08-finalize-java-manifests.sh --preflight|--apply' >&2;exit 2;; esac
export TZ=America/Chicago
mkdir -p "$LOG_DIR"
STAMP="$(date +"$LOG_TIMESTAMP_FORMAT")-$$"
LOG="$LOG_DIR/$LOG_PREFIX-$STAMP.log"
(umask 077; set -C; : > "$LOG") || exit 1
START=$(date +%s)
exec > >(tee -a "$LOG") 2>&1
log(){ echo "[h-08 manifest $VERSION] STEP=$1 STATUS=$2 CHICAGO=$(date +%FT%T%z) ELAPSED_SECONDS=$(($(date +%s)-START))"; }
finish(){ local rc=$?;trap - EXIT;if ((rc==0));then log FINAL PASS;else log FINAL FAIL;fi;echo "EXIT=$rc LOG=$LOG";exit "$rc"; }
trap finish EXIT
log PREFLIGHT START
python "$ROOT/src/py2/h-08-finalize-java-manifests.py" \
 --root "$ROOT" --dataset-dir "$DATASET_DIR" --compact-manifest "$COMPACT_MANIFEST" \
 --detailed-manifest "$DETAILED_MANIFEST" --code-map "$CODE_MAP" \
 --target-csv "$TARGET_CSV" --expected-code-sha256 "$EXPECTED_CODE_SHA256" \
 --expected-target-sha256 "$EXPECTED_TARGET_SHA256" \
 --old-left "$OLD_LEFT" --old-right "$OLD_RIGHT" --new-left "$NEW_LEFT" --new-right "$NEW_RIGHT" \
 --expected-csv-files "$EXPECTED_CSV_FILES" --expected-pairs "$EXPECTED_PAIRS" \
 --expected-positive "$EXPECTED_POSITIVE" --expected-negative "$EXPECTED_NEGATIVE" \
 --expected-target-rows "$EXPECTED_TARGET_ROWS" --expected-code-rows "$EXPECTED_CODE_ROWS" \
 --expected-referenced-code-ids "$EXPECTED_REFERENCED_CODE_IDS" \
 --backup-dir "$BACKUP_DIR" --result-dir "$RESULT_DIR" --run-id "$STAMP" \
 --version "$VERSION" --mode "$MODE"
log "$MODE" PASS
