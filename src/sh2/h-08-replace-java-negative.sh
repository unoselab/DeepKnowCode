#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERSION="v1"
PROJECT="activemq"
LEFT_ID="activemq_1032_727"
RIGHT_ID="activemq_1202_1047"
TARGET_CSV="$ROOT/data/evaluation/ground_truth/java/pred_activemq_with_refactorability.csv"
GROUND_TRUTH_DIR="$ROOT/data/evaluation/ground_truth/java"
CODE_MAP="$ROOT/data/evaluation/code/java.jsonl"
CODE_SHA256="923418867148fb1c1d7ec37d449adfad2d24bc98d1d08be2086c0d75d2c4c4be"
ARCHIVE_DIR="$ROOT/data/evaluation/ground_truth_archives/h08-manual-replacements"
JOURNAL_DIR="$ROOT/results/h-08/java/manual-curation"
LOG_DIR="$ROOT/logs/h-08/java/manual-curation"
LOG_PREFIX="h-08-negative-replacement-v1"
LOG_TIMESTAMP_FORMAT="%Y%m%d-%H%M%S-%N"
EXPECTED_CSV_FILES=48
EXPECTED_POSITIVE=17596
EXPECTED_NEGATIVE_BEFORE=17595
EXPECTED_PAIRS_BEFORE=35191
EXPECTED_TARGET_ROWS_BEFORE=1443
# Human-reviewed pair-level metadata; verify schema semantics before applying.
PAIR_IN_TYPE_OK="False"
PAIR_OUT_TYPE_OK="False"
PAIR_TYPE_OK="False"
PAIR_REFACTORABLE="False"
MODE="${1:---preflight}"
case "$MODE" in --preflight) MODE=preflight;; --apply) MODE=apply;; *) echo 'Usage: bash ... --preflight|--apply' >&2;exit 2;; esac
mkdir -p "$LOG_DIR"
STAMP="$(TZ=America/Chicago date +"$LOG_TIMESTAMP_FORMAT")-$$"
LOG="$LOG_DIR/$LOG_PREFIX-$STAMP.log"
(umask 077; set -C; : > "$LOG") || { echo "Cannot create unique log: $LOG" >&2; exit 1; }
START=$(date +%s)
exec > >(tee -a "$LOG") 2>&1
export TZ=America/Chicago
log(){ echo "[h-08 negative v1] STEP=$1 STATUS=$2 CHICAGO=$(date +%FT%T%z) ELAPSED_SECONDS=$(($(date +%s)-START))"; }
finish(){ local rc=$?; trap - EXIT; if ((rc==0));then log FINAL PASS;else log FINAL FAIL;fi; echo "EXIT=$rc LOG=$LOG";exit "$rc"; }
trap finish EXIT
log PREFLIGHT START
python "$ROOT/src/py2/h-08-replace-java-negative.py" \
 --target "$TARGET_CSV" --ground-truth-dir "$GROUND_TRUTH_DIR" \
 --archive-dir "$ARCHIVE_DIR" --journal-dir "$JOURNAL_DIR" \
 --code-map "$CODE_MAP" --original-code-sha256 "$CODE_SHA256" \
 --left-id "$LEFT_ID" --right-id "$RIGHT_ID" --project "$PROJECT" \
 --in-type-ok "$PAIR_IN_TYPE_OK" --out-type-ok "$PAIR_OUT_TYPE_OK" \
 --type-ok "$PAIR_TYPE_OK" --refactorable "$PAIR_REFACTORABLE" \
 --expected-csv-files "$EXPECTED_CSV_FILES" --expected-positive "$EXPECTED_POSITIVE" \
 --expected-negative-before "$EXPECTED_NEGATIVE_BEFORE" \
 --expected-pairs-before "$EXPECTED_PAIRS_BEFORE" \
 --expected-target-rows-before "$EXPECTED_TARGET_ROWS_BEFORE" \
 --version "$VERSION" --mode "$MODE"
log "$MODE" PASS
