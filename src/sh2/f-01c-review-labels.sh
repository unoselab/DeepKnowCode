#!/usr/bin/env bash
# F-01c v18 independent human negative-label review wrapper.
# Run from DeepKnowCode root; all configuration in this Shell script.
# Step 1: F01C_PREPARED_REVIEW_RUN="$PWD/results/f-01c/codellama-7b/2026-1112-115525-prepare" bash src/sh2/f-01c-review-labels.sh export
# Step 2: Inspect generated label_review_full_code.md and edit label_review_decisions.json.
# Step 3: F01C_PREPARED_REVIEW_RUN="..." F01C_LABEL_REVIEW_DECISIONS="/absolute/path/label_review_decisions.json" bash src/sh2/f-01c-review-labels.sh verify
# This command NEVER approves GPU training; verified label evidence is an input to later approval.
set -Eeuo pipefail
export TZ=America/Chicago
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PYTHON_BIN="${F01C_PYTHON_BIN:-python}"
PYFILE="$ROOT/src/py2/f01c_label_review.py"
MODE="${1:?Use export or verify}"
[[ "$MODE" == export || "$MODE" == verify ]] || { echo 'Mode must be export or verify' >&2; exit 2; }
[[ $# -eq 1 ]] || { echo 'Unexpected extra positional parameters' >&2; exit 2; }
PREPARED_RUN="${F01C_PREPARED_REVIEW_RUN:?Explicit F01C_PREPARED_REVIEW_RUN required}"
DECISIONS="${F01C_LABEL_REVIEW_DECISIONS:-$ROOT/results/f-01c/codellama-7b/review-not-yet-submitted.json}"
EXPECTED_TRAIN_SHA256="e2ec9d2cccc0adb0b73f81f60b3041b5a7f4fa519729f501c1cdc583c1e9b564"
EXPECTED_VAL_SHA256="4404f90262c1021917ce31f0c454f87a5a634a0875307b8e84bbd51abefc752a"
EXPECTED_MANIFEST_SHA256="149b9439b8eeb2d5dbdf81931cd621b962e02066de0467ab330ce2a7fd933579"
EXPECTED_PAIR_COUNT=8
MAX_SEQ_LENGTH=2048
LOG_DIR="$ROOT/logs/f-01c/codellama-7b"
LOG_PREFIX="f-01c-v18-label-review"
LOG_TIMESTAMP_FORMAT='+%Y-%m%d-%H%M%S'
CHICAGO_TIMESTAMP="$(TZ=America/Chicago date "$LOG_TIMESTAMP_FORMAT")"
mkdir -p "$LOG_DIR" "$ROOT/results/f-01c/codellama-7b"
UNIQUE="$(date +%s)-$$"
LOG_FILE="$LOG_DIR/$LOG_PREFIX-$MODE-$CHICAGO_TIMESTAMP-$UNIQUE.log"
OUT="$ROOT/results/f-01c/codellama-7b/$CHICAGO_TIMESTAMP-$UNIQUE-label-review-$MODE"
( set -o noclobber; : > "$LOG_FILE" ) || { echo 'Refusing log overwrite' >&2; exit 2; }
finish() { rc=$?; trap - EXIT; printf '[F-01c v18] EXIT=%s MODE=%s CHICAGO=%s LOG=%s OUT=%s\n' "$rc" "$MODE" "$(TZ=America/Chicago date -Is)" "$LOG_FILE" "$OUT" | tee -a "$LOG_FILE"; exit "$rc"; }
trap finish EXIT
{
  echo "[F-01c v18] MODE=$MODE CHICAGO=$(TZ=America/Chicago date -Is) PREPARED=$PREPARED_RUN OUT=$OUT"
  "$PYTHON_BIN" -u "$PYFILE" "$MODE" \
    --prepared-run "$PREPARED_RUN" --output "$OUT" --decisions "$DECISIONS" \
    --expected-train-sha256 "$EXPECTED_TRAIN_SHA256" \
    --expected-val-sha256 "$EXPECTED_VAL_SHA256" \
    --expected-manifest-sha256 "$EXPECTED_MANIFEST_SHA256" \
    --expected-pairs "$EXPECTED_PAIR_COUNT" --max-tokens "$MAX_SEQ_LENGTH"
} 2>&1 | tee -a "$LOG_FILE"
