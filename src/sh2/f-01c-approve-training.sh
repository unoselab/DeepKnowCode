#!/usr/bin/env bash
# F-01c v22: explicit training approval. Does not start GPU training.
# Run from repository root after v20 LABEL_REVIEW_ALL_NON_CLONE.
# F01C_PREPARED_REVIEW_RUN="$PWD/results/f-01c/codellama-7b/2026-1112-115525-prepare" \
# F01C_LABEL_REVIEW_DECISIONS="$PWD/results/f-01c/codellama-7b/2026-1112-120407-1794506647-222708-label-review-export/label_review_decisions.json" \
# F01C_LABEL_VERIFICATION_REPORT="$PWD/results/f-01c/codellama-7b/2026-1112-124523-1794509123-224320-label-review-verify/label_review_verification.json" \
# bash src/sh2/f-01c-approve-training.sh approve
# Then use the printed OUT=/.../training-approval folder in main --smoke/--train.
# Every run logs stdout/stderr with Chicago timestamp and unique file; no overwrite.
set -Eeuo pipefail
export TZ=America/Chicago
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PYTHON_BIN="${F01C_PYTHON_BIN:-python}"
MODE="${1:?Use approve}"; [[ "$MODE" == approve && $# -eq 1 ]] || { echo 'Usage: approve' >&2; exit 2; }
PREPARED="${F01C_PREPARED_REVIEW_RUN:?Required prepared run}"
DECISIONS="${F01C_LABEL_REVIEW_DECISIONS:?Required reviewed decision file}"
VERIFICATION="${F01C_LABEL_VERIFICATION_REPORT:?Required label verification report}"
for file in "$DECISIONS" "$VERIFICATION" "$PREPARED/prepared/manifest.json" "$PREPARED/prepared/train.jsonl" "$PREPARED/prepared/validation.jsonl"; do
  [[ -f "$file" ]] || { echo "Missing required file: $file" >&2; exit 2; }
done
EXPECTED_TRAIN_SHA256="e2ec9d2cccc0adb0b73f81f60b3041b5a7f4fa519729f501c1cdc583c1e9b564"
EXPECTED_VALIDATION_SHA256="4404f90262c1021917ce31f0c454f87a5a634a0875307b8e84bbd51abefc752a"
EXPECTED_MANIFEST_SHA256="149b9439b8eeb2d5dbdf81931cd621b962e02066de0467ab330ce2a7fd933579"
LOG_DIR="$ROOT/logs/f-01c/codellama-7b"
RESULTS_DIR="$ROOT/results/f-01c/codellama-7b"
LOG_PREFIX='f-01c-v22-training-approval'
LOG_TIMESTAMP_FORMAT='+%Y-%m%d-%H%M%S'
CHICAGO_TIMESTAMP="$(TZ=America/Chicago date "$LOG_TIMESTAMP_FORMAT")"
UNIQUE="$(date +%s)-$$"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"
LOG_FILE="$LOG_DIR/$LOG_PREFIX-$CHICAGO_TIMESTAMP-$UNIQUE.log"
OUT="$RESULTS_DIR/$CHICAGO_TIMESTAMP-$UNIQUE-training-approval"
( set -o noclobber; : > "$LOG_FILE" ) || exit 2
finish() { rc=$?; trap - EXIT; printf '[F-01c v22] EXIT=%s CHICAGO=%s LOG=%s OUT=%s\n' "$rc" "$(TZ=America/Chicago date -Is)" "$LOG_FILE" "$OUT" | tee -a "$LOG_FILE"; exit "$rc"; }
trap finish EXIT
{
  echo "[F-01c v22] EXPLICIT_APPROVAL CHICAGO=$(TZ=America/Chicago date -Is) PREPARED=$PREPARED"
  "$PYTHON_BIN" -u "$ROOT/src/py2/f01c_training_approval.py" approve \
    --prepared-run "$PREPARED" --decisions "$DECISIONS" --verification "$VERIFICATION" \
    --approval-report "$OUT/training_approval.json" --output "$OUT" \
    --expected-train-sha256 "$EXPECTED_TRAIN_SHA256" \
    --expected-validation-sha256 "$EXPECTED_VALIDATION_SHA256" \
    --expected-manifest-sha256 "$EXPECTED_MANIFEST_SHA256"
} 2>&1 | tee -a "$LOG_FILE"
