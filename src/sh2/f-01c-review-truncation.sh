#!/usr/bin/env bash
# Independent read-only v13 structural review wrapper. Does not invoke another wrapper.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="$ROOT/src/py2/f01c_one_sided_review.py"
VERSION="v13"
MODEL_MODE="codellama-7b"
# Shell owns all review inputs; select newest audit or use explicit environment override.
AUDIT_RUN="${F01C_REVIEW_AUDIT_RUN:-}"
if [[ -z "$AUDIT_RUN" ]]; then
  shopt -s nullglob
  runs=("$ROOT"/results/f-01c/"$MODEL_MODE"/*-structural-audit)
  [[ ${#runs[@]} -gt 0 ]] || { echo 'ERROR: no structural audit runs found' >&2; exit 2; }
  IFS=$'\n' runs=($(printf '%s\n' "${runs[@]}" | sort -r))
  unset IFS
  AUDIT_RUN="${runs[0]}"
fi
PROPOSALS="$AUDIT_RUN/structural_truncation_proposals.jsonl"
AUDIT="$AUDIT_RUN/structural_truncation_audit.json"
MAX_SEQ_LENGTH=2048
MAX_REMOVED_FOR_FOCUSED_REVIEW=12
MAX_TOKEN_REDUCTION_FOR_FOCUSED_REVIEW=400
PAIR_FILTER="${1:-all}"
LOG_DIR="$ROOT/logs/f-01c/$MODEL_MODE"
LOG_PREFIX="f-01c-v13-truncation-review"
LOG_TIMESTAMP_FORMAT="+%Y-%m%d-%H%M%S"
LOG_SAVE_STDOUT=true
LOG_SAVE_STDERR=true
LOG_TEE=true
mkdir -p "$LOG_DIR" "$ROOT/results/f-01c/$MODEL_MODE"
CHICAGO_TIMESTAMP="$(TZ=America/Chicago date "$LOG_TIMESTAMP_FORMAT")"
# Prevent filename collisions even if multiple runs begin in same second.
RUN_ID="$(TZ=America/Chicago date +%s)-$$"
OUTPUT_DIR="$ROOT/results/f-01c/$MODEL_MODE/${CHICAGO_TIMESTAMP}-truncation-review-${RUN_ID}"
LOG_FILE="$LOG_DIR/${LOG_PREFIX}-${CHICAGO_TIMESTAMP}-${RUN_ID}.log"
if [[ -e "$LOG_FILE" || -e "$OUTPUT_DIR" ]]; then
  echo 'ERROR: log/output collision' >&2; exit 1
fi
exec > >(tee -a "$LOG_FILE") 2>&1
printf '[F-01c %s] Chicago=%s MODE=truncation-review FILTER=%s OUTPUT=%s LOG=%s\n' "$VERSION" "$(TZ=America/Chicago date --iso-8601=seconds)" "$PAIR_FILTER" "$OUTPUT_DIR" "$LOG_FILE"
printf '[F-01c %s] MAX_SEQ_LENGTH=%s FOCUSED_REMOVED_MAX=%s FOCUSED_REDUCTION_MAX=%s LOG_STDOUT=%s LOG_STDERR=%s LOG_TEE=%s\n' "$VERSION" "$MAX_SEQ_LENGTH" "$MAX_REMOVED_FOR_FOCUSED_REVIEW" "$MAX_TOKEN_REDUCTION_FOR_FOCUSED_REVIEW" "$LOG_SAVE_STDOUT" "$LOG_SAVE_STDERR" "$LOG_TEE"
"$PYTHON_BIN" -u "$SCRIPT" \
  --proposals "$PROPOSALS" --audit "$AUDIT" --output-dir "$OUTPUT_DIR" \
  --pair-filter "$PAIR_FILTER" --log-timestamp "$CHICAGO_TIMESTAMP" --version "$VERSION" \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --max-removed-for-focused-review "$MAX_REMOVED_FOR_FOCUSED_REVIEW" \
  --max-token-reduction-for-focused-review "$MAX_TOKEN_REDUCTION_FOR_FOCUSED_REVIEW"
printf '[F-01c %s] REVIEW COMPLETE: TRAINING APPROVAL=NO\n' "$VERSION"
