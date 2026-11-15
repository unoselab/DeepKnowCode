#!/usr/bin/env bash
# J-01a v6: standalone, non-destructive Python code-collection conversion.
set -Eeuo pipefail
export TZ=America/Chicago
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
EXPERIMENT="j-01a"
VERSION="v6"
PYTHON_BIN="python"
ENGINE="$ROOT/src/py3/j-01a-evaluation-code-python.py"
SOURCE="$ROOT/data/evaluation/ground_truth/python/verified_pairs.jsonl"
REFERENCE="$ROOT/data/evaluation/code/java.jsonl"
OUTPUT="$ROOT/data/evaluation/code/python.jsonl"
EXPECTED_PAIRS="7942"
LEFT_FIELD="left"
RIGHT_FIELD="right"
CODE_FIELD="source_code"
SOURCE_ID_FIELD="id"
ID_POLICY="content-sha256"
ID_PREFIX="py-sha256:"
LABEL_FIELD="gold_label"
PAIR_ID_FIELD="pair_id"
EXPECTED_REFERENCE_KEYS='["id","code"]'
STRICT_PAIR_ID_UNIQUE="true"
LOG_DIR="$ROOT/logs/j-01a"
RESULT_DIR="$ROOT/results/j-01a"
LOG_PREFIX="j-01a-v6"
LOG_TIMESTAMP_FORMAT='+%Y-%m%d-%H%M%S'
LOG_TEE="true"
LOG_STDERR="true"
MODE="${1:-}"
case "$MODE" in --inspect|--convert) ;; *) echo "Usage: bash src/sh3/j-01a-evaluation-code-python.sh --inspect|--convert" >&2; exit 2;; esac
mkdir -p "$LOG_DIR" "$RESULT_DIR"
STAMP="$(date "$LOG_TIMESTAMP_FORMAT")"
RUN_DIR="$(mktemp -d "$RESULT_DIR/${STAMP}-XXXXXX")"
LOG_PATH="$LOG_DIR/${LOG_PREFIX}-${STAMP}-$$-${RUN_DIR##*-}.log"
( set -o noclobber; : > "$LOG_PATH" )
START_EPOCH="$(date +%s)"
exec > >(tee -a "$LOG_PATH") 2>&1
finish() {
  local rc=$?; trap - EXIT
  local elapsed=$(( $(date +%s) - START_EPOCH ))
  if ((rc == 0)); then
    echo "[$(date -Iseconds)] PASS FINAL experiment=$EXPERIMENT version=$VERSION elapsed_sec=$elapsed exit=$rc"
  else
    echo "[$(date -Iseconds)] FAIL FINAL experiment=$EXPERIMENT version=$VERSION elapsed_sec=$elapsed exit=$rc"
  fi
  printf "LOG='%s' RUN='%s' OUTPUT='%s'\n" "$LOG_PATH" "$RUN_DIR" "$OUTPUT"
  exit "$rc"
}
trap finish EXIT
trap 'echo "[$(date -Iseconds)] FAIL step=conversion line=$LINENO"' ERR
echo "[$(date -Iseconds)] START experiment=$EXPERIMENT version=$VERSION mode=$MODE log=$LOG_PATH"
echo "[$(date -Iseconds)] START step=validate_files"
[[ -f "$ENGINE" && -f "$SOURCE" && -f "$REFERENCE" ]] 
echo "[$(date -Iseconds)] PASS step=validate_files"
if [[ "$MODE" == "--convert" ]]; then
  [[ ! -e "$OUTPUT" ]] || { echo "REFUSE_OVERWRITE: $OUTPUT" >&2; exit 3; }
fi
REPORT="$RUN_DIR/conversion_report.json"
echo "[$(date -Iseconds)] START step=$MODE"
"$PYTHON_BIN" "$ENGINE" \
  --mode "${MODE#--}" --source "$SOURCE" --reference "$REFERENCE" \
  --output "$OUTPUT" --report "$REPORT" \
  --expected-pairs "$EXPECTED_PAIRS" --left-field "$LEFT_FIELD" --right-field "$RIGHT_FIELD" \
  --code-field "$CODE_FIELD" --source-id-field "$SOURCE_ID_FIELD" \
  --id-policy "$ID_POLICY" --id-prefix "$ID_PREFIX" \
  --label-field "$LABEL_FIELD" --pair-id-field "$PAIR_ID_FIELD" \
  --expected-reference-keys "$EXPECTED_REFERENCE_KEYS" \
  --strict-pair-id-unique "$STRICT_PAIR_ID_UNIQUE"
echo "[$(date -Iseconds)] PASS step=$MODE"
