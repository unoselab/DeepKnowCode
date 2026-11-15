#!/usr/bin/env bash
# Standalone read-only v2 dataset comparison. Chicago timestamped unique logs.
set -Eeuo pipefail
export TZ=America/Chicago
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PYTHON_BIN="python"
VERSION="v2"
PYFILE="$ROOT/src/py/compare-prepared-datasets.py"
FROZEN_DIR="$ROOT/results/f-01c/codellama-7b/2026-1112-115525-prepare/prepared"
CURRENT_DIR="$ROOT/data/training/prepared/balanced_multilingual"
MAX_EXAMPLES=15
LOG_DIR="$ROOT/logs/dataset-compare"
LOG_PREFIX="prepared-compare-v2"
LOG_TIMESTAMP_FORMAT='+%Y-%m%d-%H%M%S'
LOG_TEE_OPTIONS='-a'
FOCUS_PAIRS=(
  'jackrabbit_818_918::jackrabbit_143_287'
  'ignite_72_43::ignite_89_103'
  'jackrabbit_177_378::jackrabbit_143_287'
  'libgdx_402_904::libgdx_350_707'
  'ignite_247_509::ignite_190_392'
  'derby_793_823::derby_539_168'
  'libgdx_364_736::libgdx_185_483'
  'ignite_84_87::ignite_1569_944'
)
[[ $# -eq 0 ]] || { echo "Usage: bash src/sh/compare-prepared-datasets.sh" >&2; exit 2; }
[[ -f "$PYFILE" && -d "$FROZEN_DIR" && -d "$CURRENT_DIR" ]] || { echo 'Missing Python engine or input directory' >&2; exit 2; }
mkdir -p "$LOG_DIR"
STAMP="$(TZ=America/Chicago date "$LOG_TIMESTAMP_FORMAT")"
UNIQUE="$(date +%s)-$$-$RANDOM"
LOG_FILE="$LOG_DIR/$LOG_PREFIX-$STAMP-$UNIQUE.log"
REPORT="$LOG_DIR/$LOG_PREFIX-$STAMP-$UNIQUE.json"
( set -C; : > "$LOG_FILE" ) || { echo 'Refusing overwrite' >&2; exit 2; }
exec > >(tee "$LOG_TEE_OPTIONS" "$LOG_FILE") 2>&1
START_SEC="$(date +%s)"
at_chicago() { TZ=America/Chicago date -Is; }
finish(){ local rc=$?; trap - EXIT ERR; printf 'FINAL_STATUS=%s EXIT=%s CHICAGO=%s ELAPSED_SECONDS=%s REPORT=%s LOG=%s\n' "$( [[ $rc -eq 0 ]] && echo PASS || echo FAIL )" "$rc" "$(at_chicago)" "$(( $(date +%s)-START_SEC ))" "$REPORT" "$LOG_FILE"; exit "$rc"; }
trap finish EXIT
echo "START VERSION=$VERSION CHICAGO=$(at_chicago)"
echo "INPUT_FROZEN=$FROZEN_DIR INPUT_CURRENT=$CURRENT_DIR"
ARGS=(--frozen-dir "$FROZEN_DIR" --current-dir "$CURRENT_DIR" --max-examples "$MAX_EXAMPLES" --report-file "$REPORT")
for pair in "${FOCUS_PAIRS[@]}"; do ARGS+=(--focus-pair "$pair"); done
echo "STEP=READ_ONLY_COMPARE START CHICAGO=$(at_chicago)"
"$PYTHON_BIN" -u "$PYFILE" "${ARGS[@]}"
echo "STEP=READ_ONLY_COMPARE PASS CHICAGO=$(at_chicago)"
