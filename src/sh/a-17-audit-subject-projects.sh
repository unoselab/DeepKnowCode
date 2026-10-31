#!/usr/bin/env bash
set -euo pipefail
ROOT="${A17_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA_ROOT="${A17_DATA_ROOT:-$ROOT/data/evaluation}"
STAMP="$(date +%Y%m%d-%H%M%S)-$$"
OUT="${A17_OUTPUT_DIR:-$ROOT/results/a-17/$STAMP}"
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/a-17-audit-subject-projects-$STAMP.log"
PYTHON="${A17_PYTHON:-python}"
echo "[A-17] root=$ROOT data=$DATA_ROOT output=$OUT log=$LOG"
"$PYTHON" "$ROOT/src/py/a-17-audit-subject-projects.py" --data-root "$DATA_ROOT" --output-dir "$OUT" --expected-projects 72 --expected-pairs 39002 2>&1 | tee "$LOG"
EXTRA=()
[[ -z "${A17_M1_PREPARED:-}" ]] || EXTRA+=(--m1-prepared "$A17_M1_PREPARED")
[[ -z "${A17_M2_PREPARED:-}" ]] || EXTRA+=(--m2-prepared "$A17_M2_PREPARED")
[[ -z "${A17_M3_PREPARED:-}" ]] || EXTRA+=(--m3-prepared "$A17_M3_PREPARED")
"$PYTHON" "$ROOT/src/py/a-17-audit-adaptation-data.py" --root "$ROOT" --output-dir "$OUT" "${EXTRA[@]}" 2>&1 | tee -a "$LOG"
