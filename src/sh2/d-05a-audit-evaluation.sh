#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT/src/py2/d-05a-audit-evaluation.py"
[[ -f "$PY" ]] || { echo "Missing $PY" >&2; exit 1; }
export TZ=America/Chicago
STAMP="$(TZ=America/Chicago date +%Y%m%d-%H%M%S)-$$"
OUT="$ROOT/results/d-05a/evaluation-inventory-$STAMP"
mkdir -p "$ROOT/logs/d-05a"
LOG="$ROOT/logs/d-05a/d-05a-$STAMP.log"
python "$PY" --root "$ROOT" --out "$OUT" "$@" 2>&1 | tee "$LOG"
