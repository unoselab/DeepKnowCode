#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="$(date -u +%Y%m%d-%H%M%S)-$$"
OUT="${C01_OUTPUT:-$ROOT/results/c-01/$STAMP}"
LOG_DIR="$ROOT/logs/c-01"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/$STAMP.log"
# Directly invoke Python programs; no other shell wrapper is called.
{
  python -u "$ROOT/src/py/c-01-compare-python-clone-detectors.py" --project-root "$ROOT" --output "$OUT"
  python -u "$ROOT/src/py/c-01-manual-inspection.py" \
    --stage "$ROOT/results/b-01/_staging/salt" --output "$OUT" "$@"
} 2>&1 | tee "$LOG"
echo "Log: $LOG"
