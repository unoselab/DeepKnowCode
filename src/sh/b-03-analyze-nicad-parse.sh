#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$ROOT/logs" "$ROOT/results/b-03"
STAMP="$(date -u +%Y%m%d-%H%M%S)-$$"
LOG="$ROOT/logs/b-03-analyze-nicad-parse-$STAMP.log"
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" "$ROOT/src/py/b-03-analyze-nicad-parse.py" --root "$ROOT" "$@" 2>&1 | tee "$LOG"
exit "${PIPESTATUS[0]}"
