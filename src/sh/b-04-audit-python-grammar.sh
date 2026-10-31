#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$ROOT/logs"
STAMP="$(date -u +%Y%m%d-%H%M%S)-$$"
LOG="$ROOT/logs/b-04-audit-python-grammar-$STAMP.log"
PYTHON_BIN="${PYTHON_BIN:-python}"
set +e
"$PYTHON_BIN" "$ROOT/src/py/b-04-audit-python-grammar.py" --root "$ROOT" "$@" 2>&1 | tee "$LOG"
code=${PIPESTATUS[0]}
set -e
exit "$code"
