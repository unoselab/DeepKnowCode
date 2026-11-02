#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$ROOT/logs/c-01"
STAMP="$(TZ=America/Chicago date +%Y%m%d-%H%M%S)-$$"
exec 9>"$ROOT/logs/c-01/nicad.lock"
flock -n 9 || { echo "Another C-01 run is active" >&2; exit 6; }
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -u "$ROOT/src/py/c-01-clone-detection.py" --root "$ROOT" "$@" 2>&1 | tee "$ROOT/logs/c-01/${STAMP}.log"
exit "${PIPESTATUS[0]}"
