#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$ROOT/logs"
STAMP="$(date -u +%Y%m%d-%H%M%S)-$$"
LOG="$ROOT/logs/b-02-audit-nicad-parse-$STAMP.log"
command -v flock >/dev/null || { echo 'ERROR: flock unavailable' >&2; exit 2; }
mkdir -p "$ROOT/results/b-02"
exec 9>"$ROOT/results/b-02/.b-02.lock"
flock -n 9 || { echo 'ERROR: another B-02 probe is running' >&2; exit 3; }
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" "$ROOT/src/py/b-02-audit-nicad-parse.py" --root "$ROOT" "$@" 2>&1 | tee "$LOG"
exit "${PIPESTATUS[0]}"
