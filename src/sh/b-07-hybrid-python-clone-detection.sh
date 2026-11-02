#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$ROOT/logs/b-07"
STAMP="$(TZ=America/Chicago date +%Y%m%d-%H%M%S)-$$"
LOG="$ROOT/logs/b-07/$STAMP.log"
command -v flock >/dev/null || { echo 'flock missing' >&2; exit 2; }
exec 9>"$ROOT/logs/b-07/nicad.lock"
flock -n 9 || { echo 'Another B-07 NiCad process is active' >&2; exit 6; }
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -u "$ROOT/src/py/b-07-multi-project.py" --root "$ROOT" "$@" 2>&1 | tee "$LOG"
exit "${PIPESTATUS[0]}"
