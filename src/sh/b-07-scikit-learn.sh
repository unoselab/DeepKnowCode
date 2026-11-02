#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
command -v flock >/dev/null || { echo "flock missing" >&2; exit 2; }
mkdir -p "$ROOT/logs/b-07"
exec 9>"$ROOT/logs/b-07/nicad.lock"
flock -n 9 || { echo "Another B-07 NiCad process is active" >&2; exit 6; }
"$PYTHON_BIN" -u "$ROOT/src/py/b-07-scikit-manifest.py" --root "$ROOT"
MANIFEST="results/b-07/subject-manifest-v9.csv"
if [[ "${1:-}" == "--dry-run" ]]; then
  "$PYTHON_BIN" -u "$ROOT/src/py/b-07-multi-project.py" --root "$ROOT" --manifest "$MANIFEST" --project scikit-learn --dry-run
  exit $?
fi
if (( $# > 0 )); then echo "Usage: bash src/sh/b-07-scikit-learn.sh [--dry-run]" >&2; exit 2; fi
STAMP="$(TZ=America/Chicago date +%Y%m%d-%H%M%S)-$$"
LOG="$ROOT/logs/b-07/scikit-learn-v9-$STAMP.log"
set +e
"$PYTHON_BIN" -u "$ROOT/src/py/b-07-multi-project.py" --root "$ROOT" --manifest "$MANIFEST" --project scikit-learn 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
exit "$rc"
