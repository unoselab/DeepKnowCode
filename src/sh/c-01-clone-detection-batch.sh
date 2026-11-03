#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$ROOT/logs/c-01"
exec 9>"$ROOT/logs/c-01/nicad.lock"
flock -n 9 || { echo "Another C-01 run is active" >&2; exit 6; }
"${PYTHON_BIN:-python}" -u "$ROOT/src/py/c-01-batch.py" --root "$ROOT" "$@"
