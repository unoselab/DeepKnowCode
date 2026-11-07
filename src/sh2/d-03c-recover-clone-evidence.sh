#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TZ=America/Chicago
STAMP="$(date +%Y%m%d-%H%M%S)-$$"
RUN_DIR="$ROOT/results/d-03c/${STAMP}-detector-types"
mkdir -p "$RUN_DIR"
printf '[D-03c v4] timestamp_chicago=%s run=%s\n' "$(date --iso-8601=seconds)" "$RUN_DIR"
python "$ROOT/src/py2/d-03c-recover-clone-evidence.py" --root "$ROOT" --run-dir "$RUN_DIR" "$@"
