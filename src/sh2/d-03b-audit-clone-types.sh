#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TZ=America/Chicago
TIMESTAMP="$(date +%Y%m%d-%H%M%S)-$$"
RUN="$ROOT/results/d-03b/$TIMESTAMP-clone-types"
mkdir -p "$ROOT/results/d-03b"
if [[ -e "$RUN" ]]; then echo "[d-03b TYPE AUDIT] Run exists: $RUN" >&2; exit 1; fi
echo "[d-03b TYPE AUDIT] run=$RUN timestamp_chicago=$(date -Iseconds)"
python "$ROOT/src/py2/d-03b-audit-clone-types.py" --root "$ROOT" --run-dir "$RUN" "$@"
