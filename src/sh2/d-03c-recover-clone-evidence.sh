#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TZ=America/Chicago
STAMP="$(date +%Y%m%d-%H%M%S)-$$"
OUT="$ROOT/results/d-03c/$STAMP-lineage-recovery"
if [[ -e "$OUT" ]]; then echo "[D-03c v2] run already exists: $OUT" >&2; exit 1; fi
printf '[D-03c v2] timestamp_chicago=%s run=%s\n' "$(date -Iseconds)" "$OUT"
python "$ROOT/src/py2/d-03c-recover-clone-evidence.py" --root "$ROOT" --run-dir "$OUT" "$@"
