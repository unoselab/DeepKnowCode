#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$ROOT/logs" "$ROOT/results/a-09"
LOG="$ROOT/logs/a-09-audit-provenance-$STAMP.log"
OUT="$ROOT/results/a-09/$STAMP"
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
printf '[A-09] Start %s\n' "$(date -Iseconds)"
set +e
python "$ROOT/src/py/a-09-audit-provenance.py" --out "$OUT" "$@"
STATUS=$?
set -e
printf '[A-09] exit=%s log=%s output=%s\n' "$STATUS" "$LOG" "$OUT"
exit "$STATUS"
