#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)-$$"
OUT="$ROOT/results/a-10/$STAMP"
LOG="$ROOT/logs/a-10-audit-split-independence-$STAMP.log"
mkdir -p "$OUT" "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1
printf '[A-10] Start %s\n' "$(date -Iseconds)"
set +e
python "$ROOT/src/py/a-10-audit-split-independence.py" --out "$OUT" "$@"
STATUS=$?
set -e
printf '[A-10] exit=%s log=%s output=%s\n' "$STATUS" "$LOG" "$OUT"
exit "$STATUS"
