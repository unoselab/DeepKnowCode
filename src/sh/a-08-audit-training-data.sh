#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$ROOT/logs" "$ROOT/results/a-08"
LOG="$ROOT/logs/a-08-audit-training-data-$STAMP.log"
OUT="$ROOT/results/a-08/$STAMP"
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
printf '[A-08] Start %s\n' "$(date -Iseconds)"
set +e
CMD=(python "$ROOT/src/py/a-08-audit-training-data.py" --output "$OUT")
if [[ "${A08_STAGE:-0}" == "1" ]]; then CMD+=(--stage); fi
if [[ "${A08_STRICT_FUNCTION_SPLIT:-0}" == "1" ]]; then CMD+=(--strict-function-split); fi
"${CMD[@]}"
STATUS=$?
set -e
printf '[A-08] exit=%s log=%s output=%s\n' "$STATUS" "$LOG" "$OUT"
exit "$STATUS"
