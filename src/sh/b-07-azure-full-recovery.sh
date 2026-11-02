#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:---prepare}"
if [ "$#" -gt 0 ]; then shift; fi
STAGE="b0720261101233600604623"
mkdir -p "$ROOT/logs/b-07"
exec 9>"$ROOT/logs/b-07/nicad.lock"
flock -n 9 || { echo 'Another NiCad job is active' >&2; exit 6; }
PYTHON_BIN="${PYTHON_BIN:-python}"
STAMP="$(TZ=America/Chicago date +%Y%m%d-%H%M%S)-$$"
case "$MODE" in
  --prepare)
    OUT="$ROOT/results/b-07/azure-v8-renaming/$STAMP"
    "$PYTHON_BIN" -u "$ROOT/src/py/b-07-azure-full-renaming.py" \
      --root "$ROOT" --stage-name "$STAGE" --output-dir "$OUT" "$@"
    ;;
  --finish)
    OUT="$ROOT/results/b-07/$STAMP/projects/azure-sdk-for-python"
    "$PYTHON_BIN" -u "$ROOT/src/py/b-07-hybrid-python-clone-detection.py" \
      --root "$ROOT" --project azure-sdk-for-python --azure-cache-stage "$STAGE" --out "$OUT" "$@"
    ;;
  *) echo 'Usage: b-07-azure-full-recovery.sh --prepare [--chunk-size N] | --finish [--per-file-timeout N]' >&2; exit 2;;
esac
