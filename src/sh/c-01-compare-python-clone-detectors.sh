#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="$(date -u +%Y%m%d-%H%M%S)-$$"
OUT="${C01_OUTPUT:-$ROOT/results/c-01/$STAMP}"
exec python "$ROOT/src/py/c-01-compare-python-clone-detectors.py" \
  --project-root "$ROOT" --output "$OUT" "$@"
