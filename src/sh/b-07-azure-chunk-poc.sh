#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NICAD="$ROOT/tools/nicad-7.0.1-linux-x86_64"
STAMP="$(TZ=America/Chicago date +%Y%m%d-%H%M%S)-$$"
OUT="$ROOT/results/b-07/azure-chunk-poc-v5/$STAMP"
STAGE="b0720261101233600604623"
XML="$NICAD/nicadclones/$STAGE/${STAGE}_functions.xml"
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" "$ROOT/src/py/b-07-azure-chunk-poc.py" --root "$ROOT" --source-xml "$XML" --out "$OUT" "$@"
