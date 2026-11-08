#!/usr/bin/env bash
# D-05b v1 independent standalone entrypoint.
set -euo pipefail
ROOT="$(pwd -P)"
SCRIPT="$ROOT/src/py2/d-05b-build-python-ground-truth.py"
if [[ ! -f "$SCRIPT" ]]; then echo "Missing script: $SCRIPT" >&2; exit 2; fi
TS="$(TZ=America/Chicago date +%Y%m%d-%H%M%S)-$$"
OUT="$ROOT/results/d-05b/python-ground-truth-audit-$TS"
python "$SCRIPT" --root "$ROOT" --out "$OUT" --candidate-output "$ROOT/data/evaluation/ground_truth/python"
