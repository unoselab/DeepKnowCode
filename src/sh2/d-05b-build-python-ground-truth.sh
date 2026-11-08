#!/usr/bin/env bash
# D-05b v4 standalone. Does NOT call earlier wrappers or delete any files.
set -euo pipefail
ROOT="$(pwd -P)"
SCRIPT="$ROOT/src/py2/d-05b-build-python-ground-truth.py"
TARGET="$ROOT/data/evaluation/ground_truth/python"
[[ -f "$SCRIPT" ]] || { echo "BLOCKED: missing $SCRIPT" >&2; exit 2; }
[[ -d "$ROOT/data/evaluation/ground_truth/python_legacy" ]] || { echo 'BLOCKED: missing python_legacy' >&2; exit 2; }
if [[ ! -e "$TARGET" ]]; then
  mkdir -- "$TARGET"
fi
[[ -d "$TARGET" && -z "$(ls -A "$TARGET")" ]] || { echo "BLOCKED: target is not empty: $TARGET" >&2; exit 2; }
TS="$(TZ=America/Chicago date +%Y%m%d-%H%M%S)-$$"
OUT="$ROOT/results/d-05b/python-ground-truth-audit-v4-$TS"
python "$SCRIPT" --root "$ROOT" --out "$OUT" --candidate-output "$TARGET" "$@"
