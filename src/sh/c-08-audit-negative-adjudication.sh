#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${C08_PYTHON:-$HOME/miniconda3/envs/deepknowcode/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi
# Standalone wrapper; does not invoke other shell wrappers.
exec "$PYTHON_BIN" "$PROJECT_ROOT/src/py/c-08-audit-negative-adjudication.py" \
  --review-csv "$PROJECT_ROOT/results/c-07-review/c-07-adjudication-2026-11-07.csv" \
  --c06 "$PROJECT_ROOT/results/c-06/20261103-140738-251117" \
  "$@"
