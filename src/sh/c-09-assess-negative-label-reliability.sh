#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${C09_PYTHON:-$HOME/miniconda3/envs/deepknowcode/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then PYTHON_BIN="$(command -v python3)"; fi
cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" "$PROJECT_ROOT/src/py/c-09-assess-negative-label-reliability.py" \
  --c05 "$PROJECT_ROOT/results/c-05/20261103-135539-677672" \
  --c08 "$PROJECT_ROOT/results/c-08/20261106-205327-891891" \
  "$@"
