#!/usr/bin/env bash
# C-10 standalone server wrapper; no conda activation or wrapper chaining.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/deepknowcode/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then PYTHON_BIN="$(command -v python3)"; fi
exec "$PYTHON_BIN" "$PROJECT_ROOT/src/py/c-10-construct-corrected-python-dataset.py" \
  --c04 "$PROJECT_ROOT/results/c-04/20261103-193519-475848" \
  --c09 "$PROJECT_ROOT/results/c-09/20261106-213122-749426" \
  "$@"
