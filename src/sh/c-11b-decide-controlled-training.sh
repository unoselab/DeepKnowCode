#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/deepknowcode/bin/python}"
if [[ ! -x "$PYTHON" ]]; then PYTHON="$(command -v python3)"; fi
exec "$PYTHON" "$PROJECT_ROOT/src/py/c-11b-decide-controlled-training.py" --root "$PROJECT_ROOT" "$@"
