#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
if [[ -x "$HOME/miniconda3/envs/deepknowcode/bin/python" ]]; then
 PYTHON="$HOME/miniconda3/envs/deepknowcode/bin/python"
else
 PYTHON="${PYTHON:-python3}"
fi
exec "$PYTHON" "$PROJECT_ROOT/src/py/c-11a-audit-python-training-dataset.py" --root "$PROJECT_ROOT" "$@"
