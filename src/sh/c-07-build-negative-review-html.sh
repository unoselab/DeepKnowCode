#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
export TZ=America/Chicago
PYTHON="$HOME/miniconda3/envs/deepknowcode/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$(command -v python3 || true)"
fi
if [[ -z "$PYTHON" || ! -x "$PYTHON" ]]; then
    echo 'C-07 ERROR: Python 3 not found' >&2
    exit 2
fi
exec "$PYTHON" "$PROJECT_ROOT/src/py/c-07-build-negative-review-html.py" --root "$PROJECT_ROOT" "$@"
