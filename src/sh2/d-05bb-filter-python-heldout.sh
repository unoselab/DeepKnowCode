#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${PYTHON_BIN:-python}" -u "$ROOT/src/py2/d-05bb-filter-python-heldout.py" --root "$ROOT" "$@"
