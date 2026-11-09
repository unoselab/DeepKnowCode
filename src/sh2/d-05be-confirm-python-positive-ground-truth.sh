#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${PYTHON_BIN:-python}" -u "$ROOT/src/py2/d-05be-confirm-python-positive-ground-truth.py" --root "$ROOT" "$@"
