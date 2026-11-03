#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec "${PYTHON_BIN:-python}" -u "$ROOT/src/py/c-02-check-clone-results-batch.py" --root "$ROOT" "$@"
