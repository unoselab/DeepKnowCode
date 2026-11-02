#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec python -u "$ROOT/src/py/c-02-check-clone-results.py" "$@"
