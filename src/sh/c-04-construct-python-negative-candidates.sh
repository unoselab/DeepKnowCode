#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "$ROOT/src/py/c-04-construct-python-negative-candidates.py" --root "$ROOT" "$@"
