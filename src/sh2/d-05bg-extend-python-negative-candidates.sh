#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "$ROOT/src/py2/d-05bf-extend-python-negative-candidates.py" --root "$ROOT" "$@"
