#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
exec python3 "$ROOT/src/py/c-03-prepare-python-training-data.py" --root "$ROOT" "$@"
