#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python -u "$ROOT/src/py/c-01-clone-detection-result.py" --root "$ROOT" "$@"
