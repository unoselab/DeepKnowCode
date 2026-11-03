#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TZ=America/Chicago
exec python "${ROOT}/src/py/c-05-verify-python-negative-candidates.py" --root "${ROOT}" "$@"
