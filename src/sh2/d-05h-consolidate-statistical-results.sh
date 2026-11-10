#!/usr/bin/env bash
# D-05h standalone wrapper; does not invoke any other shell wrapper.

# bash src/sh2/d-05h-consolidate-statistical-results.sh --dry-run

set -Eeuo pipefail
export TZ=America/Chicago
export MPLBACKEND=Agg
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${PYTHON_BIN:-python}" -u "$ROOT/src/py2/d-05h-consolidate-statistical-results.py" --root "$ROOT" "$@"
