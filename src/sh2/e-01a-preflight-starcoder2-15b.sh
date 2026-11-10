#!/usr/bin/env bash
# E-01a v1 standalone wrapper: does not invoke any other shell wrappers.
set -Eeuo pipefail
export TZ=America/Chicago
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${PYTHON_BIN:-python}" -u "$ROOT/src/py2/e-01a-preflight-starcoder2-15b.py" --root "$ROOT" "$@"
