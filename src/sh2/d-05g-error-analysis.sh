#!/usr/bin/env bash
# Independent D-05g v1 CPU analysis wrapper. No other shell wrapper invoked.

# bash src/sh2/d-05g-error-analysis.sh \
#   --input results/d-05c/20261109-004326-58663 \
#   --dry-run

set -Eeuo pipefail
export TZ=America/Chicago
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${PYTHON_BIN:-python}" -u "$ROOT/src/py2/d-05g-error-analysis.py" --root "$ROOT" "$@"
