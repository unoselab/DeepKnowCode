#!/usr/bin/env bash
# D-05f v1 standalone wrapper; does not call other shell wrappers.

# bash src/sh2/d-05f-mcnemar-project-comparison.sh \
#   --input results/d-05c/20261109-004326-58663 \
#   --dry-run

set -Eeuo pipefail
export TZ=America/Chicago
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$ROOT/src/py2/d-05f-mcnemar-project-comparison.py"
[[ -f "$SCRIPT" ]] || { echo "D-05f missing Python: $SCRIPT" >&2; exit 2; }
exec python "$SCRIPT" --root "$ROOT" "$@"
