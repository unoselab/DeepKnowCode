#!/usr/bin/env bash
# D-05e v1 standalone wrapper, independent of earlier shell wrappers.

# bash src/sh2/d-05e-paired-bootstrap.sh \
#   --input results/d-05c/20261109-004326-58663 \
#   --dry-run

set -Eeuo pipefail
export TZ=America/Chicago
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$ROOT/src/py2/d-05e-paired-bootstrap.py"
[[ -f "$SCRIPT" ]] || { echo "Missing D-05e Python script: $SCRIPT" >&2; exit 2; }
exec python "$SCRIPT" --repo-root "$ROOT" "$@"
