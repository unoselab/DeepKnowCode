#!/usr/bin/env bash
set -euo pipefail
ROOT="$(pwd)"
SCRIPT="${ROOT}/src/py2/d-05bh-build-python-negative-review.py"
if [[ ! -f "$SCRIPT" ]]; then
  echo "ERROR: missing $SCRIPT; execute from DeepKnowCode repository root" >&2
  exit 2
fi
exec python "$SCRIPT" --repo-root "$ROOT" "$@"
