#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$ROOT/results/a-03a" "$ROOT/logs"
python "$ROOT/src/py/a-03-check-model-readiness.py" --output-dir "$ROOT/results/a-03a" "$@"
