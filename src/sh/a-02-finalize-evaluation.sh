#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
python "$ROOT/src/py/a-02-finalize-evaluation.py" \
  --v2-results "$ROOT/results/a-02" \
  --v3-results "$ROOT/results/a-02-v3" \
  --output-dir "$ROOT/results/a-02-final" "$@"
