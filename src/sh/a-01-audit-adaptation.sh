#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="/home/user1-system11/research_dream/llm-clone/Task/Clone-Detection-BigCloneBench"

python "$ROOT/src/py/a-01-audit-adaptation.py" \
  --data-root "$DATA_ROOT"
