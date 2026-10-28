#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RQ4_ROOT="${RQ4_ROOT:-/home/user1-system11/research_dream/2026JournalData/RQ4/c-evaluate_clone}"
python "$ROOT/src/py/a-02-audit-project-evaluation.py" \
  --rq4-root "$RQ4_ROOT" \
  --output-dir "$ROOT/results/a-02" \
  "$@"
