#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOURNAL_GRAPH2="${JOURNAL_GRAPH2:-$HOME/research_dream/2026Journal/graph2}"
python "$ROOT/src/py/a-02-audit-paper-figures.py" \
  --journal-graph2 "$JOURNAL_GRAPH2" \
  --v2-results "$ROOT/results/a-02" \
  --output-dir "$ROOT/results/a-02-v3" "$@"
