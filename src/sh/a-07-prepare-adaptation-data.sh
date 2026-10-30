#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$ROOT/logs" "$ROOT/results/a-07"
STAMP="$(date +%Y%m%d-%H%M%S)-$$"
LOG="$ROOT/logs/a-07-prepare-adaptation-data-${STAMP}.log"
OUT="$ROOT/results/a-07/$STAMP"
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
trap 'status=$?; echo "[A-07] exit=$status log=$LOG output=$OUT"' EXIT
MODE="${A07_MODE:-inventory}"
ARGS=(--workspace "$ROOT" --output-dir "$OUT" --mode "$MODE")
if [[ "$MODE" == inventory ]]; then
  RQ3="${A07_SEARCH_ROOT:-$HOME/research_dream/2026JournalData/RQ3}"
  ARGS+=(--search-root "$RQ3")
fi
if [[ "${A07_CONFIRM_PROVENANCE:-0}" == 1 ]]; then ARGS+=(--confirm-provenance); fi
# Extra arguments allow explicit --train-pairs and --code-corpus paths.
python "$ROOT/src/py/a-07-prepare-adaptation-data.py" "${ARGS[@]}" "$@"
