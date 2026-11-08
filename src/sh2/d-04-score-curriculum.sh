#!/usr/bin/env bash
# D-04 v4: independently executed diagnostics; preserves original --verify/--score.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:---verify}"
shift || true
case "$MODE" in
  --verify) PYFILE="$ROOT/src/py2/d-04-score-curriculum.py"; COMMAND=verify ;;
  --score) PYFILE="$ROOT/src/py2/d-04-score-curriculum.py"; COMMAND=score ;;
  --validate-structural) PYFILE="$ROOT/src/py2/d-04-validate-structural.py"; COMMAND='' ;;
  --diagnose) PYFILE="$ROOT/src/py2/d-04-diagnose-curriculum.py"; COMMAND='' ;;
  *) echo "Usage: bash src/sh2/d-04-score-curriculum.sh --verify|--score|--diagnose|--validate-structural [Python options]" >&2;exit 2 ;;
esac
mkdir -p "$ROOT/logs/d-04"
export TZ=America/Chicago
STAMP="$(date +%Y%m%d-%H%M%S)"
LOGFILE="$ROOT/logs/d-04/d-04-${STAMP}-$$.log"
cd "$ROOT"
if [[ "$MODE" == --diagnose || "$MODE" == --validate-structural ]];then
  python "$PYFILE" --root "$ROOT" "$@" 2>&1 | tee "$LOGFILE"
else
  python "$PYFILE" "$COMMAND" --root "$ROOT" "$@" 2>&1 | tee "$LOGFILE"
fi
