#!/usr/bin/env bash
# D-04 v6: preserves all earlier modes; adds CPU-only order / Gloo-DDP audits.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:---verify}"
shift || true
case "$MODE" in
  --verify) PYFILE="$ROOT/src/py2/d-04-score-curriculum.py"; COMMAND=verify ;;
  --score) PYFILE="$ROOT/src/py2/d-04-score-curriculum.py"; COMMAND=score ;;
  --validate-structural) PYFILE="$ROOT/src/py2/d-04-validate-structural.py"; COMMAND='' ;;
  --prepare-curriculum) PYFILE="$ROOT/src/py2/d-04-prepare-curriculum.py"; COMMAND='' ;;
  --audit-order) PYFILE="$ROOT/src/py2/d-04-audit-distributed-order.py"; COMMAND=audit-order ;;
  --audit-ddp) PYFILE="$ROOT/src/py2/d-04-audit-distributed-order.py"; COMMAND=audit-ddp ;;
  --diagnose) PYFILE="$ROOT/src/py2/d-04-diagnose-curriculum.py"; COMMAND='' ;;
  *) echo "Usage: bash src/sh2/d-04-score-curriculum.sh --verify|--score|--diagnose|--validate-structural|--prepare-curriculum|--audit-order|--audit-ddp [Python options]" >&2;exit 2 ;;
esac
mkdir -p "$ROOT/logs/d-04"
export TZ=America/Chicago
STAMP="$(date +%Y%m%d-%H%M%S)"
LOGFILE="$ROOT/logs/d-04/d-04-${STAMP}-$$.log"
cd "$ROOT"
if [[ "$MODE" == --audit-ddp ]];then
  python -m torch.distributed.run --standalone --nproc_per_node=3 "$PYFILE" "$COMMAND" --root "$ROOT" "$@" 2>&1 | tee "$LOGFILE"
elif [[ "$MODE" == --audit-order ]];then
  python "$PYFILE" "$COMMAND" --root "$ROOT" "$@" 2>&1 | tee "$LOGFILE"
elif [[ "$MODE" == --diagnose || "$MODE" == --validate-structural || "$MODE" == --prepare-curriculum ]];then
  python "$PYFILE" --root "$ROOT" "$@" 2>&1 | tee "$LOGFILE"
else
  python "$PYFILE" "$COMMAND" --root "$ROOT" "$@" 2>&1 | tee "$LOGFILE"
fi
