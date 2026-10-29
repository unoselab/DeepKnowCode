#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$ROOT/logs"
STAMP="$(date +%Y%m%d-%H%M%S)-$$"
LOG="$ROOT/logs/a-05a-stage-evaluation-data-$STAMP.log"
exec > >(tee "$LOG") 2>&1
trap 'rc=$?; echo "Exit=$rc; log='$LOG'"' EXIT
echo "Staging inputs in $ROOT/data/evaluation"
args=(--workspace "$ROOT")
if [[ "${A05A_REPLACE:-0}" == 1 ]]; then args+=(--replace); fi
if [[ -n "${A05A_RQ3_ROOT:-}" ]]; then args+=(--rq3-root "$A05A_RQ3_ROOT"); fi
if [[ -n "${A05A_RQ4_ROOT:-}" ]]; then args+=(--rq4-root "$A05A_RQ4_ROOT"); fi
python "$ROOT/src/py/a-05a-stage-evaluation-data.py" "${args[@]}"
