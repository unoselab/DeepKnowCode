#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT/src/py2/e-01c-train-standard-mixed-lora.py"
[[ -f "$PY" ]] || { echo "Missing E-01c Python: $PY" >&2; exit 2; }
[[ $# -gt 0 ]] || { echo 'Usage: bash src/sh2/e-01c-train-standard-mixed-lora.sh --preflight|--prepare|--train|--verify [--run-dir PATH]' >&2; exit 2; }
MODE="$1"; shift
case "$MODE" in
  --preflight) CMD=preflight ;;
  --prepare) CMD=prepare ;;
  --train) CMD=train ;;
  --verify) CMD=verify ;;
  *) echo "Unsupported mode: $MODE" >&2; exit 2 ;;
esac
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec python "$PY" "$CMD" --root "$ROOT" --gpus "${E01C_GPUS:-0,1,2}" "$@"
